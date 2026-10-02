"""B5: 保存 PNG から視差を再計算し、(1) 推論帯なしの距離別精度、(2) 保存 npy との再現一致（全フレーム）、
(3) 推論帯ありの再計算値と公式集計（checks.evaluate_distance_accuracy.summarize_session）の一致、を 1 回の走査で出す。

定義はすべて既存関数に委ねる（独自の統計は足さない）:
  視差   : tools.analysis_common.preprocess_to → disparity_of_proc（Rust ネイティブ ZNCC・gate1 前の生値）
  ゲート : tools.analysis_common.apply_gates（gate1 |d|>DISP_MAX、band=(0.45,5.0) は推論帯）
  深度   : tools.analysis_common.depth_of（core.depth.compute_depth・EPS_DEN）
  統計   : tools.analysis_common.evaluate_session（pooled・gate2 0<z<50000・全画面）
  照合   : checks.verify_clip_replay._compare（float32・NaN 位置・有限値のビット一致）
  公式   : checks.evaluate_distance_accuracy.summarize_session（保存 npy を集計）
係数は meta.json の params_path を core.params.load_params で読む（runner・verify_clip_replay と同じ経路。
CSV を float() で読むと a_pos・u_neg・a_neg が 1 ulp ずれるので使わない）。キャッシュは使わない（メモリ上で計算）。

実行（開発リポの root）:
  uv run python -u <this>.py --manifest outputs/distance_manifest_b5.csv --skip-first 10 \
      --replay-only b5_scene_01 --out-dir <出力先>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import cv2
import numpy as np

import config
from checks.evaluate_distance_accuracy import load_manifest, summarize_session
from checks.verify_clip_replay import _compare
from core.params import DepthParams, load_params
from tools.analysis_common import (DispSettings, apply_gates, depth_of, disparity_of_proc, evaluate_session,
                                   preprocess_to)

STAT_KEYS = ["frames_used", "n_valid_total", "valid_rate", "n_valid_min", "n_valid_max", "mean_est_mm",
             "mean_err_mm", "mae_mm", "std_err_mm", "rmse_mm", "median_err_mm", "frame_mean_est_std_mm"]
OFF_COLUMNS = ["session", "gt_mm", "gate", "frames_used", "frames_total", "roi", "n_roi_px_per_frame",
               "n_valid_total", "valid_rate", "n_valid_min", "n_valid_max", "mean_est_mm", "mean_err_mm", "mae_mm",
               "std_err_mm", "rmse_mm", "median_err_mm", "frame_mean_est_std_mm", "band_on_selfcheck",
               "params_path", "note"]
REPLAY_COLUMNS = ["session", "frames_png", "frames_npy", "frames_checked", "frames_match", "first_fail", "first_fail_msg",
                  "params_path"]
CONFIG_KEYS = {"IMAGE_SIZE": list(config.IMAGE_SIZE), "KERNEL_SIZE": config.KERNEL_SIZE,
               "SEARCH_RANGE_X": config.SEARCH_RANGE_X, "EDGE_TH": config.EDGE_TH, "DISP_MAX": config.DISP_MAX,
               "INFER_BAND_LOW": config.INFER_BAND_LOW, "INFER_BAND_HIGH": config.INFER_BAND_HIGH}


def session_params(session: str) -> tuple[DepthParams, str]:
    meta = json.loads((Path(config.OUTPUTS_DIR) / session / "meta.json").read_text(encoding="utf-8"))
    params = load_params(meta["params_path"])
    if params != DepthParams.from_dict(meta["params"]):
        raise ValueError(f"{session}: load_params(params_path) と meta.json の params が一致しません")
    for k, v in CONFIG_KEYS.items():
        mv = meta["config"].get(k)
        if (list(mv) if isinstance(mv, (list, tuple)) else mv) != v:
            raise ValueError(f"{session}: meta.json の config.{k}={mv!r} が現行 config {v!r} と不一致")
    return params, meta["params_path"]


def run_session(session: str, gt_mm: int | None, *, skip_first: int, max_frames: int | None,
                s: DispSettings, band: tuple[float, float]) -> tuple[dict, list[dict]]:
    cap_dir = Path(config.CAPTURES_DIR) / session
    out_dir = Path(config.OUTPUTS_DIR) / session
    pngs = sorted(p for p in cap_dir.glob("??????.png") if p.stem.isdigit())
    npys = sorted(p for p in out_dir.glob("??????.npy") if p.stem.isdigit())
    if [p.stem for p in pngs] != [p.stem for p in npys]:
        raise ValueError(f"{session}: PNG と npy の連番が一致しません ({len(pngs)} vs {len(npys)})")
    params, params_path = session_params(session)
    check = pngs if max_frames is None else pngs[:skip_first + max_frames]
    n_ok, first_fail, first_msg = 0, "", ""
    disps: list[np.ndarray] = []
    t0 = time.perf_counter()
    for k, p in enumerate(check):
        raw = cv2.imread(str(p))
        if raw is None:
            raise ValueError(f"{p}: png を読めません")
        disp = disparity_of_proc(preprocess_to(raw, s.image_size), s)
        z_on = depth_of(apply_gates(disp, disp_max=s.disp_max, band=band), params)
        ok, msg = _compare(np.load(str(out_dir / f"{p.stem}.npy")), z_on)
        if ok:
            n_ok += 1
        elif not first_fail:
            first_fail, first_msg = p.stem, msg
        if gt_mm is not None and k >= skip_first:
            disps.append(disp)
    replay = {"session": session, "frames_png": len(pngs), "frames_npy": len(npys), "frames_checked": len(check),
              "frames_match": n_ok, "first_fail": first_fail, "first_fail_msg": first_msg, "params_path": params_path}
    print(f"[replay] {session}: {n_ok}/{len(check)} match ({time.perf_counter() - t0:.1f} s)", flush=True)
    rows: list[dict] = []
    if gt_mm is None:
        return replay, rows
    stack = np.stack(disps)
    w, h = s.image_size
    common = {"session": session, "gt_mm": gt_mm, "frames_total": len(pngs), "roi": f"0,0,{w},{h}",
              "n_roi_px_per_frame": w * h, "params_path": params_path}
    m_off = evaluate_session(stack, gt_mm, params, disp_max=s.disp_max, band=None)
    m_on = evaluate_session(stack, gt_mm, params, disp_max=s.disp_max, band=band)
    official = summarize_session(out_dir, gt_mm, roi=(0, 0, w, h), skip_first=skip_first, max_frames=max_frames)
    diffs = [k for k in STAT_KEYS if not (m_on[k] == official[k] or (np.isnan(m_on[k]) and np.isnan(official[k])))]
    selfcheck = "exact" if not diffs else "DIFF:" + "|".join(diffs)
    print(f"[stats] {session}: band_off MAE={m_off['mae_mm']:.3f} bias={m_off['mean_err_mm']:+.3f} "
          f"valid={m_off['valid_rate']:.4f} | band_on(recomputed) MAE={m_on['mae_mm']:.3f} selfcheck={selfcheck}",
          flush=True)
    rows.append({**common, "gate": "band_off(gate1 only)", **{k: m_off[k] for k in STAT_KEYS},
                 "band_on_selfcheck": selfcheck})
    return replay, rows


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as f:   # 排他作成（上書きしない）・csv は \r\n（既存の B2 CSV と同じ）
        wr = csv.DictWriter(f, fieldnames=columns)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in columns})


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--skip-first", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None, help="試験用。本番は指定しない")
    ap.add_argument("--replay-only", nargs="*", default=[], help="再現一致だけを見るセッション（例 b5_scene_01）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="b5_20261001")
    args = ap.parse_args()
    s = DispSettings()
    band = (config.INFER_BAND_LOW, config.INFER_BAND_HIGH)
    if s.key() != "K25_S41_T20" or s.disp_max != config.DISP_MAX:
        print(f"[ERROR] 設定が B5 と異なります: {s.key()} disp_max={s.disp_max}")
        return 2
    targets = [(sess, gt, note) for sess, gt, note in load_manifest(Path(args.manifest))]
    off_rows, replay_rows = [], []
    for sess, gt, note in targets:
        rep, rows = run_session(sess, gt, skip_first=args.skip_first, max_frames=args.max_frames, s=s, band=band)
        for r in rows:
            r["note"] = note
        replay_rows.append(rep)
        off_rows += rows
    for sess in args.replay_only:
        rep, _ = run_session(sess, None, skip_first=args.skip_first, max_frames=args.max_frames, s=s, band=band)
        replay_rows.append(rep)
    out = Path(args.out_dir)
    write_csv(out / f"distance_accuracy_{args.tag}_bandoff.csv", OFF_COLUMNS, off_rows)
    write_csv(out / f"replay_check_{args.tag}.csv", REPLAY_COLUMNS, replay_rows)
    all_match = all(r["frames_match"] == r["frames_checked"] for r in replay_rows)
    all_exact = all(r["band_on_selfcheck"] == "exact" for r in off_rows)
    print(f"\nreplay all match: {all_match} / band_on selfcheck all exact: {all_exact}")
    return 0 if (all_match and all_exact) else 1


if __name__ == "__main__":
    sys.exit(main())
