"""B5 の補足集計（完全性の基準と板の端の感度）。

1. 完全性（撮り直しの判定に使う基準・精度の値は使わない）:
   - PNG と npy の連番の一致・枚数（先頭 10 を除いて ≥ 60）
   - meta.json が採用 16 セッションで started_at 以外同一、かつ config 7 キーと係数（core.params.load_params）が現行と一致
   - 板の被覆（ノイズ板のセッションのみ）: 全フレームで元画像の左右端の列（x=100..109・1170..1179）の平均輝度。
     左右の黒帯（x<100・x>=1180）の内側に板の外（暗部）が写り込んでいないかを見る。閾値 40（参考値）
   - runner ログの `processed=N ... valid=V` が npy N の非 NaN 画素数と一致すること（ログとセッションの対応の確認。
     PowerShell の折り返しで分かれた桁は結合する）
   - gate2（0 < z < 50000）で落ちる画素数と、有効画素の最大深度（外れ値の確認）
2. (a) 板の端の感度: 端の暗部が ZNCC の窓（半幅 12＋探索 ±20 ＝ ±32 px）を通じて影響し得る列を除いた ROI で
   公式集計（summarize_session）をもう一度出す。
3. (b) 不採用テイク（outputs/_archive_20261002/ の 2 本）の参考集計（全画面と ROI）・E2E（ログを UTF-16 のまま読む）。
   正式値には使わない。

ROI は処理解像度 528×352 の画素座標・右下排他。元画像 x → 処理画像 x' = (x − 100) × 528 / 1080。
実行（開発リポの root）: uv run python -u tools/record_repro/2026-10-01/b5_supplementary.py --out-dir <出力先> --tag b5_20261001
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import cv2
import numpy as np

import config
from checks.evaluate_distance_accuracy import _LOG_LINE, summarize_session
from core.params import DepthParams, load_params

ADOPTED = [f"b5_dist_{mm}" for mm in (1000, 1150, 1250, 1450, 1650, 1850, 2050, 2250, 2450, 2650, 2850, 3050,
                                      3450, 3750, 4000)]
SCENE = "b5_scene_01"
ARCHIVE = {"b5_dist_3450": ("_archive_20261002/b5_dist_3450", 3450, "3450 mm 1 回目（板の左端 55 px）"),
           "b5_dist_4000_retake": ("_archive_20261002/b5_dist_4000_retake", 4000, "4000 mm 撮り直し（板の左端 31 px）")}
# 端の暗部（元画像 x）→ 処理画像の列 → ±32 px の影響を除く ROI（少し余裕を取る）
ROI_SENS = {
    "b5_dist_3750": ((35, 0, 528, 352), "左端の暗部 x=100..101 → x'≈0。影響は x'≤32（評価領域 x'≥33 の外）。余裕を取り x'≥35"),
    "b5_dist_4000": ((0, 0, 484, 352), "右端の暗部 x=1160..1179 → x'≈518..527。影響は x'≥486。余裕を取り x'<484"),
    "_archive_20261002/b5_dist_3450": ((60, 0, 528, 352), "左端の暗部 x=100..154 → x'≈0..26。影響は x'≤58。x'≥60"),
    "_archive_20261002/b5_dist_4000_retake": ((48, 0, 528, 352), "左端の暗部 x=100..130 → x'≈0..14。影響は x'≤46。x'≥48"),
}
CONFIG_KEYS = {"IMAGE_SIZE": list(config.IMAGE_SIZE), "KERNEL_SIZE": config.KERNEL_SIZE,
               "SEARCH_RANGE_X": config.SEARCH_RANGE_X, "EDGE_TH": config.EDGE_TH, "DISP_MAX": config.DISP_MAX,
               "INFER_BAND_LOW": config.INFER_BAND_LOW, "INFER_BAND_HIGH": config.INFER_BAND_HIGH}
STAT_KEYS = ["frames_used", "frames_total", "roi", "n_valid_total", "valid_rate", "n_valid_min", "n_valid_max",
             "mean_est_mm", "mean_err_mm", "mae_mm", "std_err_mm", "rmse_mm", "median_err_mm",
             "frame_mean_est_avg_mm", "frame_mean_est_std_mm"]
_VALID = re.compile(r"processed=(\d+) \(frame id=\d+\): [\d.]+ ms \(~[\d.]+ fps\), valid=(\d+)")


def read_log(path: Path) -> str:
    raw = path.read_bytes()
    text = raw.decode("utf-16") if raw[:2] == b"\xff\xfe" else raw.decode("utf-8")
    # PowerShell のコンソール幅で valid=NNNN の桁が次の行に折り返された箇所を結合する。
    # 次の行が数字だけのときに限る（次のログ行の日付まで結合しないため）
    return re.sub(r"(valid=\d+)\r?\n(\d+)(?=\r?\n|$)", r"\1\2", text)


def e2e_from_text(text: str) -> dict:
    recs = []
    for line in text.splitlines():
        m = _LOG_LINE.search(line)
        if m:
            recs.append((datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S,%f"), int(m.group("processed")),
                         int(m.group("fid"))))
    (t0, p0, f0), (t1, p1, f1) = recs[0], recs[-1]
    span = (t1 - t0).total_seconds()
    return {"e2e_fps": (p1 - p0) / span, "camera_fps_est": (f1 - f0) / span, "drop_ratio": 1 - (p1 - p0) / (f1 - f0),
            "log_span_s": span, "n_log_lines": len(recs)}


def completeness(session: str, log_path: Path, *, noise_board: bool, ref_meta: dict) -> dict:
    cap = Path(config.CAPTURES_DIR) / session
    out = Path(config.OUTPUTS_DIR) / session
    pngs = sorted(p for p in cap.glob("??????.png") if p.stem.isdigit())
    npys = sorted(p for p in out.glob("??????.npy") if p.stem.isdigit())
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    params_now = load_params(config.PARAMS_PATH)
    meta_cfg_ok = all((list(meta["config"].get(k)) if isinstance(meta["config"].get(k), (list, tuple))
                       else meta["config"].get(k)) == v for k, v in CONFIG_KEYS.items())
    strip = {k: v for k, v in meta.items() if k != "started_at"}
    ref_strip = {k: v for k, v in ref_meta.items() if k != "started_at"}
    row = {"session": session, "frames_png": len(pngs), "frames_npy": len(npys),
           "seq_match": [p.stem for p in pngs] == [p.stem for p in npys],
           "frames_after_skip10": max(0, len(npys) - 10), "started_at": meta.get("started_at"),
           "meta_config_matches_current": meta_cfg_ok,
           "meta_params_match_current_load_params": DepthParams.from_dict(meta["params"]) == params_now,
           "meta_identical_to_b5_dist_1000_except_started_at": strip == ref_strip}
    # 板の被覆（全フレーム）
    if noise_board:
        lmin, rmin, n_dark = 255.0, 255.0, 0
        for p in pngs:
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            lm, rm = float(im[:, 100:110].mean()), float(im[:, 1170:1180].mean())
            lmin, rmin = min(lmin, lm), min(rmin, rm)
            n_dark += int(lm < 40 or rm < 40)
        row.update({"edge_left_mean_min": round(lmin, 1), "edge_right_mean_min": round(rmin, 1),
                    "frames_with_dark_edge": n_dark})
    # ログの valid=V と npy の非 NaN 画素数
    text = read_log(log_path)
    pairs = [(int(a), int(b)) for a, b in _VALID.findall(text)]
    mism = [(n, v) for n, v in pairs if n < len(npys) and int(np.count_nonzero(~np.isnan(np.load(npys[n])))) != v]
    row.update({"log_valid_lines": len(pairs), "log_valid_mismatch": len(mism),
                "log_valid_beyond_npy": sum(1 for n, _ in pairs if n >= len(npys))})
    # gate2 で落ちる画素と有効画素の最大深度（先頭 10 を除く）
    n_nonnan, n_g2_drop, n_neg, n_ge_max, zmax = 0, 0, 0, 0, 0.0
    for p in npys[10:]:
        z = np.load(p)
        fin = ~np.isnan(z)
        ok = fin & (z > config.DEPTH_MIN) & (z < config.DEPTH_MAX)
        n_nonnan += int(fin.sum())
        n_g2_drop += int((fin & ~ok).sum())
        n_neg += int((fin & (z <= config.DEPTH_MIN)).sum())
        n_ge_max += int((fin & (z >= config.DEPTH_MAX)).sum())
        if ok.any():
            zmax = max(zmax, float(z[ok].max()))
    row.update({"nonnan_px_after_skip10": n_nonnan, "gate2_dropped_px": n_g2_drop, "gate2_dropped_le0": n_neg,
                "gate2_dropped_ge50000": n_ge_max, "valid_depth_max_mm": round(zmax, 1)})
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with path.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="b5_20261001")
    ap.add_argument("--log-dir", default="outputs/logs", help="採用セッションのログ（UTF-16 原本でも UTF-8 写しでも可）")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    ref_meta = json.loads((Path(config.OUTPUTS_DIR) / "b5_dist_1000" / "meta.json").read_text(encoding="utf-8"))

    comp = []
    for s in ADOPTED + [SCENE]:
        r = completeness(s, Path(args.log_dir) / f"{s}.txt", noise_board=(s != SCENE), ref_meta=ref_meta)
        r["take"] = "adopted"
        comp.append(r)
        print("[complete]", json.dumps(r, ensure_ascii=False), flush=True)
    for name, (rel, _gt, desc) in ARCHIVE.items():
        r = completeness(rel, Path(config.OUTPUTS_DIR) / "_archive_20261002" / "logs" / f"{name}.txt",
                         noise_board=True, ref_meta=ref_meta)
        r["take"] = f"archived (not used): {desc}"
        comp.append(r)
        print("[complete]", json.dumps(r, ensure_ascii=False), flush=True)
    write_csv(out_dir / f"completeness_{args.tag}.csv", comp)

    sens = []
    gts = {"b5_dist_3750": 3750, "b5_dist_4000": 4000, "_archive_20261002/b5_dist_3450": 3450,
           "_archive_20261002/b5_dist_4000_retake": 4000}
    for sess, (roi, why) in ROI_SENS.items():
        d = Path(config.OUTPUTS_DIR) / sess
        for label, r_ in (("full", (0, 0, 528, 352)), ("edge_excluded", roi)):
            m = summarize_session(d, gts[sess], roi=r_, skip_first=10, max_frames=None)
            row = {"session": sess, "take": "archived (not used)" if sess.startswith("_archive") else "adopted",
                   "roi_label": label, "roi_reason": why if label != "full" else "",
                   **{k: m[k] for k in STAT_KEYS}, "valid_px_per_frame": m["n_valid_total"] / m["frames_used"]}
            sens.append(row)
            print(f"[roi] {sess} {label} roi={m['roi']} valid/frame={row['valid_px_per_frame']:.0f} "
                  f"bias={m['mean_err_mm']:+.2f} MAE={m['mae_mm']:.2f} median_err={m['median_err_mm']:+.2f} "
                  f"RMSE={m['rmse_mm']:.1f}", flush=True)
    write_csv(out_dir / f"roi_sensitivity_and_archived_{args.tag}.csv", sens)

    e2e = []
    for name, (rel, _gt, desc) in ARCHIVE.items():
        r = {"log": f"_archive_20261002/logs/{name}.txt", "desc": desc,
             **e2e_from_text(read_log(Path(config.OUTPUTS_DIR) / "_archive_20261002" / "logs" / f"{name}.txt"))}
        e2e.append(r)
        print("[e2e archived]", json.dumps(r, ensure_ascii=False), flush=True)
    write_csv(out_dir / f"e2e_archived_{args.tag}.csv", e2e)

    ok = all(r["seq_match"] and r["frames_after_skip10"] >= 60 and r["meta_config_matches_current"]
             and r["meta_params_match_current_load_params"] and r["log_valid_mismatch"] == 0
             for r in comp if r["take"] == "adopted")
    ok_meta_same = all(r["meta_identical_to_b5_dist_1000_except_started_at"] for r in comp if r["take"] == "adopted")
    dark = {r["session"]: r.get("frames_with_dark_edge") for r in comp if r["take"] == "adopted" and r["session"] != SCENE}
    print(f"\nadopted completeness (seq/frames/meta/log-valid) all ok: {ok} / meta identical: {ok_meta_same}")
    print(f"frames with dark edge (adopted noise-board sessions): {dark}")
    return 0 if ok and ok_meta_same else 1


if __name__ == "__main__":
    sys.exit(main())
