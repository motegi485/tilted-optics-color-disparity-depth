"""evaluate_disparity_band_fraction.py — 校正帯（calibration band）に対する有効画素の視差分布の集計（実験ランブック手順 8'）。

校正（calibration/calibrate.py）は DISP_PLOT_LOW < |d| < DISP_PLOT_HEIGHT（= 0.45 < |d| < 7.5 px・以下
「校正帯」）の画素だけで直線 y = a·d + b を当て、d = 0 の切片を枝ごとに外挿する。推論
（realtime/pipeline.py）は |d| > DISP_MAX の画素しか捨てないので、校正帯の外側の画素にも深度が付く。
特に校正帯の下側（|d| ≤ 0.45 px）の画素は外挿域にあり、pos/neg のどちらの枝に落ちるかで異なる深度
になる（論文の限界の節）。本スクリプトは、距離別精度評価（checks.evaluate_distance_accuracy）の
有効画素を校正帯基準で 3 区分し、それぞれの画素数・割合・誤差を出す。

計算経路: 保存 PNG（data/captures/{session}/{i:06d}.png）を realtime.camera.preprocess →
core.disparity.calcDisparityByZNCC → gate1（|d| > DISP_MAX → NaN）→ core.depth.compute_depth の
順に realtime.pipeline.process_frame と同じ手順で再計算し、視差マップ d と深度マップ z を両方得る
（process_frame は z しか返さないので同じ手順を辿る）。再計算した z は保存 npy（唯一の正本）と
NaN 位置＋有限値のビット一致を確認し（checks.verify_clip_replay と同じ判定）、一致しなければ
その旨を CSV（replay_match）と標準出力に残す。

集計の定義（distance_accuracy と同じ有効画素・ROI・暖機除外）:
  - 有効画素  : gate2 と同じ config.DEPTH_MIN < z < config.DEPTH_MAX を満たす画素（ROI 内）。
                d == 0 ちょうどは compute_depth が NaN にするので含まれない
  - below_band: 有効画素のうち |d| <= band_low（既定 config.DISP_PLOT_LOW = 0.45 px）
  - in_band   : 有効画素のうち band_low < |d| < band_high（校正が使う帯・厳密不等号も校正と同じ）
  - above_band: 有効画素のうち |d| >= band_high（既定 config.DISP_PLOT_HEIGHT = 7.5 px）
    （3 区分は有効画素を過不足なく分割する）
  - *_fraction = Σ区分の画素数 / Σ有効画素（全フレーム統合）。below_band はフレームごとの割合の min/max も出す
  - below_band 画素の行 y の中央値・p10・p90（2 枝の切片の間の行に集中するかを見る）
  - 各区分の誤差 e = z − z_gt の平均・MAE

実行例:
  uv run python -m checks.evaluate_disparity_band_fraction --manifest outputs\\distance_manifest.csv `
      --skip-first 10 --out outputs\\disparity_band_fraction.csv
"""
from __future__ import annotations

# --- checks/ から root の config / core / realtime を import するためのブートストラップ ---
# 推奨実行は `uv run python -m checks.<name>`（root を cwd に）。
# 直接実行 `python checks/<name>.py` でも動くよう root を sys.path に通す（他の checks と同方式）。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
from datetime import datetime

import cv2
import numpy as np

import config
from checks.evaluate_distance_accuracy import (
    _fmt,
    _read_params_path,
    load_manifest,
    parse_gt_from_session_name,
    parse_roi,
)
from checks.verify_clip_replay import _compare
from core.depth import compute_depth
from core.disparity import calcDisparityByZNCC
from core.params import DepthParams, load_params
from realtime.camera import preprocess


BANDS = ("below_band", "in_band", "above_band")

# 出力 CSV の列順（print の表もこの順）。docs/verification-and-tools.md に列の定義を記載。
COLUMNS = [
    "session", "gt_mm", "frames_used", "frames_total", "roi", "band_low_px", "band_high_px",
    "n_valid_total",
    "n_below_band", "below_band_fraction", "below_band_fraction_min", "below_band_fraction_max",
    "n_in_band", "in_band_fraction",
    "n_above_band", "above_band_fraction",
    "below_band_row_median", "below_band_row_p10", "below_band_row_p90",
    "mean_err_below_band_mm", "mae_below_band_mm",
    "mean_err_in_band_mm", "mae_in_band_mm",
    "mean_err_above_band_mm", "mae_above_band_mm",
    "replay_match", "params_path", "note",
]


def list_capture_files(cap_dir: Path) -> list[Path]:
    """runner が保存した連番 PNG（??????.png・数字 stem）を昇順で返す。"""
    return sorted(p for p in cap_dir.glob("??????.png") if p.stem.isdigit())


def disparity_and_depth(raw: np.ndarray, params: DepthParams) -> tuple[np.ndarray, np.ndarray]:
    """raw(1280x720 BGR) → (視差マップ d, 深度マップ z)。realtime.pipeline.process_frame と同じ手順。

    process_frame は z しか返さないため同じ 4 段（R/B 分割 → ZNCC → gate1 → 参照式）をここで辿る。
    z の一致は呼び出し側が保存 npy と照合する（手順が乖離したら検出できる）。
    """
    proc = preprocess(raw)
    blue, _, red = cv2.split(proc)
    disp = calcDisparityByZNCC(red, blue, config.KERNEL_SIZE, config.SEARCH_RANGE_X, config.EDGE_TH)
    disp[np.abs(disp) > config.DISP_MAX] = np.nan          # gate1
    depth = compute_depth(
        disp,
        params.v_pos, params.u_pos, params.a_pos, params.w_pos,
        params.v_neg, params.u_neg, params.a_neg, params.w_neg,
        config.EPS_DEN)
    return disp, depth


def summarize_session(cap_dir: Path, out_dir: Path | None, gt_mm: int, params: DepthParams, *,
                      roi: tuple[int, int, int, int], skip_first: int = 0, max_frames: int | None = None,
                      band_low: float = config.DISP_PLOT_LOW, band_high: float = config.DISP_PLOT_HEIGHT,
                      depth_min: float = config.DEPTH_MIN, depth_max: float = config.DEPTH_MAX,
                      compute=disparity_and_depth) -> dict:
    """1 セッションの保存 PNG を再計算して COLUMNS のサブセット dict を返す。

    out_dir が None でなく対応する npy があれば再計算 z をビット照合し、replay_match に
    True/False を入れる（npy が 1 枚でも無い・不一致なら False。out_dir=None なら ""）。
    有効画素が 0 件のときは n_valid_total=0 とし、割合・誤差統計は NaN で埋める。
    compute は差し替え可能（テストで ZNCC を既知の視差に置き換えるため・本番は既定のまま）。
    """
    if skip_first < 0:
        raise ValueError(f"--skip-first は0以上: {skip_first}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"--max-frames は1以上: {max_frames}")
    if not (0.0 <= band_low < band_high):
        raise ValueError(f"band_low/band_high は 0 <= low < high: {band_low}, {band_high}")
    files = list_capture_files(cap_dir)
    if not files:
        raise FileNotFoundError(f"{cap_dir} に連番 png（??????.png）がありません")
    used = files[skip_first:]
    if max_frames is not None:
        used = used[:max_frames]
    if not used:
        raise ValueError(f"{cap_dir}: skip/max の指定で集計フレームが0件になりました "
                         f"(total={len(files)}, skip_first={skip_first}, max_frames={max_frames})")

    x0, y0, x1, y1 = roi
    n_valid_total = 0
    n_band = {b: 0 for b in BANDS}
    err_band: dict[str, list[np.ndarray]] = {b: [] for b in BANDS}
    below_frac_per_frame: list[float] = []
    below_rows: list[np.ndarray] = []
    replay_ok: bool | None = None if out_dir is None else True
    for p in used:
        raw = cv2.imread(str(p))
        if raw is None:
            raise ValueError(f"{p}: png を読めません")
        disp, depth = compute(raw, params)
        if out_dir is not None:
            npy = out_dir / f"{p.stem}.npy"
            if not npy.exists():
                replay_ok = False
            else:
                ok, _ = _compare(np.load(str(npy)), depth)
                if not ok:
                    replay_ok = False
        h, w = depth.shape
        if not (x1 <= w and y1 <= h):
            raise ValueError(f"{p}: ROI {roi} が深度マップの形状 (H={h}, W={w}) を超えています")
        sub_d = np.abs(disp[y0:y1, x0:x1])
        sub_z = depth[y0:y1, x0:x1].astype(np.float64)
        valid = (sub_z > depth_min) & (sub_z < depth_max)        # gate2 と同じ。NaN は False
        masks = {
            "below_band": valid & (sub_d <= band_low),
            "in_band": valid & (sub_d > band_low) & (sub_d < band_high),   # 校正と同じ厳密不等号
            "above_band": valid & (sub_d >= band_high),
        }
        n_valid = int(np.count_nonzero(valid))
        n_valid_total += n_valid
        for b in BANDS:
            n_band[b] += int(np.count_nonzero(masks[b]))
            if n_valid:
                err_band[b].append(sub_z[masks[b]] - float(gt_mm))
        if n_valid:
            below_frac_per_frame.append(int(np.count_nonzero(masks["below_band"])) / n_valid)
            below_rows.append(np.nonzero(masks["below_band"])[0] + y0)

    result: dict = {
        "gt_mm": gt_mm,
        "frames_used": len(used),
        "frames_total": len(files),
        "roi": f"{x0},{y0},{x1},{y1}",
        "band_low_px": band_low,
        "band_high_px": band_high,
        "n_valid_total": n_valid_total,
        "n_below_band": n_band["below_band"],
        "n_in_band": n_band["in_band"],
        "n_above_band": n_band["above_band"],
        "replay_match": "" if replay_ok is None else replay_ok,
    }
    nan = float("nan")
    stat_keys = ["below_band_fraction", "below_band_fraction_min", "below_band_fraction_max",
                 "in_band_fraction", "above_band_fraction",
                 "below_band_row_median", "below_band_row_p10", "below_band_row_p90"]
    stat_keys += [f"{s}_{b}_mm" for b in BANDS for s in ("mean_err", "mae")]
    result.update({k: nan for k in stat_keys})
    if n_valid_total == 0:
        return result
    for b in BANDS:
        result[f"{b}_fraction"] = n_band[b] / n_valid_total
        e = np.concatenate(err_band[b])
        if e.size:
            result[f"mean_err_{b}_mm"] = float(np.mean(e))
            result[f"mae_{b}_mm"] = float(np.mean(np.abs(e)))
    result["below_band_fraction_min"] = float(min(below_frac_per_frame))
    result["below_band_fraction_max"] = float(max(below_frac_per_frame))
    rows = np.concatenate(below_rows)
    if rows.size:
        result["below_band_row_median"] = float(np.median(rows))
        result["below_band_row_p10"] = float(np.percentile(rows, 10))
        result["below_band_row_p90"] = float(np.percentile(rows, 90))
    return result


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in COLUMNS})


def main(argv: list[str] | None = None) -> int:
    # Windows の日本語コンソール(cp932)は非対応文字の print で UnicodeEncodeError を出すため
    # UTF-8 に固定する（realtime/runner.py の main() 冒頭と同じ対処）。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="校正帯（0.45 < |d| < 7.5 px）に対する有効画素の視差分布の集計（手順 8'）")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sessions", nargs="+", metavar="NAME",
                     help="data/captures/{NAME} を再計算。末尾の `_<数字>` を正解距離 [mm] と解釈（例: dist_1500）")
    src.add_argument("--manifest", metavar="CSV",
                     help="列 session,gt_mm[,note] の CSV でセッションと正解距離を指定")
    parser.add_argument("--roi", default=None,
                        help="評価領域 x0,y0,x1,y1（処理解像度の画素座標・右下排他）。既定: 全画面")
    parser.add_argument("--skip-first", type=int, default=0,
                        help="各セッションの先頭からこの枚数を集計から除く（distance_accuracy と揃える。既定0）")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="各セッションで集計する最大フレーム数（既定: 全部）")
    parser.add_argument("--band-low", type=float, default=config.DISP_PLOT_LOW,
                        help=f"校正帯の下限 [px]（既定 config.DISP_PLOT_LOW={config.DISP_PLOT_LOW}）")
    parser.add_argument("--band-high", type=float, default=config.DISP_PLOT_HEIGHT,
                        help=f"校正帯の上限 [px]（既定 config.DISP_PLOT_HEIGHT={config.DISP_PLOT_HEIGHT}）")
    parser.add_argument("--params", default=None,
                        help="fit_result.csv のパス（既定: 各セッションの outputs/{session}/meta.json の params_path、"
                             "無ければ live＝root）")
    parser.add_argument("--no-replay-check", action="store_true",
                        help="再計算 z と保存 npy のビット照合を省略する（outputs/{session} が無いときなど）")
    parser.add_argument("--out", default=None,
                        help="集計結果 CSV の出力先（既定: outputs/disparity_band_fraction_<日時>.csv）")
    parser.add_argument("--captures-dir", default=None, help="captures の親（既定 config.CAPTURES_DIR。テスト用）")
    parser.add_argument("--outputs-dir", default=None, help="outputs の親（既定 config.OUTPUTS_DIR。テスト用）")
    args = parser.parse_args(argv)

    width, height = config.IMAGE_SIZE
    try:
        roi = parse_roi(args.roi, width, height)
        if args.manifest is not None:
            targets = load_manifest(Path(args.manifest))
        else:
            targets = [(name, parse_gt_from_session_name(name), "") for name in args.sessions]
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}")
        return 2

    cap_root = Path(args.captures_dir) if args.captures_dir else Path(config.CAPTURES_DIR)
    out_root = Path(args.outputs_dir) if args.outputs_dir else Path(config.OUTPUTS_DIR)
    rows: list[dict] = []
    any_empty = any_mismatch = False
    for session, gt, note in targets:
        cap_dir = cap_root / session
        out_dir = out_root / session
        if not cap_dir.is_dir():
            print(f"[ERROR] captures セッションが見つかりません: {cap_dir}")
            return 2
        params_path = args.params or _read_params_path(out_dir) or str(config.PARAMS_PATH)
        try:
            params = load_params(params_path)
            r = summarize_session(cap_dir, None if args.no_replay_check else out_dir, gt, params,
                                  roi=roi, skip_first=args.skip_first, max_frames=args.max_frames,
                                  band_low=args.band_low, band_high=args.band_high)
        except (ValueError, FileNotFoundError, KeyError) as e:
            print(f"[ERROR] {session}: {e}")
            return 2
        r["session"] = session
        r["note"] = note
        r["params_path"] = params_path
        if r["n_valid_total"] == 0:
            any_empty = True
            print(f"[WARN] {session}: 有効画素が0件（gate2 {config.DEPTH_MIN}<z<{config.DEPTH_MAX} と ROI {r['roi']}）")
        if r["replay_match"] is False:
            any_mismatch = True
            print(f"[WARN] {session}: 再計算した深度が保存 npy とビット一致しないフレームがあります"
                  f"（params={params_path}。checks.verify_clip_replay で確認）")
        rows.append(r)
        print(f"  {session}: valid={r['n_valid_total']} below_band={r['n_below_band']} "
              f"below_band_fraction={_fmt(r['below_band_fraction'])} replay_match={r['replay_match']}")

    show = ["session", "gt_mm", "frames_used", "n_valid_total",
            "below_band_fraction", "below_band_fraction_min", "below_band_fraction_max",
            "in_band_fraction", "above_band_fraction",
            "below_band_row_median", "below_band_row_p10", "below_band_row_p90",
            "mean_err_below_band_mm", "mae_below_band_mm", "mean_err_in_band_mm", "mae_in_band_mm",
            "mean_err_above_band_mm", "mae_above_band_mm", "replay_match"]
    print(f"\ngate2: {config.DEPTH_MIN} < z < {config.DEPTH_MAX} [mm]  band: {args.band_low} < |d| < {args.band_high} px  "
          f"roi={rows[0]['roi']}  skip_first={args.skip_first}  max_frames={args.max_frames}")
    print("  ".join(show))
    for r in rows:
        print("  ".join(_fmt(r.get(k, "")) for k in show))

    out_path = Path(args.out) if args.out else out_root / f"disparity_band_fraction_{datetime.now():%Y%m%d_%H%M%S}.csv"
    write_csv(rows, out_path)
    print(f"\nCSV: {out_path}")
    return 1 if (any_empty or any_mismatch) else 0


if __name__ == "__main__":
    sys.exit(main())
