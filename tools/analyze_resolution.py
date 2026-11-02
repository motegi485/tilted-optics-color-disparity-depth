"""tools/analyze_resolution.py — 処理解像度 IMAGE_SIZE の最適化（設定値の最適化実験 B3・2026-09-18）。

校正 raw（`capture_calib --save-raw` が `data/calib_raw/<距離>.png` に残す 1280×720 のバースト平均）と評価 raw
（`data/captures/<session>/<idx>.png`）から、処理解像度ごとに校正画像・評価フレームを `preprocess_to` で作り直し、
  K・S を解像度比 r = W/528 でスケール（K' = round_to_odd(K·r)、S' = round_to_odd(S·r)。T は既定で固定・--scale-edge-th で T/r）
  → 校正帯 (low·r, high·r)・RANSAC 閾値 th·r で校正（tools.analysis_common.calibrate_from_disparity）
  → 距離別精度（群: interior／at_calib／extrap × ゲート: nogate／bandgate）・有効画素率・σ_d・計算時間
を解像度ごとに 1 行にまとめる（tools.analysis_common のキャッシュを使うので同じ解像度の再実行は速い）。

校正距離のサブセットは --calib-distances で選ぶ。16 距離（1000〜4000 mm）のうち 1000・1100 mm は neg 枝の合焦行が
画面下端の外にあり点が無く（calibrate と同じ規則で距離ごとスキップ）、1200 mm も僅かなので、単純平均の ā を汚しうる。
どの距離を校正に使うかは calib_rows.csv（距離ごとの点数・傾き・インライヤ率）を見てユーザーが決める。
判断（精度 vs 密度 vs 速度）はユーザー。config.py は読むだけで書き換えない。

出力（--out-dir、既定 outputs/analysis/resolution_<日時>/）:
  settings.csv（1 行＝1 解像度）/ calib_rows.csv（解像度 × 校正距離・params.csv と同じ列）/
  per_session.csv（解像度 × セッション × ゲート）/ bench.csv（解像度 × bench セッション）/ summary.json /
  resolution_tradeoff.png（--no-plots で省略）

実行例:
  uv run python -m tools.analyze_resolution --manifest outputs\\distance_manifest_all.csv --skip-first 10 `
      --calib-distances 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 --base-search-range 41 `
      --bench-sessions dist_2300 scene_01
"""
from __future__ import annotations

# --- tools/ から root の config 等を import するためのブートストラップ ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import time
from datetime import datetime

import cv2
import numpy as np

import config
from checks.evaluate_distance_accuracy import load_manifest, parse_gt_from_session_name
from tools.analysis_common import (
    DispSettings,
    calibrate_from_disparity,
    ensure_calib_disparity_from_raw,
    ensure_capture_disparity,
    evaluate_session,
    list_frames,
    load_stack,
    preprocess_to,
    scale_factor,
    scaled_settings,
    summarize_holdout,
)
from tools.analyze_disparity_settings import bench_setting, stability_summary

DEFAULT_SIZES = ["1056x704", "792x528", "660x440", "528x352", "396x264", "264x176"]
EVAL_KEYS = ["mae_mean_mm", "abs_bias_max_mm", "rmse_mean_mm", "valid_rate_mean"]
GROUPS = ("interior", "at_calib", "extrap")
GATES = ("nogate", "bandgate")
PER_SESSION_COLUMNS = ["width", "height", "session", "gt_mm", "group", "gate", "frames_used", "n_valid_total",
                       "valid_rate", "mean_err_mm", "mae_mm", "rmse_mm", "median_err_mm"]
CALIB_ROW_COLUMNS = ["width", "height", "depth", "zero_disp_pos", "zero_disp_neg", "a_pos", "a_neg",
                     "inlier_pos", "inlier_neg", "n_pos", "n_neg"]
BENCH_COLUMNS = ["width", "height", "session", "frame", "edge_px_bench", "mean_ms", "p95_ms", "n_bench"]
COLUMNS = (["width", "height", "scale", "kernel_size", "search_range_x", "edge_th", "band_low", "band_high", "ransac_th",
            "calib_distances", "calib_ok", "error",
            "v_pos", "u_pos", "a_pos", "v_neg", "u_neg", "a_neg", "fit_rms_pos_px", "fit_rms_neg_px",
            "depth_rmse_mm", "inlier_pos_mean", "inlier_neg_mean", "n_skipped", "skipped"]
           + [f"{g}_{k}_{t}" for g in GROUPS for t in GATES for k in EVAL_KEYS]
           + ["n_interior", "n_at_calib", "n_extrap",
              "sigma_median_px", "sigma_median_base_px", "sigma_mean_px", "flip_rate_mean", "px_used_stability",
              "bench_session", "edge_px_bench", "mean_ms", "p95_ms", "n_bench", "seconds_disparity"])


def parse_size(text: str) -> tuple[int, int]:
    """'1056x704' → (1056, 704)。形式・正値・3:2 ± 1 % を検証する（不正は ValueError）。"""
    parts = text.lower().replace("×", "x").split("x")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise ValueError(f"--image-sizes は WxH 形式（例 1056x704）: {text!r}")
    size = (int(parts[0]), int(parts[1]))
    scale_factor(size)          # 正値・アスペクト比の検証
    return size


def assign_group(gt: float, calib_depths: list[int]) -> str:
    """評価距離を 3 群に分ける: at_calib（校正に使った距離と同じ）／interior（校正範囲の内側）／extrap（外側）。"""
    if not calib_depths:
        raise ValueError("assign_group: calib_depths が空です")
    if any(abs(gt - d) < 1e-9 for d in calib_depths):
        return "at_calib"
    return "interior" if min(calib_depths) < gt < max(calib_depths) else "extrap"


def run_resolution(size: tuple[int, int], base: DispSettings, *, scale_edge_th: bool, calib_raw_dir: Path,
                   calib_distances: list[int] | None, targets: list[tuple[str, float]], cap_root: Path,
                   skip_first: int, frames_per_session: int, band: tuple[float, float], ransac_th: float,
                   cache_root_dir: Path | None, bench_raws: list[tuple[str, str, np.ndarray]], bench_n: int,
                   ) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """1 解像度を処理して (要約 1 行, 校正距離ごとの行, セッション別の行, bench の行) を返す。"""
    r = scale_factor(size, base.image_size)
    settings = scaled_settings(base, size, scale_edge_th=scale_edge_th)
    band_r = (band[0] * r, band[1] * r)
    th_r = ransac_th * r
    w, h = size
    row: dict = {"width": w, "height": h, "scale": r, "kernel_size": settings.kernel_size,
                 "search_range_x": settings.search_range_x, "edge_th": settings.edge_th,
                 "band_low": band_r[0], "band_high": band_r[1], "ransac_th": th_r, "calib_ok": False, "error": ""}
    calib_rows: list[dict] = []
    per_rows: list[dict] = []
    bench_rows: list[dict] = []

    t0 = time.perf_counter()
    calib_paths = ensure_calib_disparity_from_raw(settings, calib_raw_dir, distances=calib_distances, root=cache_root_dir)
    disp_by_depth = {z: np.load(str(p)) for z, p in calib_paths.items()}
    row["calib_distances"] = " ".join(str(z) for z in sorted(disp_by_depth))
    stacks: dict[str, np.ndarray] = {}
    gts: dict[str, float] = {}
    for session, gt in targets:
        frames = list_frames(cap_root / session, skip_first=skip_first, max_frames=frames_per_session)
        stacks[session] = load_stack(ensure_capture_disparity(settings, session, frames, root=cache_root_dir))
        gts[session] = gt
    row["seconds_disparity"] = time.perf_counter() - t0

    try:
        res = calibrate_from_disparity(disp_by_depth, low=band_r[0], high=band_r[1], ransac_th=th_r)
    except (ValueError, RuntimeError, KeyError) as e:
        row["error"] = str(e)[:120]
    else:
        row["calib_ok"] = True
        p = res.params
        row.update({"v_pos": p.v_pos, "u_pos": p.u_pos, "a_pos": p.a_pos, "v_neg": p.v_neg, "u_neg": p.u_neg,
                    "a_neg": p.a_neg, "fit_rms_pos_px": res.fit_rms_px["pos"], "fit_rms_neg_px": res.fit_rms_px["neg"],
                    "depth_rmse_mm": res.depth_rmse_mm["combined"],
                    "inlier_pos_mean": float(np.mean([c["inlier_pos"] for c in res.rows])),
                    "inlier_neg_mean": float(np.mean([c["inlier_neg"] for c in res.rows])),
                    "n_skipped": len(res.skipped), "skipped": " ".join(str(z) for z in res.skipped)})
        calib_rows = [{"width": w, "height": h, **c} for c in res.rows]
        used_depths = [int(c["depth"]) for c in res.rows]        # 点数不足でスキップした距離は校正範囲に数えない
        groups = {s: assign_group(gts[s], used_depths) for s in stacks}
        for g in GROUPS:
            row[f"n_{g}"] = sum(1 for s in stacks if groups[s] == g)
        for tag, b in (("nogate", None), ("bandgate", band_r)):
            per = {s: evaluate_session(stacks[s], gts[s], p, disp_max=settings.disp_max, band=b) for s in stacks}
            for s_name, m in per.items():
                per_rows.append({"width": w, "height": h, "session": s_name, "gt_mm": gts[s_name],
                                 "group": groups[s_name], "gate": tag,
                                 **{k: m.get(k) for k in PER_SESSION_COLUMNS[6:]}})
            for g in GROUPS:
                summary = summarize_holdout(per, [s for s in stacks if groups[s] == g])
                row.update({f"{g}_{k}_{tag}": summary[k] for k in EVAL_KEYS})
    row.update(stability_summary(stacks))
    row["sigma_median_base_px"] = row["sigma_median_px"] / r
    for session, frame_name, raw in bench_raws:
        b = bench_setting(preprocess_to(raw, size), settings, bench_n)
        bench_rows.append({"width": w, "height": h, "session": session, "frame": frame_name, **b})
    if bench_rows:
        row.update({"bench_session": bench_rows[0]["session"],
                    **{k: bench_rows[0][k] for k in ("edge_px_bench", "mean_ms", "p95_ms", "n_bench")}})
    return row, calib_rows, per_rows, bench_rows


def _write_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def _plots(rows: list[dict], bench_rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = [r for r in rows if r.get("calib_ok")]
    if not rs:
        return
    x = [r["width"] for r in rs]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    ax = axes[0, 0]
    ax.plot(x, [r["interior_mae_mean_mm_nogate"] for r in rs], "o-", label="interior, no inference gate")
    ax.plot(x, [r["interior_mae_mean_mm_bandgate"] for r in rs], "s--", label="interior, inference gate = band")
    if any(np.isfinite(r["extrap_mae_mean_mm_nogate"]) for r in rs):
        ax.plot(x, [r["extrap_mae_mean_mm_nogate"] for r in rs], "^:", label="extrapolation, no gate")
    ax.set_ylabel("MAE mean [mm]")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(x, [r["interior_valid_rate_mean_nogate"] for r in rs], "o-", label="no gate")
    ax.plot(x, [r["interior_valid_rate_mean_bandgate"] for r in rs], "s--", label="band gate")
    ax.set_ylabel("valid rate (interior)")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    base_w = config.IMAGE_SIZE[0]
    for session in sorted({b["session"] for b in bench_rows}):
        bs = sorted((b for b in bench_rows if b["session"] == session), key=lambda b: b["width"])
        ref = next((b for b in bs if b["width"] == base_w), bs[0])      # 凡例の E は基準解像度（無ければ最小）の値
        ax.plot([b["width"] for b in bs], [b["p95_ms"] for b in bs], "o-",
                label=f"{session} (E@{ref['width']}={ref['edge_px_bench']})")
    ax.axhline(1000.0 / 30.0, color="0.5", ls="--", lw=1)
    ax.set_ylabel("disparity p95 [ms] (bench frame)")
    ax.set_yscale("log")
    if bench_rows:
        ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(x, [r["sigma_median_px"] for r in rs], "o-", label="sigma_d median [px]")
    ax.plot(x, [r["sigma_median_base_px"] for r in rs], "s--", label="sigma_d median / r [px @528]")
    ax.set_ylabel("temporal noise")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("processing width W [px] (K, S scaled with W/528)")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "resolution_tradeoff.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="処理解像度の最適化（B3）: raw から解像度別に校正・評価・計時")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sessions", nargs="+", metavar="NAME")
    src.add_argument("--manifest", metavar="CSV")
    parser.add_argument("--calib-raw-dir", default=str(config.CALIB_RAW_DIR),
                        help="capture_calib --save-raw の raw（<距離>.png・1280x720）")
    parser.add_argument("--calib-distances", type=int, nargs="*", default=None,
                        help="校正に使う距離 [mm] のサブセット（既定: ディレクトリ内すべて）")
    parser.add_argument("--image-sizes", nargs="*", default=DEFAULT_SIZES, help="WxH（3:2）。小さい順に処理する")
    parser.add_argument("--base-kernel", type=int, default=config.KERNEL_SIZE, help="528x352 での K（解像度比でスケール）")
    parser.add_argument("--base-search-range", type=int, default=config.SEARCH_RANGE_X,
                        help="528x352 での S（解像度比でスケール。B3 では 41 を明示）")
    parser.add_argument("--edge-th", type=float, default=float(config.EDGE_TH))
    parser.add_argument("--scale-edge-th", action="store_true", help="T も T/r にスケールする（既定: 固定）")
    parser.add_argument("--band-low", type=float, default=config.DISP_PLOT_LOW, help="528x352 での校正帯下限 [px]")
    parser.add_argument("--band-high", type=float, default=config.DISP_PLOT_HEIGHT, help="528x352 での校正帯上限 [px]")
    parser.add_argument("--ransac-th", type=float, default=config.RANSAC_RESIDUAL_TH, help="528x352 での RANSAC 閾値 [行]")
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--frames-per-session", type=int, default=30)
    parser.add_argument("--bench-sessions", nargs="*", default=None,
                        help="計時に使うセッション（複数可・既定 manifest 先頭。scene_01 のような非距離セッションも可）")
    parser.add_argument("--bench-n", type=int, default=10)
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--captures-dir", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.manifest is not None:
            targets = [(s, float(gt)) for s, gt, _ in load_manifest(Path(args.manifest))]
        else:
            targets = [(name, float(parse_gt_from_session_name(name))) for name in args.sessions]
        if args.frames_per_session <= 0 or args.bench_n <= 0 or args.skip_first < 0:
            raise ValueError("--frames-per-session/--bench-n は正、--skip-first は 0 以上")
        if not args.image_sizes:
            raise ValueError("--image-sizes が空です")
        sizes = sorted({parse_size(s) for s in args.image_sizes})
        base = DispSettings(kernel_size=args.base_kernel, search_range_x=args.base_search_range, edge_th=args.edge_th)
        if not (0.0 <= args.band_low < args.band_high) or args.ransac_th <= 0:
            raise ValueError("0 <= --band-low < --band-high、--ransac-th > 0 が必要")
        calib_raw_dir = Path(args.calib_raw_dir)
        if not calib_raw_dir.is_dir():
            raise FileNotFoundError(f"--calib-raw-dir がありません: {calib_raw_dir}")
        if args.calib_distances is not None and not args.calib_distances:
            raise ValueError("--calib-distances が空です")
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}")
        return 2

    cap_root = Path(args.captures_dir) if args.captures_dir else Path(config.CAPTURES_DIR)
    cache_root_dir = Path(args.cache_root) if args.cache_root else None
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(config.OUTPUTS_DIR) / "analysis" / f"resolution_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_raws: list[tuple[str, str, np.ndarray]] = []
    if not args.no_bench:
        for session in (args.bench_sessions or [targets[0][0]]):
            try:
                frame = list_frames(cap_root / session, skip_first=args.skip_first)[0]
            except (ValueError, FileNotFoundError) as e:
                print(f"[ERROR] bench: {e}")
                return 2
            raw = cv2.imread(str(frame))
            if raw is None:
                print(f"[ERROR] bench: png を読めません: {frame}")
                return 2
            bench_raws.append((session, frame.name, raw))

    rows: list[dict] = []
    calib_rows_all: list[dict] = []
    per_rows_all: list[dict] = []
    bench_rows_all: list[dict] = []
    for size in sizes:
        try:
            row, calib_rows, per_rows, bench_rows = run_resolution(
                size, base, scale_edge_th=args.scale_edge_th, calib_raw_dir=calib_raw_dir,
                calib_distances=args.calib_distances, targets=targets, cap_root=cap_root, skip_first=args.skip_first,
                frames_per_session=args.frames_per_session, band=(args.band_low, args.band_high),
                ransac_th=args.ransac_th, cache_root_dir=cache_root_dir, bench_raws=bench_raws, bench_n=args.bench_n)
        except (ValueError, FileNotFoundError) as e:
            print(f"[ERROR] {size[0]}x{size[1]}: {e}")
            return 2
        rows.append(row)
        calib_rows_all += calib_rows
        per_rows_all += per_rows
        bench_rows_all += bench_rows
        label = f"{size[0]}x{size[1]} (r={row['scale']:g}, K={row['kernel_size']} S={row['search_range_x']} T={row['edge_th']:g})"
        if row["calib_ok"]:
            print(f"  {label}: interior MAE nogate={row['interior_mae_mean_mm_nogate']:.1f} "
                  f"bandgate={row['interior_mae_mean_mm_bandgate']:.1f} |bias|max={row['interior_abs_bias_max_mm_nogate']:.1f} "
                  f"valid={row['interior_valid_rate_mean_nogate']:.3f} extrap MAE={row['extrap_mae_mean_mm_nogate']:.1f} "
                  f"sigma_med={row['sigma_median_px']:.4f} px p95={row.get('p95_ms', float('nan')):.1f} ms "
                  f"skipped=[{row['skipped']}] [disp {row['seconds_disparity']:.0f} s]", flush=True)
        else:
            print(f"  {label}: calibration FAILED: {row['error']}", flush=True)
    _write_csv(rows, out_dir / "settings.csv", COLUMNS)
    _write_csv(calib_rows_all, out_dir / "calib_rows.csv", CALIB_ROW_COLUMNS)
    _write_csv(per_rows_all, out_dir / "per_session.csv", PER_SESSION_COLUMNS)
    _write_csv(bench_rows_all, out_dir / "bench.csv", BENCH_COLUMNS)
    summary = {
        "targets": targets, "calib_raw_dir": str(calib_raw_dir), "calib_distances": args.calib_distances,
        "image_sizes": [f"{w}x{h}" for w, h in sizes],
        "base": {"kernel_size": base.kernel_size, "search_range_x": base.search_range_x, "edge_th": base.edge_th,
                 "image_size": list(base.image_size)},
        "scale_edge_th": args.scale_edge_th, "band": [args.band_low, args.band_high], "ransac_th": args.ransac_th,
        "skip_first": args.skip_first, "frames_per_session": args.frames_per_session,
        "bench_frames": [f"{s}/{f}" for s, f, _ in bench_raws], "bench_n": args.bench_n,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_plots:
        _plots(rows, bench_rows_all, out_dir)
    print(f"out: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
