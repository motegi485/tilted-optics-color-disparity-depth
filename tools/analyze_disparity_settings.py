"""tools/analyze_disparity_settings.py — ZNCC 窓 KERNEL_SIZE・エッジ閾値 EDGE_TH の感度解析。

これは設定値の感度解析の実験例で、実行には自前の撮影データ（data/captures, data/calib_imgs など）と、
列 session,gt_mm の manifest CSV（セッション名と正解距離 [mm]）が要る。

視差推定層の設定 K（ZNCC 窓・奇数）と T（Sobel エッジ閾値）を一因子ずつ振り（OFAT）、各設定で
  校正画像 10 枚 ＋ 評価各セッション連続 N フレームの視差を再計算（tools.analysis_common のキャッシュ）
  → 固定の校正帯・RANSAC で校正 → 検証距離の精度（推論ゲートなし＝nogate／校正帯ゲート＝bandgate）・有効画素率
  → 時間ノイズ σ_d（連続フレームの画素ごとの標準偏差の中央値・analyze_disparity_stability と同じ定義）
  → 計算時間（同じフレームで n 回計測）
を 1 表にまとめる。精度・有効画素の密度・計算時間のトレードオフの判断は利用者に委ね、このスクリプトは比較表を出すだけ。

出力（--out-dir、既定 outputs/analysis/disparity_settings_<日時>/）:
  settings.csv（1 行＝1 設定）/ per_session.csv（設定 × セッション × 推論ゲートの距離別指標）/ summary.json /
  mae_vs_kernel.png / mae_vs_edge.png（--no-plots で省略）

実行例:
  uv run python -m tools.analyze_disparity_settings --manifest outputs\\distance_manifest.csv --skip-first 10 `
      --kernel-sizes 7 9 11 13 15 17 21 25 29 33 --edge-ths 5 10 15 20 30 45 60 90
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
    disparity_of_proc,
    ensure_calib_disparity,
    ensure_capture_disparity,
    evaluate_session,
    list_frames,
    load_stack,
    preprocess_to,
    summarize_holdout,
)
from tools.analyze_disparity_stability import pixel_stats

EVAL_KEYS = ["mae_mean_mm", "abs_bias_max_mm", "rmse_mean_mm", "valid_rate_mean"]
PER_SESSION_COLUMNS = ["param", "value", "session", "gt_mm", "gate", "frames_used", "n_valid_total", "valid_rate",
                       "mean_err_mm", "mae_mm", "rmse_mm", "median_err_mm"]
COLUMNS = (["param", "value", "kernel_size", "search_range_x", "edge_th", "calib_ok", "error",
            "v_pos", "u_pos", "a_pos", "v_neg", "u_neg", "a_neg", "fit_rms_pos_px", "fit_rms_neg_px",
            "depth_rmse_mm", "inlier_pos_mean", "inlier_neg_mean", "n_skipped"]
           + [f"{k}_nogate" for k in EVAL_KEYS] + [f"{k}_bandgate" for k in EVAL_KEYS]
           + [f"extrap_{k}_nogate" for k in EVAL_KEYS]
           + ["sigma_median_px", "sigma_mean_px", "flip_rate_mean", "px_used_stability",
              "edge_px_bench", "mean_ms", "p95_ms", "n_bench", "seconds_disparity"])


def stability_summary(stacks: dict[str, np.ndarray], *, min_valid_frac: float = 0.5) -> dict:
    """各セッションの連続フレームから画素ごとの σ_d・符号反転率を出し、全セッションをプールして要約する。"""
    sig, flip = [], []
    for stack in stacks.values():
        st = pixel_stats(stack)
        sel = (st["valid_frac"] >= min_valid_frac) & (st["n_valid"] >= 2)
        sig.append(st["sigma"][sel])
        flip.append(st["flip_rate"][sel])
    sig = np.concatenate(sig) if sig else np.array([])
    flip = np.concatenate(flip) if flip else np.array([])
    sig = sig[np.isfinite(sig)]
    flip = flip[np.isfinite(flip)]
    return {
        "sigma_median_px": float(np.median(sig)) if sig.size else float("nan"),
        "sigma_mean_px": float(np.mean(sig)) if sig.size else float("nan"),
        "flip_rate_mean": float(np.mean(flip)) if flip.size else float("nan"),
        "px_used_stability": int(sig.size),
    }


def bench_setting(proc: np.ndarray, settings: DispSettings, n: int) -> dict:
    """同じフレームで settings の calcDisparityByZNCC を warmup 1 回 ＋ n 回計測。エッジ画素数も返す。"""
    _, _, red = cv2.split(proc)
    edge = cv2.threshold(cv2.convertScaleAbs(cv2.Sobel(red, cv2.CV_32F, 1, 0)), settings.edge_th, 255,
                         cv2.THRESH_TOZERO)[1]
    disparity_of_proc(proc, settings)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        disparity_of_proc(proc, settings)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {"edge_px_bench": int(np.count_nonzero(edge)), "mean_ms": float(np.mean(times)),
            "p95_ms": float(np.percentile(times, 95)), "n_bench": n}


def run_setting(settings: DispSettings, param: str, value: float, *, calib_dir: Path,
                targets: list[tuple[str, float]], cap_root: Path, skip_first: int, frames_per_session: int,
                band: tuple[float, float], ransac_th: float | None, cache_root_dir: Path | None,
                bench_proc: np.ndarray | None, bench_n: int) -> tuple[dict, list[dict]]:
    """1 設定を処理して (要約 1 行, セッション別の行) を返す。セッション別は距離ごとのトレードオフを見るため。"""
    per_rows: list[dict] = []
    row: dict = {"param": param, "value": value, "kernel_size": settings.kernel_size,
                 "search_range_x": settings.search_range_x, "edge_th": settings.edge_th, "calib_ok": False, "error": ""}
    t0 = time.perf_counter()
    calib_paths = ensure_calib_disparity(settings, calib_dir, root=cache_root_dir)
    disp_by_depth = {z: np.load(str(p)) for z, p in calib_paths.items()}
    z_lo, z_hi = min(disp_by_depth), max(disp_by_depth)
    stacks: dict[str, np.ndarray] = {}
    gts: dict[str, float] = {}
    for session, gt in targets:
        frames = list_frames(cap_root / session, skip_first=skip_first, max_frames=frames_per_session)
        stacks[session] = load_stack(ensure_capture_disparity(settings, session, frames, root=cache_root_dir))
        gts[session] = gt
    row["seconds_disparity"] = time.perf_counter() - t0
    holdout = [s for s in stacks if z_lo <= gts[s] <= z_hi]
    extrap = [s for s in stacks if s not in holdout]

    try:
        res = calibrate_from_disparity(disp_by_depth, low=band[0], high=band[1], ransac_th=ransac_th)
    except (ValueError, RuntimeError, KeyError) as e:
        row["error"] = str(e)[:120]
    else:
        row["calib_ok"] = True
        p = res.params
        row.update({"v_pos": p.v_pos, "u_pos": p.u_pos, "a_pos": p.a_pos, "v_neg": p.v_neg, "u_neg": p.u_neg,
                    "a_neg": p.a_neg, "fit_rms_pos_px": res.fit_rms_px["pos"], "fit_rms_neg_px": res.fit_rms_px["neg"],
                    "depth_rmse_mm": res.depth_rmse_mm["combined"],
                    "inlier_pos_mean": float(np.mean([r["inlier_pos"] for r in res.rows])),
                    "inlier_neg_mean": float(np.mean([r["inlier_neg"] for r in res.rows])),
                    "n_skipped": len(res.skipped)})
        for tag, b in (("nogate", None), ("bandgate", band)):
            per = {s: evaluate_session(stacks[s], gts[s], p, disp_max=settings.disp_max, band=b) for s in stacks}
            for s_name, m in per.items():
                per_rows.append({"param": param, "value": value, "session": s_name, "gt_mm": gts[s_name], "gate": tag,
                                 **{k: m.get(k) for k in PER_SESSION_COLUMNS[5:]}})
            h = summarize_holdout(per, holdout)
            row.update({f"{k}_{tag}": h[k] for k in EVAL_KEYS})
            if tag == "nogate":
                x = summarize_holdout(per, extrap)
                row.update({f"extrap_{k}_nogate": x[k] for k in EVAL_KEYS})
    row.update(stability_summary(stacks))
    if bench_proc is not None:
        row.update(bench_setting(bench_proc, settings, bench_n))
    return row, per_rows


def _write_csv(rows: list[dict], path: Path, columns: list[str] = COLUMNS) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def _plots(rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for param, xlabel, fname in (("kernel_size", "KERNEL_SIZE [px]", "mae_vs_kernel.png"),
                                 ("edge_th", "EDGE_TH", "mae_vs_edge.png")):
        rs = [r for r in rows if r["param"] == param and r.get("calib_ok")]
        if not rs:
            continue
        x = [r["value"] for r in rs]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        axes[0].plot(x, [r["mae_mean_mm_nogate"] for r in rs], "o-", label="no inference gate")
        axes[0].plot(x, [r["mae_mean_mm_bandgate"] for r in rs], "s--", label="inference gate = band")
        axes[0].set_ylabel("held-out MAE mean [mm]")
        axes[0].legend(fontsize=8)
        axes[1].plot(x, [r["valid_rate_mean_nogate"] for r in rs], "o-", label="no gate")
        axes[1].plot(x, [r["valid_rate_mean_bandgate"] for r in rs], "s--", label="band gate")
        axes[1].set_ylabel("valid rate")
        axes[1].legend(fontsize=8)
        if all(np.isfinite(r.get("p95_ms", np.nan)) for r in rs):
            axes[2].plot(x, [r["p95_ms"] for r in rs], "o-")
            axes[2].axhline(1000.0 / 30.0, color="0.5", ls="--", lw=1)
            axes[2].set_ylabel("disparity p95 [ms] (bench frame)")
        for ax in axes:
            ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="ZNCC 窓・エッジ閾値の一因子探索")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sessions", nargs="+", metavar="NAME")
    src.add_argument("--manifest", metavar="CSV")
    parser.add_argument("--calib-dir", default=str(config.CALIB_IMGS_DIR))
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--frames-per-session", type=int, default=30)
    parser.add_argument("--kernel-sizes", type=int, nargs="*", default=[7, 9, 11, 13, 15, 17, 21, 25, 29, 33])
    parser.add_argument("--edge-ths", type=float, nargs="*", default=[5, 10, 15, 20, 30, 45, 60, 90])
    parser.add_argument("--base-kernel", type=int, default=config.KERNEL_SIZE, help="T を振るときに固定する K")
    parser.add_argument("--base-edge-th", type=float, default=float(config.EDGE_TH), help="K を振るときに固定する T")
    parser.add_argument("--search-range", type=int, default=config.SEARCH_RANGE_X)
    parser.add_argument("--band-low", type=float, default=config.DISP_PLOT_LOW)
    parser.add_argument("--band-high", type=float, default=config.DISP_PLOT_HEIGHT)
    parser.add_argument("--ransac-th", type=float, default=config.RANSAC_RESIDUAL_TH)
    parser.add_argument("--bench-session", default=None)
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
        if args.frames_per_session <= 0 or args.bench_n <= 0:
            raise ValueError("--frames-per-session/--bench-n は正")
        settings_list: list[tuple[DispSettings, str, float]] = []
        for k in args.kernel_sizes:
            settings_list.append((DispSettings(kernel_size=k, search_range_x=args.search_range,
                                               edge_th=args.base_edge_th), "kernel_size", float(k)))
        for t in args.edge_ths:
            settings_list.append((DispSettings(kernel_size=args.base_kernel, search_range_x=args.search_range,
                                               edge_th=float(t)), "edge_th", float(t)))
        if not settings_list:
            raise ValueError("振る設定がありません（--kernel-sizes / --edge-ths）")
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}")
        return 2

    cap_root = Path(args.captures_dir) if args.captures_dir else Path(config.CAPTURES_DIR)
    cache_root_dir = Path(args.cache_root) if args.cache_root else None
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(config.OUTPUTS_DIR) / "analysis" / f"disparity_settings_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_proc = None
    bench_name = ""
    if not args.no_bench:
        session = args.bench_session or targets[0][0]
        try:
            frame = list_frames(cap_root / session, skip_first=args.skip_first)[0]
        except (ValueError, FileNotFoundError) as e:
            print(f"[ERROR] bench: {e}")
            return 2
        raw = cv2.imread(str(frame))
        if raw is None:
            print(f"[ERROR] bench: png を読めません: {frame}")
            return 2
        bench_proc = preprocess_to(raw, config.IMAGE_SIZE)
        bench_name = f"{session}/{frame.name}"

    rows: list[dict] = []
    per_rows_all: list[dict] = []
    for settings, param, value in settings_list:
        try:
            row, per_rows = run_setting(settings, param, value, calib_dir=Path(args.calib_dir), targets=targets,
                              cap_root=cap_root, skip_first=args.skip_first, frames_per_session=args.frames_per_session,
                              band=(args.band_low, args.band_high), ransac_th=args.ransac_th,
                              cache_root_dir=cache_root_dir, bench_proc=bench_proc, bench_n=args.bench_n)
        except (ValueError, FileNotFoundError) as e:
            print(f"[ERROR] {param}={value:g}: {e}")
            return 2
        rows.append(row)
        per_rows_all += per_rows
        if row["calib_ok"]:
            print(f"  {param}={value:g} ({settings.key()}): MAE nogate={row['mae_mean_mm_nogate']:.1f} "
                  f"bandgate={row['mae_mean_mm_bandgate']:.1f} |bias|max={row['abs_bias_max_mm_nogate']:.1f} "
                  f"valid={row['valid_rate_mean_nogate']:.3f} sigma_med={row['sigma_median_px']:.4f} "
                  f"p95={row.get('p95_ms', float('nan')):.1f} ms  [disp {row['seconds_disparity']:.0f} s]", flush=True)
        else:
            print(f"  {param}={value:g}: calibration FAILED: {row['error']}", flush=True)
    _write_csv(rows, out_dir / "settings.csv")
    _write_csv(per_rows_all, out_dir / "per_session.csv", PER_SESSION_COLUMNS)
    summary = {
        "targets": targets, "skip_first": args.skip_first, "frames_per_session": args.frames_per_session,
        "band": [args.band_low, args.band_high], "ransac_th": args.ransac_th, "search_range_x": args.search_range,
        "base_kernel": args.base_kernel, "base_edge_th": args.base_edge_th, "bench_frame": bench_name,
        "bench_n": args.bench_n, "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_plots:
        _plots(rows, out_dir)
    print(f"out: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
