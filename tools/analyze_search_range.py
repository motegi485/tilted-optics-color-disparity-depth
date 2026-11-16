"""tools/analyze_search_range.py — 探索範囲 SEARCH_RANGE_X の解析（設定値の感度解析）。

これは設定値の感度解析の実験例で、実行には自前の撮影データ（data/captures など）と、
列 session,gt_mm の manifest CSV（セッション名と正解距離 [mm]）が要る。

探索範囲 S（奇数・半幅 (S−1)/2・到達上限 (S−1)/2 − 0.5）を決めるための 4 つの材料を 1 回の実行で出す。

  (a) 分布: 現行設定の視差キャッシュ（全フレーム）から、距離ごとの |d| の p50/p90/p99/p99.9/最大と、
      到達上限近傍（|d| >= reach_max − 0.1）に張り付く画素の割合。
  (b) S 比較: S を変えて同じフレーム（各セッション連続 --frames-per-session 枚）を再計算し、基準 S との
      差 |Δd| > 0.5 px の画素率（探索範囲を変えると結果が変わる＝誤マッチ候補）、基準では有効だったが
      失われた／新たに得られた画素率、基準の到達上限を超える |d| の画素率。
  (c) 予測: 現行の参照式係数（params/fit_result.csv）から、画面上端（pos 枝）・下端（neg 枝）で生じる視差
      d = (y − y0(z))/ā を z = --z-min..--z-max で計算した「対象距離レンジで必要な最大視差」の曲線。
  (d) 計算時間: S ごとに calcDisparityByZNCC を同じフレームで n 回計測（checks.bench_disparity と同じ流儀）。

出力（--out-dir、既定 outputs/analysis/search_range_<日時>/）:
  distribution.csv / search_range_compare.csv / predicted_max_disparity.csv / bench.csv / summary.json /
  predicted_vs_measured.png / search_range_compare.png

実行例:
  uv run python -m tools.analyze_search_range --manifest outputs\\distance_manifest.csv --skip-first 10 `
      --search-ranges 21 41 51 --frames-per-session 30
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
from core.params import DepthParams, load_params
from tools.analysis_common import (
    DispSettings,
    disparity_of_proc,
    edge_strength_of_proc,
    ensure_capture_disparity,
    list_frames,
    load_stack,
    preprocess_to,
)

DIST_COLUMNS = ["session", "gt_mm", "frames", "n_valid", "p50", "p90", "p99", "p999", "max",
                "frac_ge_reach", "frac_gt_high", "reach_max_px", "high_px"]
CMP_COLUMNS = ["session", "gt_mm", "search_range_x", "reach_max_px", "frames", "n_valid_base", "n_valid",
               "n_both", "frac_changed", "frac_lost", "frac_gained", "frac_beyond_base_reach", "max_abs_d"]
PRED_COLUMNS = ["z_mm", "y0_pos", "y0_neg", "d_top_pos", "d_bottom_neg", "d_max_abs"]
BENCH_COLUMNS = ["search_range_x", "n", "mean_ms", "min_ms", "p95_ms", "edge_px", "frame"]


def distribution_row(session: str, gt_mm: float, stack: np.ndarray, *, reach_max: float, high: float) -> dict:
    """視差スタック (N,H,W) の |d| 分布（有効画素のみ）。"""
    ad = np.abs(stack[~np.isnan(stack)]).astype(np.float64)
    n = int(ad.size)
    row = {"session": session, "gt_mm": gt_mm, "frames": int(stack.shape[0]), "n_valid": n,
           "reach_max_px": reach_max, "high_px": high}
    if n == 0:
        row.update({k: float("nan") for k in ("p50", "p90", "p99", "p999", "max", "frac_ge_reach", "frac_gt_high")})
        return row
    p50, p90, p99, p999 = np.percentile(ad, [50, 90, 99, 99.9])
    row.update({"p50": float(p50), "p90": float(p90), "p99": float(p99), "p999": float(p999),
                "max": float(ad.max()), "frac_ge_reach": float(np.mean(ad >= reach_max - 0.1)),
                "frac_gt_high": float(np.mean(ad > high))})
    return row


def compare_row(session: str, gt_mm: float, base: np.ndarray, other: np.ndarray, *,
                search_range_x: int, reach_max: float, base_reach: float, change_px: float = 0.5) -> dict:
    """同じフレーム群について基準 S の視差 base と別の S の視差 other を比較する。"""
    if base.shape != other.shape:
        raise ValueError(f"compare_row: 形状不一致 {base.shape} != {other.shape}")
    vb, vo = ~np.isnan(base), ~np.isnan(other)
    both = vb & vo
    n_base, n_other, n_both = int(vb.sum()), int(vo.sum()), int(both.sum())
    changed = float(np.mean(np.abs(base[both] - other[both]) > change_px)) if n_both else float("nan")
    ao = np.abs(other[vo])
    return {
        "session": session, "gt_mm": gt_mm, "search_range_x": search_range_x, "reach_max_px": reach_max,
        "frames": int(base.shape[0]), "n_valid_base": n_base, "n_valid": n_other, "n_both": n_both,
        "frac_changed": changed,
        "frac_lost": float((vb & ~vo).sum() / n_base) if n_base else float("nan"),
        "frac_gained": float((vo & ~vb).sum() / n_other) if n_other else float("nan"),
        "frac_beyond_base_reach": float(np.mean(ao > base_reach)) if n_other else float("nan"),
        "max_abs_d": float(ao.max()) if n_other else float("nan"),
    }


def predicted_max_disparity(params: DepthParams, z_values: np.ndarray, height: int) -> list[dict]:
    """参照式 y0(z) = v/z + u と d = (y − y0)/ā から、画面上端（y=0・pos 枝）と下端（y=H−1・neg 枝）の視差。"""
    rows = []
    for z in z_values:
        y0p = params.v_pos / z + params.u_pos
        y0n = params.v_neg / z + params.u_neg
        d_top = (0.0 - y0p) / params.a_pos            # ā < 0・y0p > 0 なら正
        d_bot = ((height - 1) - y0n) / params.a_neg   # ā < 0・y0n < H−1 なら負
        rows.append({"z_mm": float(z), "y0_pos": float(y0p), "y0_neg": float(y0n),
                     "d_top_pos": float(d_top), "d_bottom_neg": float(d_bot),
                     "d_max_abs": float(max(abs(d_top), abs(d_bot)))})
    return rows


def bench_search_ranges(proc: np.ndarray, settings_list: list[DispSettings], n: int) -> list[dict]:
    """同じフレームで S ごとの calcDisparityByZNCC の計算時間（warmup 1 回・n 回計測・平均/最小/p95）。"""
    blue, _, red = cv2.split(proc)
    edge = cv2.threshold(edge_strength_of_proc(proc), settings_list[0].edge_th, 255, cv2.THRESH_TOZERO)[1]
    edge_px = int(np.count_nonzero(edge))
    rows = []
    for s in settings_list:
        disparity_of_proc(proc, s)                     # warmup
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            disparity_of_proc(proc, s)
            times.append((time.perf_counter() - t0) * 1000.0)
        rows.append({"search_range_x": s.search_range_x, "n": n, "mean_ms": float(np.mean(times)),
                     "min_ms": float(np.min(times)), "p95_ms": float(np.percentile(times, 95)), "edge_px": edge_px})
    return rows


def _write_csv(rows: list[dict], columns: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def _plots(dist_rows: list[dict], pred_rows: list[dict], cmp_rows: list[dict], base: DispSettings,
           out_dir: Path) -> None:
    """軸ラベルは既定フォントで描ける英語にする。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    z = [r["z_mm"] for r in pred_rows]
    ax.plot(z, [abs(r["d_top_pos"]) for r in pred_rows], "C0-", label="predicted |d| at top row (pos)")
    ax.plot(z, [abs(r["d_bottom_neg"]) for r in pred_rows], "C1-", label="predicted |d| at bottom row (neg)")
    gt = [r["gt_mm"] for r in dist_rows]
    ax.plot(gt, [r["p99"] for r in dist_rows], "ks", ms=4, label="measured p99")
    ax.plot(gt, [r["p999"] for r in dist_rows], "k^", ms=4, label="measured p99.9")
    ax.plot(gt, [r["max"] for r in dist_rows], "kx", ms=5, label="measured max")
    ax.axhline(base.reach_max, color="0.4", ls="--", lw=1, label=f"reach max (S={base.search_range_x})")
    ax.axhline(config.DISP_PLOT_HEIGHT, color="0.6", ls=":", lw=1, label="DISP_PLOT_HEIGHT")
    ax.set_xlabel("distance z [mm]")
    ax.set_ylabel("|d| [px]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_vs_measured.png", dpi=150)
    plt.close(fig)

    if cmp_rows:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for s in sorted({r["search_range_x"] for r in cmp_rows}):
            rs = [r for r in cmp_rows if r["search_range_x"] == s]
            ax.plot([r["gt_mm"] for r in rs], [r["frac_changed"] * 100 for r in rs], "o-", label=f"S={s}: |Δd|>0.5 px")
            ax.plot([r["gt_mm"] for r in rs], [r["frac_lost"] * 100 for r in rs], "x--", label=f"S={s}: lost")
        ax.set_xlabel("distance z [mm]")
        ax.set_ylabel(f"% of pixels vs S={base.search_range_x}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "search_range_compare.png", dpi=150)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="探索範囲 SEARCH_RANGE_X の解析（分布・S 比較・予測曲線・計算時間）")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sessions", nargs="+", metavar="NAME")
    src.add_argument("--manifest", metavar="CSV")
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None, help="分布 (a) に使う最大フレーム数（既定: 全部）")
    parser.add_argument("--frames-per-session", type=int, default=30, help="S 比較 (b) に使う連続フレーム数（既定 30）")
    parser.add_argument("--search-ranges", type=int, nargs="*", default=[21, 41, 51],
                        help="比較する S（奇数）。空なら (b) を省略")
    parser.add_argument("--kernel-size", type=int, default=config.KERNEL_SIZE)
    parser.add_argument("--search-range", type=int, default=config.SEARCH_RANGE_X, help="基準 S（既定 config）")
    parser.add_argument("--edge-th", type=float, default=float(config.EDGE_TH))
    parser.add_argument("--band-high", type=float, default=config.DISP_PLOT_HEIGHT, help="frac_gt_high の閾値")
    parser.add_argument("--params", default=str(config.PARAMS_PATH), help="予測曲線に使う係数 CSV")
    parser.add_argument("--z-min", type=float, default=1000.0)
    parser.add_argument("--z-max", type=float, default=4000.0)
    parser.add_argument("--z-step", type=float, default=100.0)
    parser.add_argument("--bench-session", default=None, help="計算時間 (d) に使うセッション（既定: 最初のセッション）")
    parser.add_argument("--bench-n", type=int, default=20)
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--captures-dir", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    try:
        base = DispSettings(kernel_size=args.kernel_size, search_range_x=args.search_range, edge_th=args.edge_th)
        others = [DispSettings(kernel_size=args.kernel_size, search_range_x=s, edge_th=args.edge_th)
                  for s in args.search_ranges if s != base.search_range_x]
        if args.manifest is not None:
            targets = load_manifest(Path(args.manifest))
        else:
            targets = [(name, parse_gt_from_session_name(name), "") for name in args.sessions]
        params = load_params(args.params)
        if args.frames_per_session <= 0 or args.bench_n <= 0 or args.z_step <= 0 or args.z_min > args.z_max:
            raise ValueError("--frames-per-session/--bench-n/--z-step は正・--z-min <= --z-max")
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"[ERROR] {e}")
        return 2

    cap_root = Path(args.captures_dir) if args.captures_dir else Path(config.CAPTURES_DIR)
    cache_root_dir = Path(args.cache_root) if args.cache_root else None
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(config.OUTPUTS_DIR) / "analysis" / f"search_range_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dist_rows: list[dict] = []
    cmp_rows: list[dict] = []
    bench_proc = None
    for session, gt, note in targets:
        cap_dir = cap_root / session
        try:
            frames_all = list_frames(cap_dir, skip_first=args.skip_first, max_frames=args.max_frames)
            stack = load_stack(ensure_capture_disparity(base, session, frames_all, root=cache_root_dir))
        except (ValueError, FileNotFoundError) as e:
            print(f"[ERROR] {session}: {e}")
            return 2
        dist_rows.append(distribution_row(session, gt, stack, reach_max=base.reach_max, high=args.band_high))
        r = dist_rows[-1]
        print(f"  {session}: n_valid={r['n_valid']} p99={r['p99']:.2f} p99.9={r['p999']:.2f} max={r['max']:.2f} "
              f"frac_ge_reach={r['frac_ge_reach']*100:.4f}%", flush=True)
        if others:
            frames_cmp = frames_all[:args.frames_per_session]
            base_stack = stack[:len(frames_cmp)]
            for s in others:
                other = load_stack(ensure_capture_disparity(s, session, frames_cmp, root=cache_root_dir))
                cmp_rows.append(compare_row(session, gt, base_stack, other, search_range_x=s.search_range_x,
                                            reach_max=s.reach_max, base_reach=base.reach_max))
                c = cmp_rows[-1]
                print(f"    S={s.search_range_x}: changed={c['frac_changed']*100:.3f}% lost={c['frac_lost']*100:.3f}% "
                      f"beyond_base_reach={c['frac_beyond_base_reach']*100:.4f}% max={c['max_abs_d']:.2f}", flush=True)
        if bench_proc is None and (args.bench_session is None or args.bench_session == session):
            raw = cv2.imread(str(frames_all[0]))
            bench_proc = (session, frames_all[0].name, preprocess_to(raw, base.image_size))
        del stack

    z_values = np.arange(args.z_min, args.z_max + args.z_step / 2, args.z_step)
    pred_rows = predicted_max_disparity(params, z_values, base.image_size[1])
    bench_rows: list[dict] = []
    if not args.no_bench and bench_proc is not None:
        bench_rows = bench_search_ranges(bench_proc[2], [base] + others, args.bench_n)
        for b in bench_rows:
            b["frame"] = f"{bench_proc[0]}/{bench_proc[1]}"

    _write_csv(dist_rows, DIST_COLUMNS, out_dir / "distribution.csv")
    _write_csv(cmp_rows, CMP_COLUMNS, out_dir / "search_range_compare.csv")
    _write_csv(pred_rows, PRED_COLUMNS, out_dir / "predicted_max_disparity.csv")
    _write_csv(bench_rows, BENCH_COLUMNS, out_dir / "bench.csv")
    in_range = [r for r in pred_rows]
    summary = {
        "base": {"kernel_size": base.kernel_size, "search_range_x": base.search_range_x, "edge_th": base.edge_th,
                 "disp_max": base.disp_max, "reach_max": base.reach_max},
        "compared_search_ranges": [s.search_range_x for s in others],
        "params_path": args.params, "skip_first": args.skip_first, "frames_per_session": args.frames_per_session,
        "predicted_max_abs_d_over_z_range": max(r["d_max_abs"] for r in in_range),
        "measured_p999_max": max(r["p999"] for r in dist_rows), "measured_max": max(r["max"] for r in dist_rows),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_plots:
        _plots(dist_rows, pred_rows, cmp_rows, base, out_dir)

    print("\nz [mm]  y0_pos  y0_neg  d_top(pos)  d_bottom(neg)  (predicted from params)")
    for r in pred_rows[::max(1, len(pred_rows) // 10)]:
        print(f"  {r['z_mm']:6.0f}  {r['y0_pos']:7.1f}  {r['y0_neg']:7.1f}  {r['d_top_pos']:7.2f}  {r['d_bottom_neg']:7.2f}")
    if bench_rows:
        print("\nS   mean_ms  p95_ms  (edge_px=%d, frame=%s)" % (bench_rows[0]["edge_px"], bench_rows[0]["frame"]))
        for b in bench_rows:
            print(f"  {b['search_range_x']:3d}  {b['mean_ms']:7.2f}  {b['p95_ms']:7.2f}")
    print(f"out: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
