"""tools/analyze_calib_sensitivity.py — 校正帯・RANSAC・推論ゲートのグリッド探索（最適化実験ランブック A2・A4）。

校正帯（DISP_PLOT_LOW < |d| < DISP_PLOT_HEIGHT）と RANSAC 残差閾値を振って校正し直し、各設定の 8 係数で
校正と独立な距離（held-out）の精度を評価する。視差は設定に依存しないので、視差キャッシュ
（tools.analysis_common）を読むだけで何十通りでも回せる。

  1. grid: low × high（--lows × --highs）を RANSAC 閾値 --th で校正 → held-out／外挿の指標。推論ゲートは
     「なし（現行）」と「校正帯と同じ」の両方で評価する（A4 の主要 2 案を同時に済ませる）。
  2. th sweep: grid で最良だった (low, high) を固定し、RANSAC 閾値 --ths（None＝sklearn の MAD 自動）を振る。
  3. inference band: 校正帯を固定（--fixed-low/--fixed-high、既定は grid の最良）し、推論側の帯 (low′, high′) を
     独立に振る（A4 の第 3 案）。校正はしない。
  4. residual: 現行帯で当てた RANSAC 直線に対する残差を、帯の外側の画素も含めて |d| 0.25 px 刻みで集計する
     （小視差域の系統的ずれ・大視差域の残差増大を定量化。校正帯の理論的解釈の材料）。

「最良」の既定は held-out の MAE 平均（--objective で abs_bias_max / rmse_mean に変更可）。判断はユーザー。

出力（--out-dir、既定 outputs/analysis/calib_sensitivity_<日時>/）:
  grid.csv / th_sweep.csv / inference_band.csv / residual_vs_disparity.csv / summary.json / heatmap_*.png

実行例:
  uv run python -m tools.analyze_calib_sensitivity --manifest outputs\\distance_manifest.csv --skip-first 10 `
      --frames-per-session 30
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

import numpy as np

import config
from calibration.calibrate import extract_sign_points, ransac_fit_line
from checks.evaluate_distance_accuracy import load_manifest, parse_gt_from_session_name
from core.params import DepthParams
from tools.analysis_common import (
    CalibResult,
    DispSettings,
    calibrate_from_disparity,
    ensure_calib_disparity,
    ensure_capture_disparity,
    evaluate_session,
    list_frames,
    load_stack,
    summarize_holdout,
)

DEFAULT_LOWS = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5, 2.0]
DEFAULT_HIGHS = [3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 11.0, 13.0, 14.5]
DEFAULT_THS = ["1", "2", "4", "8", "16", "32", "none"]

CALIB_KEYS = ["v_pos", "u_pos", "a_pos", "v_neg", "u_neg", "a_neg", "fit_rms_pos_px", "fit_rms_neg_px",
              "depth_rmse_pos_mm", "depth_rmse_neg_mm", "depth_rmse_mm", "inlier_pos_mean", "inlier_neg_mean",
              "n_used_pos", "n_used_neg", "n_skipped", "y0_gap_2300_px"]
EVAL_KEYS = ["mae_mean_mm", "abs_bias_max_mm", "rmse_mean_mm", "valid_rate_mean"]
GRID_COLUMNS = (["low", "high", "th", "calib_ok", "error"] + CALIB_KEYS
                + [f"{k}_nogate" for k in EVAL_KEYS] + [f"{k}_bandgate" for k in EVAL_KEYS]
                + [f"extrap_{k}_nogate" for k in EVAL_KEYS] + [f"extrap_{k}_bandgate" for k in EVAL_KEYS])
INF_COLUMNS = ["calib_low", "calib_high", "th", "inf_low", "inf_high"] + EVAL_KEYS + [f"extrap_{k}" for k in EVAL_KEYS]
RES_COLUMNS = ["sign", "bin_lo", "bin_hi", "n_px", "resid_median_row", "resid_p25_row", "resid_p75_row",
               "inlier_rate_th"]


def parse_th(text: str) -> float | None:
    t = text.strip().lower()
    if t in ("none", "mad", "auto"):
        return None
    v = float(t)
    if v <= 0:
        raise ValueError(f"RANSAC 閾値は正か none: {text!r}")
    return v


def calib_summary(res: CalibResult) -> dict:
    p = res.params
    with np.errstate(divide="ignore", invalid="ignore"):
        y0p = p.v_pos / 2300.0 + p.u_pos
        y0n = p.v_neg / 2300.0 + p.u_neg
    return {
        "v_pos": p.v_pos, "u_pos": p.u_pos, "a_pos": p.a_pos, "v_neg": p.v_neg, "u_neg": p.u_neg, "a_neg": p.a_neg,
        "fit_rms_pos_px": res.fit_rms_px["pos"], "fit_rms_neg_px": res.fit_rms_px["neg"],
        "depth_rmse_pos_mm": res.depth_rmse_mm["pos"], "depth_rmse_neg_mm": res.depth_rmse_mm["neg"],
        "depth_rmse_mm": res.depth_rmse_mm["combined"],
        "inlier_pos_mean": float(np.mean([r["inlier_pos"] for r in res.rows])),
        "inlier_neg_mean": float(np.mean([r["inlier_neg"] for r in res.rows])),
        "n_used_pos": res.n_used["pos"], "n_used_neg": res.n_used["neg"], "n_skipped": len(res.skipped),
        "y0_gap_2300_px": float(y0n - y0p),
    }


def evaluate_all(params: DepthParams, stacks: dict[str, np.ndarray], gts: dict[str, float], *,
                 holdout: list[str], extrap: list[str], disp_max: float, band: tuple[float, float] | None,
                 roi: tuple[int, int, int, int] | None = None) -> tuple[dict, dict, dict[str, dict]]:
    per = {s: evaluate_session(stacks[s], gts[s], params, disp_max=disp_max, band=band, roi=roi) for s in stacks}
    return summarize_holdout(per, holdout), summarize_holdout(per, extrap), per


def residual_vs_disparity(disp_by_depth: dict[int, np.ndarray], *, low: float, high: float,
                          ransac_th: float | None, ransac_seed: int, bin_px: float, reach_max: float,
                          inlier_th: float) -> list[dict]:
    """現行帯で当てた各距離・各枝の RANSAC 直線に対し、帯の外側も含めた全有効画素の残差 row − (a·d + b) を集計する。"""
    per_sign: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"pos": [], "neg": []}
    for depth in sorted(disp_by_depth):
        disp = disp_by_depth[depth]
        for label, sign in (("pos", +1), ("neg", -1)):
            d_band, y_band = extract_sign_points(disp, sign, low, high)
            if len(d_band) < 2:
                continue
            try:
                a, b, _ = ransac_fit_line(d_band, y_band, ransac_th, ransac_seed)
            except ValueError:
                continue
            d_all, y_all = extract_sign_points(disp, sign, 0.0, reach_max + 1.0)   # 帯の外側も含む（d=0 は除外）
            per_sign[label].append((np.abs(d_all), y_all - (a * d_all + b)))
    rows = []
    edges = np.arange(0.0, reach_max + bin_px, bin_px)
    for label, chunks in per_sign.items():
        if not chunks:
            continue
        ad = np.concatenate([c[0] for c in chunks])
        res = np.concatenate([c[1] for c in chunks])
        idx = np.digitize(ad, edges) - 1
        for k in range(len(edges) - 1):
            m = idx == k
            n = int(m.sum())
            row = {"sign": label, "bin_lo": float(edges[k]), "bin_hi": float(edges[k + 1]), "n_px": n}
            if n == 0:
                row.update({"resid_median_row": float("nan"), "resid_p25_row": float("nan"),
                            "resid_p75_row": float("nan"), "inlier_rate_th": float("nan")})
            else:
                r = res[m]
                q25, q50, q75 = np.percentile(r, [25, 50, 75])
                row.update({"resid_median_row": float(q50), "resid_p25_row": float(q25), "resid_p75_row": float(q75),
                            "inlier_rate_th": float(np.mean(np.abs(r) <= inlier_th))})
            rows.append(row)
    return rows


def _write_csv(rows: list[dict], columns: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def _heatmaps(grid_rows: list[dict], lows: list[float], highs: list[float], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for key, label in (("mae_mean_mm_nogate", "held-out MAE mean [mm] (no inference gate)"),
                       ("mae_mean_mm_bandgate", "held-out MAE mean [mm] (inference gate = calibration band)"),
                       ("abs_bias_max_mm_nogate", "held-out max |bias| [mm] (no inference gate)"),
                       ("valid_rate_mean_bandgate", "held-out valid rate (inference gate = calibration band)"),
                       ("depth_rmse_mm", "calibration depth RMSE [mm] (intercepts)")):
        z = np.full((len(highs), len(lows)), np.nan)
        for r in grid_rows:
            if r.get("calib_ok") and np.isfinite(r.get(key, np.nan)):
                z[highs.index(r["high"]), lows.index(r["low"])] = r[key]
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(z, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(lows)))
        ax.set_xticklabels([f"{v:g}" for v in lows])
        ax.set_yticks(range(len(highs)))
        ax.set_yticklabels([f"{v:g}" for v in highs])
        ax.set_xlabel("calibration band low [px]")
        ax.set_ylabel("calibration band high [px]")
        ax.set_title(label, fontsize=10)
        for i in range(len(highs)):
            for j in range(len(lows)):
                if np.isfinite(z[i, j]):
                    ax.text(j, i, f"{z[i, j]:.3g}", ha="center", va="center", fontsize=6, color="w")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_{key}.png", dpi=150)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="校正帯・RANSAC・推論ゲートのグリッド探索（A2・A4）")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sessions", nargs="+", metavar="NAME")
    src.add_argument("--manifest", metavar="CSV")
    parser.add_argument("--calib-dir", default=str(config.CALIB_IMGS_DIR))
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--frames-per-session", type=int, default=30, help="評価に使う連続フレーム数（既定 30。0=全部）")
    parser.add_argument("--kernel-size", type=int, default=config.KERNEL_SIZE)
    parser.add_argument("--search-range", type=int, default=config.SEARCH_RANGE_X)
    parser.add_argument("--edge-th", type=float, default=float(config.EDGE_TH))
    parser.add_argument("--lows", type=float, nargs="+", default=DEFAULT_LOWS)
    parser.add_argument("--highs", type=float, nargs="+", default=DEFAULT_HIGHS)
    parser.add_argument("--th", default=str(config.RANSAC_RESIDUAL_TH), help="grid で使う RANSAC 閾値（none=MAD）")
    parser.add_argument("--ths", nargs="+", default=DEFAULT_THS, help="th sweep の値（none=MAD 自動）")
    parser.add_argument("--fixed-low", type=float, default=None, help="inference band で固定する校正帯下限（既定: grid 最良）")
    parser.add_argument("--fixed-high", type=float, default=None)
    parser.add_argument("--objective", choices=("mae_mean_mm", "abs_bias_max_mm", "rmse_mean_mm"), default="mae_mean_mm")
    parser.add_argument("--residual-bin", type=float, default=0.25)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--captures-dir", default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-inference-band", action="store_true")
    args = parser.parse_args(argv)

    try:
        settings = DispSettings(kernel_size=args.kernel_size, search_range_x=args.search_range, edge_th=args.edge_th)
        th_grid = parse_th(args.th)
        ths = [parse_th(t) for t in args.ths]
        if args.manifest is not None:
            targets = load_manifest(Path(args.manifest))
        else:
            targets = [(name, parse_gt_from_session_name(name), "") for name in args.sessions]
        if args.frames_per_session < 0 or args.residual_bin <= 0:
            raise ValueError("--frames-per-session は 0 以上・--residual-bin は正")
        lows = sorted(set(args.lows))
        highs = sorted(set(args.highs))
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}")
        return 2

    cap_root = Path(args.captures_dir) if args.captures_dir else Path(config.CAPTURES_DIR)
    cache_root_dir = Path(args.cache_root) if args.cache_root else None
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(config.OUTPUTS_DIR) / "analysis" / f"calib_sensitivity_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 入力: 校正キャッシュ・評価キャッシュ ---
    try:
        calib_paths = ensure_calib_disparity(settings, Path(args.calib_dir), root=cache_root_dir)
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] calib: {e}")
        return 2
    disp_by_depth = {z: np.load(str(p)) for z, p in calib_paths.items()}
    z_lo, z_hi = min(disp_by_depth), max(disp_by_depth)
    stacks: dict[str, np.ndarray] = {}
    gts: dict[str, float] = {}
    for session, gt, note in targets:
        try:
            frames = list_frames(cap_root / session, skip_first=args.skip_first,
                                 max_frames=args.frames_per_session or None)
            stacks[session] = load_stack(ensure_capture_disparity(settings, session, frames, root=cache_root_dir))
        except (ValueError, FileNotFoundError) as e:
            print(f"[ERROR] {session}: {e}")
            return 2
        gts[session] = float(gt)
    holdout = [s for s in stacks if z_lo <= gts[s] <= z_hi]
    extrap = [s for s in stacks if s not in holdout]
    print(f"calib distances: {sorted(disp_by_depth)}  held-out: {holdout}  extrapolation: {extrap}  "
          f"frames/session: {[stacks[s].shape[0] for s in stacks]}", flush=True)

    # --- 1. grid ---
    t0 = time.perf_counter()
    grid_rows: list[dict] = []
    for low in lows:
        for high in highs:
            row: dict = {"low": low, "high": high, "th": args.th, "calib_ok": False, "error": ""}
            if not low < high:
                row["error"] = "low >= high"
                grid_rows.append(row)
                continue
            try:
                res = calibrate_from_disparity(disp_by_depth, low=low, high=high, ransac_th=th_grid)
            except (ValueError, RuntimeError, KeyError) as e:
                row["error"] = str(e)[:120]
                grid_rows.append(row)
                continue
            row["calib_ok"] = True
            row.update(calib_summary(res))
            for tag, band in (("nogate", None), ("bandgate", (low, high))):
                h, x, _ = evaluate_all(res.params, stacks, gts, holdout=holdout, extrap=extrap,
                                       disp_max=settings.disp_max, band=band)
                row.update({f"{k}_{tag}": h[k] for k in EVAL_KEYS})
                row.update({f"extrap_{k}_{tag}": x[k] for k in EVAL_KEYS})
            grid_rows.append(row)
            print(f"  low={low:g} high={high:g}: MAE nogate={row['mae_mean_mm_nogate']:.1f} "
                  f"bandgate={row['mae_mean_mm_bandgate']:.1f} |bias|max={row['abs_bias_max_mm_nogate']:.1f} "
                  f"rmse_calib={row['depth_rmse_mm']:.1f}", flush=True)
    print(f"grid: {len(grid_rows)} settings in {time.perf_counter() - t0:.1f} s", flush=True)
    _write_csv(grid_rows, GRID_COLUMNS, out_dir / "grid.csv")

    ok_rows = [r for r in grid_rows if r["calib_ok"] and np.isfinite(r[f"{args.objective}_nogate"])]
    if not ok_rows:
        print("[ERROR] 校正に成功した設定がありません")
        return 1
    best = min(ok_rows, key=lambda r: r[f"{args.objective}_nogate"])
    best_gate = min(ok_rows, key=lambda r: r[f"{args.objective}_bandgate"])
    current = next((r for r in ok_rows if r["low"] == config.DISP_PLOT_LOW and r["high"] == config.DISP_PLOT_HEIGHT), None)

    # --- 2. th sweep at best (low, high) ---
    th_rows: list[dict] = []
    for th in ths:
        row = {"low": best["low"], "high": best["high"], "th": "none" if th is None else f"{th:g}", "calib_ok": False,
               "error": ""}
        try:
            res = calibrate_from_disparity(disp_by_depth, low=best["low"], high=best["high"], ransac_th=th)
        except (ValueError, RuntimeError, KeyError) as e:
            row["error"] = str(e)[:120]
            th_rows.append(row)
            continue
        row["calib_ok"] = True
        row.update(calib_summary(res))
        for tag, band in (("nogate", None), ("bandgate", (best["low"], best["high"]))):
            h, x, _ = evaluate_all(res.params, stacks, gts, holdout=holdout, extrap=extrap,
                                   disp_max=settings.disp_max, band=band)
            row.update({f"{k}_{tag}": h[k] for k in EVAL_KEYS})
            row.update({f"extrap_{k}_{tag}": x[k] for k in EVAL_KEYS})
        th_rows.append(row)
    _write_csv(th_rows, GRID_COLUMNS, out_dir / "th_sweep.csv")

    # --- 3. inference band at fixed calibration band ---
    inf_rows: list[dict] = []
    if not args.skip_inference_band:
        f_low = args.fixed_low if args.fixed_low is not None else best["low"]
        f_high = args.fixed_high if args.fixed_high is not None else best["high"]
        try:
            res = calibrate_from_disparity(disp_by_depth, low=f_low, high=f_high, ransac_th=th_grid)
        except (ValueError, RuntimeError, KeyError) as e:
            print(f"[ERROR] inference band: 固定校正帯 ({f_low}, {f_high}) の校正に失敗: {e}")
            return 1
        for il in lows:
            for ih in highs:
                if not il < ih:
                    continue
                h, x, _ = evaluate_all(res.params, stacks, gts, holdout=holdout, extrap=extrap,
                                       disp_max=settings.disp_max, band=(il, ih))
                inf_rows.append({"calib_low": f_low, "calib_high": f_high, "th": args.th, "inf_low": il, "inf_high": ih,
                                 **{k: h[k] for k in EVAL_KEYS}, **{f"extrap_{k}": x[k] for k in EVAL_KEYS}})
        _write_csv(inf_rows, INF_COLUMNS, out_dir / "inference_band.csv")

    # --- 4. residual vs |d| at current band ---
    res_rows = residual_vs_disparity(disp_by_depth, low=config.DISP_PLOT_LOW, high=config.DISP_PLOT_HEIGHT,
                                     ransac_th=th_grid, ransac_seed=config.RANSAC_RANDOM_STATE,
                                     bin_px=args.residual_bin, reach_max=settings.reach_max,
                                     inlier_th=th_grid if th_grid is not None else config.RANSAC_RESIDUAL_TH)
    _write_csv(res_rows, RES_COLUMNS, out_dir / "residual_vs_disparity.csv")

    best_inf = min(inf_rows, key=lambda r: r[args.objective]) if inf_rows else None
    summary = {
        "settings": {"key": settings.key(), "disp_max": settings.disp_max, "reach_max": settings.reach_max},
        "calib_distances": sorted(disp_by_depth), "holdout": holdout, "extrapolation": extrap,
        "frames_per_session": {s: int(stacks[s].shape[0]) for s in stacks},
        "objective": args.objective, "grid_th": args.th, "lows": lows, "highs": highs,
        "current": {k: current.get(k) for k in ("low", "high") + tuple(f"{k}_{t}" for t in ("nogate", "bandgate") for k in EVAL_KEYS) + ("depth_rmse_mm",)} if current else None,
        "best_nogate": {k: best.get(k) for k in ("low", "high") + tuple(f"{k}_nogate" for k in EVAL_KEYS) + ("depth_rmse_mm",)},
        "best_bandgate": {k: best_gate.get(k) for k in ("low", "high") + tuple(f"{k}_bandgate" for k in EVAL_KEYS)},
        "best_inference_band": ({k: best_inf.get(k) for k in ("calib_low", "calib_high", "inf_low", "inf_high") + tuple(EVAL_KEYS)}
                                if best_inf else None),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_plots:
        _heatmaps(grid_rows, lows, highs, out_dir)

    print(f"\nbest (no gate, {args.objective}): low={best['low']:g} high={best['high']:g} "
          f"MAE={best['mae_mean_mm_nogate']:.1f} |bias|max={best['abs_bias_max_mm_nogate']:.1f} "
          f"valid={best['valid_rate_mean_nogate']:.3f}")
    print(f"best (band gate): low={best_gate['low']:g} high={best_gate['high']:g} "
          f"MAE={best_gate['mae_mean_mm_bandgate']:.1f} |bias|max={best_gate['abs_bias_max_mm_bandgate']:.1f} "
          f"valid={best_gate['valid_rate_mean_bandgate']:.3f}")
    if current:
        print(f"current (0.45, 7.5): MAE nogate={current['mae_mean_mm_nogate']:.1f} bandgate={current['mae_mean_mm_bandgate']:.1f} "
              f"|bias|max nogate={current['abs_bias_max_mm_nogate']:.1f}")
    if best_inf:
        print(f"best inference band @ calib ({best_inf['calib_low']:g}, {best_inf['calib_high']:g}): "
              f"inf=({best_inf['inf_low']:g}, {best_inf['inf_high']:g}) MAE={best_inf['mae_mean_mm']:.1f} "
              f"valid={best_inf['valid_rate_mean']:.3f}")
    print("\nth sweep:")
    for r in th_rows:
        if r["calib_ok"]:
            print(f"  th={r['th']:>5s}: MAE nogate={r['mae_mean_mm_nogate']:.1f} |bias|max={r['abs_bias_max_mm_nogate']:.1f} "
                  f"rmse_calib={r['depth_rmse_mm']:.1f} inlier={r['inlier_pos_mean']:.2f}/{r['inlier_neg_mean']:.2f}")
        else:
            print(f"  th={r['th']:>5s}: FAILED {r['error']}")
    print(f"out: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
