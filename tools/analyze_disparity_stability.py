"""tools/analyze_disparity_stability.py — 視差の時間的安定性の解析（最適化実験ランブック A1）。

静止した板を録画した評価セッション（`data/captures/dist_<mm>/`）では、被写体が動かないので視差 d のフレーム間の
ばらつきは測定ノイズそのものである。各画素について有効フレーム数・平均 μ・標準偏差 σ_d・符号反転率
（多数派の符号と異なるフレームの割合）を計算し、|μ|（視差の大きさ）とエッジ強度（赤チャンネルの水平 Sobel
絶対値・時間平均）でビン分けして集計する。

何に使うか（ランブック「対象定数と最適化の設計」）:
  - 校正帯の下限（DISP_PLOT_LOW）: |d| が小さいほど符号（pos/neg の枝選択）が不安定になる。反転率が十分小さく
    なる最小 |d| が「符号を信頼できる下限」の実測値。
  - 校正帯の上限（DISP_PLOT_HEIGHT）: 大ぼけ側で σ_d が増え始める |d|。
  - エッジ閾値（EDGE_TH）: エッジ強度が低いほど σ_d が大きい。σ_d が急増する強度が閾値の実測根拠。

出力（--out-dir、既定 outputs/analysis/disparity_stability_<日時>/）:
  by_disparity.csv  |μ| ビンごとの画素数・σ_d の中央値/平均/p90・反転率・有効率・エッジ強度中央値（session 列は "all" と各セッション）
  by_edge.csv       エッジ強度ビンごとの画素数・σ_d の中央値・反転率・有効率・|μ| 中央値
  summary.json      設定・フレーム数・参考値（反転率 < flip_ref となる最小 |μ| ビン、σ_d が小視差域の最小値の 2 倍を超える |μ| ビン 等）
  *.png             図（--no-plots で省略）

実行例:
  uv run python -m tools.analyze_disparity_stability --manifest outputs\\distance_manifest.csv --skip-first 10
"""
from __future__ import annotations

# --- tools/ から root の config 等を import するためのブートストラップ ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import warnings
from datetime import datetime

import cv2
import numpy as np

import config
from checks.evaluate_distance_accuracy import load_manifest, parse_gt_from_session_name
from tools.analysis_common import (
    DispSettings,
    edge_strength_of_proc,
    ensure_capture_disparity,
    list_frames,
    load_stack,
    preprocess_to,
)

DISP_COLUMNS = ["session", "bin_lo", "bin_hi", "n_px", "sigma_median", "sigma_mean", "sigma_p90",
                "flip_rate_mean", "valid_frac_mean", "edge_median"]
EDGE_COLUMNS = ["session", "bin_lo", "bin_hi", "n_px", "sigma_median", "sigma_mean", "flip_rate_mean",
                "valid_frac_mean", "abs_mu_median"]


def pixel_stats(stack: np.ndarray) -> dict[str, np.ndarray]:
    """(N,H,W) の視差スタック → 画素ごとの n_valid・mu・sigma（ddof=1）・flip_rate・valid_frac（各 (H,W)）。

    sigma は n_valid >= 2 の画素のみ（他は NaN）。flip_rate は多数派符号と異なる有効フレームの割合で、
    多数派が決まらない画素（正負同数）は NaN。
    """
    if stack.ndim != 3 or stack.shape[0] < 1:
        raise ValueError(f"pixel_stats: (N,H,W) を想定。実際は {stack.shape}")
    n = stack.shape[0]
    valid = ~np.isnan(stack)
    n_valid = valid.sum(axis=0)
    # 全フレーム無効の画素で nanmean/nanstd が出す RuntimeWarning（Mean of empty slice）は仕様どおり NaN なので抑制する
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(stack, axis=0)
        sigma = np.full(mu.shape, np.nan, dtype=np.float64)
        ge2 = n_valid >= 2
        # nanstd(ddof=1) は n_valid < 2 で警告を出すため、対象画素だけ計算する
        sigma[ge2] = np.nanstd(stack[:, ge2], axis=0, ddof=1)
        sgn = np.sign(stack)
        s_sum = np.nansum(sgn, axis=0)
        majority = np.sign(s_sum)
        flips = (sgn != majority[None, :, :]) & valid
        flip_rate = np.where(n_valid > 0, flips.sum(axis=0) / np.maximum(n_valid, 1), np.nan)
        flip_rate = np.where(majority == 0, np.nan, flip_rate)
    return {
        "n_valid": n_valid.astype(np.int64),
        "mu": mu.astype(np.float64),
        "sigma": sigma,
        "flip_rate": flip_rate.astype(np.float64),
        "valid_frac": (n_valid / n).astype(np.float64),
    }


def _bin_rows(session: str, x: np.ndarray, edges: np.ndarray, fields: dict[str, np.ndarray],
              columns: list[str]) -> list[dict]:
    """x をビン分け（[lo, hi)・最後のビンは hi を含む）し、fields の統計を列名の規則で出す。"""
    rows = []
    idx = np.digitize(x, edges) - 1
    idx[x == edges[-1]] = len(edges) - 2
    for b in range(len(edges) - 1):
        m = idx == b
        row: dict = {"session": session, "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
                     "n_px": int(m.sum())}
        for col in columns[4:]:
            if row["n_px"] == 0:
                row[col] = float("nan")
                continue
            name, stat = col.rsplit("_", 1)
            vals = fields[name][m]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                row[col] = float("nan")
            elif stat == "median":
                row[col] = float(np.median(vals))
            elif stat == "mean":
                row[col] = float(np.mean(vals))
            elif stat == "p90":
                row[col] = float(np.percentile(vals, 90))
            else:
                raise ValueError(f"未知の統計名: {col}")
        rows.append(row)
    return rows


def bin_by_disparity(session: str, st: dict[str, np.ndarray], edge_mean: np.ndarray, *,
                     disp_bin: float, disp_max: float, min_valid_frac: float) -> list[dict]:
    sel = (st["valid_frac"] >= min_valid_frac) & (st["n_valid"] >= 2)
    edges = np.arange(0.0, disp_max + disp_bin, disp_bin)
    fields = {"sigma": st["sigma"][sel], "flip_rate": st["flip_rate"][sel],
              "valid_frac": st["valid_frac"][sel], "edge": edge_mean[sel].astype(np.float64)}
    return _bin_rows(session, np.abs(st["mu"][sel]), edges, fields, DISP_COLUMNS)


def bin_by_edge(session: str, st: dict[str, np.ndarray], edge_mean: np.ndarray, *,
                edge_bin: float, min_valid_frac: float) -> list[dict]:
    sel = (st["valid_frac"] >= min_valid_frac) & (st["n_valid"] >= 2)
    edges = np.arange(0.0, 255.0 + edge_bin, edge_bin)
    fields = {"sigma": st["sigma"][sel], "flip_rate": st["flip_rate"][sel],
              "valid_frac": st["valid_frac"][sel], "abs_mu": np.abs(st["mu"][sel])}
    return _bin_rows(session, edge_mean[sel].astype(np.float64), edges, fields, EDGE_COLUMNS)


def reference_thresholds(rows_all: list[dict], *, flip_ref: float = 0.05, sigma_ratio: float = 2.0,
                         min_px: int = 200) -> dict:
    """統合ビン表から参考値を抜く（判断はユーザー。ここは機械的な抽出のみ）。

    - flip_below_ref_from_px: 反転率が flip_ref 未満になり、それ以降のビンでも超えない最小の bin_lo
    - sigma_floor: 画素数 >= min_px のビンでの σ_d 中央値の最小値と、その bin_lo
    - sigma_exceeds_ratio_from_px: σ_d 中央値が sigma_floor × sigma_ratio を初めて超える bin_lo（floor より上側）
    """
    rows = [r for r in rows_all if r["n_px"] >= min_px and np.isfinite(r["sigma_median"])]
    out: dict = {"flip_ref": flip_ref, "sigma_ratio": sigma_ratio, "min_px": min_px}
    if not rows:
        return out
    flip_from = None
    for i, r in enumerate(rows):
        if np.isfinite(r["flip_rate_mean"]) and all(
                (not np.isfinite(q["flip_rate_mean"])) or q["flip_rate_mean"] < flip_ref for q in rows[i:]):
            flip_from = r["bin_lo"]
            break
    out["flip_below_ref_from_px"] = flip_from
    k = int(np.argmin([r["sigma_median"] for r in rows]))
    out["sigma_floor"] = rows[k]["sigma_median"]
    out["sigma_floor_bin_lo"] = rows[k]["bin_lo"]
    exceed = None
    for r in rows[k:]:
        if r["sigma_median"] > rows[k]["sigma_median"] * sigma_ratio:
            exceed = r["bin_lo"]
            break
    out["sigma_exceeds_ratio_from_px"] = exceed
    return out


def _write_csv(rows: list[dict], columns: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def _plot(rows_disp: list[dict], rows_edge: list[dict], out_dir: Path) -> None:
    """図を PNG 保存する。軸ラベルは既定フォント（DejaVu Sans）で描ける英語にする（日本語は豆腐になる）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sessions = sorted({r["session"] for r in rows_disp} - {"all"})

    def series(rows, session, ycol):
        rs = [r for r in rows if r["session"] == session and r["n_px"] > 0 and np.isfinite(r[ycol])]
        return [(r["bin_lo"] + r["bin_hi"]) / 2 for r in rs], [r[ycol] for r in rs]

    for rows, ycol, fname, xlabel, ylabel in (
            (rows_disp, "sigma_median", "sigma_vs_disp.png", "|d| [px] (temporal mean)", "sigma_d median [px]"),
            (rows_disp, "flip_rate_mean", "flip_vs_disp.png", "|d| [px] (temporal mean)", "sign-flip rate"),
            (rows_edge, "sigma_median", "sigma_vs_edge.png", "edge strength (|Sobel_x| of red, temporal mean)", "sigma_d median [px]")):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for s in sessions:
            x, y = series(rows, s, ycol)
            ax.plot(x, y, color="0.7", lw=0.8)
        x, y = series(rows, "all", ycol)
        ax.plot(x, y, color="C0", lw=2.0, label="all sessions (median)")
        if ycol == "sigma_median":                       # 誤マッチの始まりは中央値より p90／平均に先に出る
            second = "sigma_p90" if rows is rows_disp else "sigma_mean"
            x90, y90 = series(rows, "all", second)
            ax.plot(x90, y90, color="C1", lw=1.5, ls="--", label=f"all sessions ({second.split(chr(95))[1]})")
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="視差の時間的安定性（σ_d・符号反転率 vs |d|・エッジ強度）の集計（A1）")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sessions", nargs="+", metavar="NAME")
    src.add_argument("--manifest", metavar="CSV")
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=config.KERNEL_SIZE)
    parser.add_argument("--search-range", type=int, default=config.SEARCH_RANGE_X)
    parser.add_argument("--edge-th", type=float, default=float(config.EDGE_TH))
    parser.add_argument("--disp-bin", type=float, default=0.1, help="|d| のビン幅 [px]（既定 0.1）")
    parser.add_argument("--edge-bin", type=float, default=10.0, help="エッジ強度のビン幅（既定 10）")
    parser.add_argument("--min-valid-frac", type=float, default=0.5,
                        help="集計対象にする画素の最小有効率（既定 0.5。時間統計が意味を持つ画素に限定）")
    parser.add_argument("--flip-ref", type=float, default=0.05, help="参考値抽出に使う反転率の基準（既定 0.05）")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--cache-root", default=None, help="視差キャッシュの親（既定 config.OUTPUTS_DIR）")
    parser.add_argument("--captures-dir", default=None, help="captures の親（既定 config.CAPTURES_DIR。テスト用）")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    try:
        settings = DispSettings(kernel_size=args.kernel_size, search_range_x=args.search_range, edge_th=args.edge_th)
        if args.manifest is not None:
            targets = load_manifest(Path(args.manifest))
        else:
            targets = [(name, parse_gt_from_session_name(name), "") for name in args.sessions]
        if not (0.0 < args.min_valid_frac <= 1.0) or args.disp_bin <= 0 or args.edge_bin <= 0:
            raise ValueError("--min-valid-frac は (0,1]・--disp-bin/--edge-bin は正")
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}")
        return 2

    cap_root = Path(args.captures_dir) if args.captures_dir else Path(config.CAPTURES_DIR)
    cache_root_dir = Path(args.cache_root) if args.cache_root else None
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(config.OUTPUTS_DIR) / "analysis" / f"disparity_stability_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_disp: list[dict] = []
    rows_edge: list[dict] = []
    pooled = {"mu": [], "sigma": [], "flip_rate": [], "valid_frac": [], "n_valid": [], "edge": []}
    frames_used: dict[str, int] = {}
    for session, gt, note in targets:
        cap_dir = cap_root / session
        try:
            frames = list_frames(cap_dir, skip_first=args.skip_first, max_frames=args.max_frames)
            paths = ensure_capture_disparity(settings, session, frames, root=cache_root_dir)
        except (ValueError, FileNotFoundError) as e:
            print(f"[ERROR] {session}: {e}")
            return 2
        stack = load_stack(paths)
        st = pixel_stats(stack)
        edge_acc = np.zeros(stack.shape[1:], dtype=np.float64)
        for p in frames:
            raw = cv2.imread(str(p))
            if raw is None:
                print(f"[ERROR] {session}: png を読めません: {p}")
                return 2
            edge_acc += edge_strength_of_proc(preprocess_to(raw, settings.image_size))
        edge_mean = edge_acc / len(frames)
        frames_used[session] = len(frames)
        rows_disp += bin_by_disparity(session, st, edge_mean, disp_bin=args.disp_bin, disp_max=settings.disp_max,
                                      min_valid_frac=args.min_valid_frac)
        rows_edge += bin_by_edge(session, st, edge_mean, edge_bin=args.edge_bin, min_valid_frac=args.min_valid_frac)
        for k in ("mu", "sigma", "flip_rate", "valid_frac", "n_valid"):
            pooled[k].append(st[k].ravel())
        pooled["edge"].append(edge_mean.ravel())
        sel = (st["valid_frac"] >= args.min_valid_frac) & (st["n_valid"] >= 2)
        print(f"  {session}: frames={len(frames)} px_used={int(sel.sum())} "
              f"sigma_median={np.nanmedian(st['sigma'][sel]):.3f} px "
              f"flip_mean={np.nanmean(st['flip_rate'][sel]):.4f}", flush=True)

    st_all = {k: np.concatenate(v) for k, v in pooled.items() if k != "edge"}
    edge_all = np.concatenate(pooled["edge"])
    rows_disp = bin_by_disparity("all", st_all, edge_all, disp_bin=args.disp_bin, disp_max=settings.disp_max,
                                 min_valid_frac=args.min_valid_frac) + rows_disp
    rows_edge = bin_by_edge("all", st_all, edge_all, edge_bin=args.edge_bin,
                            min_valid_frac=args.min_valid_frac) + rows_edge
    _write_csv(rows_disp, DISP_COLUMNS, out_dir / "by_disparity.csv")
    _write_csv(rows_edge, EDGE_COLUMNS, out_dir / "by_edge.csv")
    ref = reference_thresholds([r for r in rows_disp if r["session"] == "all"], flip_ref=args.flip_ref)
    summary = {
        "settings": {"kernel_size": settings.kernel_size, "search_range_x": settings.search_range_x,
                     "edge_th": settings.edge_th, "image_size": list(settings.image_size), "key": settings.key()},
        "skip_first": args.skip_first, "max_frames": args.max_frames, "min_valid_frac": args.min_valid_frac,
        "disp_bin": args.disp_bin, "edge_bin": args.edge_bin, "frames_used": frames_used,
        "reference": ref, "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_plots:
        _plot(rows_disp, rows_edge, out_dir)

    print("\n|d| bin [px]  n_px  sigma_median  flip_rate_mean  valid_frac_mean  (all sessions, first 30 bins)")
    for r in [r for r in rows_disp if r["session"] == "all"][:30]:
        print(f"  {r['bin_lo']:4.1f}-{r['bin_hi']:4.1f}  {r['n_px']:8d}  {r['sigma_median']:8.4f}  "
              f"{r['flip_rate_mean']:8.4f}  {r['valid_frac_mean']:8.3f}")
    print(f"\nreference: {ref}")
    print(f"out: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
