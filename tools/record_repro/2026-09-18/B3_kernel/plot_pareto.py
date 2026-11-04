"""処理解像度 × ZNCC 窓 K のスイープの集約と Pareto 図。

記録フォルダ: project_knowledge/records/2026-09-18/B3_kernel
入力（記録フォルダ基準・`tools.analyze_resolution --scale-edge-th` の出力の写し）:
  K<K>/settings.csv・K<K>/bench.csv・K<K>/summary.json（基準 K = 13, 21, 25, 29, 33）
  ../B3_scaleT/settings.csv・bench.csv（基準 K = 17。396/528/660 の行だけ使う）
出力（記録フォルダ）:
  summary_all.csv — 1 行 = 1 (解像度, 基準 K) 条件
  pareto.png      — 推論帯あり MAE・ゲートなし MAE・|bias|max を典型シーンの ZNCC p95 に対して描く
使い方（開発リポ root）: uv run python tools/record_repro/2026-09-18/B3_kernel/plot_pareto.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parents[4] / "project_knowledge" / "records" / "2026-09-18" / "B3_kernel"  # 入出力は記録フォルダ
RUNS = {17: HERE.parent / "B3_scaleT", **{k: HERE / f"K{k}" for k in (13, 21, 25, 29, 33)}}
WIDTHS = (396, 528, 660)
# 解像度の色は固定順（3 色は全ペアで色覚の検証済み）
COLORS = {396: "#2a78d6", 528: "#eb6834", 660: "#1baf7a"}
BUDGET_MS = 33.3   # 30 fps
TARGET_MS = 30.0   # E2E の前後処理 ≈ 3 ms を差し引いた ZNCC の目安
KEEP = [
    "width", "base_kernel", "kernel_size", "search_range_x", "edge_th",
    "fit_rms_pos_px", "fit_rms_neg_px", "depth_rmse_mm",
    "interior_mae_mean_mm_nogate", "interior_mae_mean_mm_bandgate",
    "interior_abs_bias_max_mm_nogate", "interior_abs_bias_max_mm_bandgate",
    "interior_valid_rate_mean_nogate", "interior_valid_rate_mean_bandgate",
    "extrap_mae_mean_mm_nogate", "extrap_mae_mean_mm_bandgate",
    "sigma_median_base_px", "flip_rate_mean", "seconds_disparity",
    "p95_ms_scene_01", "p95_ms_dist_2300", "edge_px_scene_01", "edge_px_dist_2300",
]


def load_run(base_k: int, run_dir: Path) -> pd.DataFrame | None:
    """1 回の analyze_resolution 出力を読み、bench の p95 を解像度ごとに横持ちにして返す。"""
    if not (run_dir / "settings.csv").is_file():
        return None
    s = pd.read_csv(run_dir / "settings.csv")
    s = s[s["width"].isin(WIDTHS)].copy()
    meta = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert meta["base"]["kernel_size"] == base_k, (run_dir, meta["base"])
    s["base_kernel"] = base_k
    b = pd.read_csv(run_dir / "bench.csv")
    for sess in ("scene_01", "dist_2300"):
        sub = b[b["session"] == sess].set_index("width")
        s[f"p95_ms_{sess}"] = s["width"].map(sub["p95_ms"])
        s[f"edge_px_{sess}"] = s["width"].map(sub["edge_px_bench"])
    return s[KEEP]


def pareto_mask(df: pd.DataFrame, x: str, y: str) -> pd.Series:
    """x（時間）も y（誤差）も小さい方が良いときの Pareto 最適点（他のどの点にも両方で負けない）。"""
    keep = []
    for i, row in df.iterrows():
        dominated = ((df[x] <= row[x]) & (df[y] <= row[y]) & ((df[x] < row[x]) | (df[y] < row[y]))).any()
        keep.append(not dominated)
    return pd.Series(keep, index=df.index)


def main() -> int:
    frames = [f for k, d in RUNS.items() if (f := load_run(k, d)) is not None]
    if not frames:
        print("[ERROR] 入力がありません")
        return 2
    df = pd.concat(frames, ignore_index=True).sort_values(["width", "base_kernel"]).reset_index(drop=True)
    df["pareto_bandgate_typical"] = pareto_mask(df, "p95_ms_scene_01", "interior_mae_mean_mm_bandgate")
    df["within_budget_typical"] = df["p95_ms_scene_01"] <= BUDGET_MS
    df.to_csv(HERE / "summary_all.csv", index=False, lineterminator="\n")

    # 予算内で MAE 最小の条件（同点は計算の軽い方）
    feas = df[df["within_budget_typical"]].sort_values(["interior_mae_mean_mm_bandgate", "p95_ms_scene_01"])
    print(f"conditions: {len(df)}  (base K: {sorted(df['base_kernel'].unique())})")
    cols = ["width", "base_kernel", "kernel_size", "edge_th", "interior_mae_mean_mm_bandgate",
            "interior_mae_mean_mm_nogate", "interior_abs_bias_max_mm_bandgate", "interior_valid_rate_mean_nogate",
            "p95_ms_scene_01", "p95_ms_dist_2300", "pareto_bandgate_typical"]
    print(df[cols].round(3).to_string(index=False))
    if len(feas):
        best = feas.iloc[0]
        print(f"best within {BUDGET_MS} ms (typical): W={int(best['width'])} K'={int(best['kernel_size'])} "
              f"bandgate MAE={best['interior_mae_mean_mm_bandgate']:.1f} mm p95={best['p95_ms_scene_01']:.1f} ms")

    panels = [
        ("interior_mae_mean_mm_bandgate", "interior MAE, inference gate = band [mm]"),
        ("interior_mae_mean_mm_nogate", "interior MAE, no inference gate [mm]"),
        ("interior_abs_bias_max_mm_bandgate", "interior |bias| max, band gate [mm]"),
        ("interior_valid_rate_mean_nogate", "interior valid rate, no gate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (col, label) in zip(axes.ravel(), panels):
        for w in WIDTHS:
            sub = df[df["width"] == w].sort_values("kernel_size")
            if sub.empty:
                continue
            ax.plot(sub["p95_ms_scene_01"], sub[col], "-o", color=COLORS[w], lw=1.5, ms=6,
                    label=f"{w}x{int(w * 2 / 3)} (S'={int(sub['search_range_x'].iloc[0])}, T'={sub['edge_th'].iloc[0]:g})")
            for _, r in sub.iterrows():
                ax.annotate(f"K'={int(r['kernel_size'])}", (r["p95_ms_scene_01"], r[col]),
                            textcoords="offset points", xytext=(5, 4), fontsize=8, color="#444444")
        if col == "interior_mae_mean_mm_bandgate":
            p = df[df["pareto_bandgate_typical"]]
            ax.scatter(p["p95_ms_scene_01"], p[col], s=140, facecolors="none", edgecolors="#222222",
                       lw=1.2, zorder=3, label="Pareto-optimal (time vs MAE)")
        ax.axvline(BUDGET_MS, color="#888888", ls="--", lw=1)
        ax.axvline(TARGET_MS, color="#bbbbbb", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("ZNCC p95 on scene_01 (typical scene) [ms]  (dashed 33.3 ms = 30 fps, dotted 30 ms)")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("B3-2: processing resolution x ZNCC window K  (S and band scaled with r, T' = 30/r, "
                 "10-distance calibration, 17 sessions x 30 frames)", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "pareto.png", dpi=130)
    print(f"wrote {HERE / 'summary_all.csv'} and {HERE / 'pareto.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
