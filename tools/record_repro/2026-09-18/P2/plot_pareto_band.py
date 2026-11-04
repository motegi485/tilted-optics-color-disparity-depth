"""第 2 パス（528×352・K=25・S=41）の集約: T × 推論帯の密度 vs 精度の Pareto。

記録フォルダ: project_knowledge/records/2026-09-18/P2
入力（記録フォルダ基準・各ツールの出力の写し）:
  edge_th/settings.csv                 — tools.analyze_disparity_settings の T スイープ（K=25・S=41・校正帯 (0.45, 7.5)）
  T<T>/inference_band.csv・grid.csv・th_sweep.csv・summary.json — tools.analyze_calib_sensitivity（T = 20, 30, 45, 60）
出力（記録フォルダ）:
  summary_band.csv — 1 行 = 1 (T, 推論帯 low′, high′) 条件（校正帯は各 T の grid 最良で固定）＋ 推論ゲートなしの行
  pareto_band.png  — held-out MAE・|bias|max・外挿 MAE を有効画素率に対して描く（T は色・Pareto 最適点に印）
使い方（開発リポ root）: uv run python tools/record_repro/2026-09-18/P2/plot_pareto_band.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parents[4] / "project_knowledge" / "records" / "2026-09-18" / "P2"  # 入出力は記録フォルダ
TS = (20, 30, 45, 60)
# T の色は固定順（3 色は全ペアで色覚の検証済み）。
# T=45 は無彩色＋別マーカーで従属系列として描く（4 色目を使わない）
COLORS = {20: "#2a78d6", 30: "#eb6834", 60: "#1baf7a", 45: "#8a8a8a"}
MARKERS = {20: "o", 30: "o", 60: "o", 45: "^"}


def load_band(t: int) -> pd.DataFrame | None:
    d = HERE / f"T{t}"
    if not (d / "inference_band.csv").is_file():
        return None
    df = pd.read_csv(d / "inference_band.csv")
    df.insert(0, "edge_th", t)
    # 推論ゲートなし（gate1 のみ）の行を grid.csv の同じ校正帯から足す
    g = pd.read_csv(d / "grid.csv")
    cl, ch, th = df["calib_low"].iloc[0], df["calib_high"].iloc[0], df["th"].iloc[0]
    row = g[(g["low"] == cl) & (g["high"] == ch) & (g["th"] == th)]
    if len(row) == 1:
        r = row.iloc[0]
        extra = {"edge_th": t, "calib_low": cl, "calib_high": ch, "th": th, "inf_low": float("nan"), "inf_high": float("nan")}
        for k in ("mae_mean_mm", "abs_bias_max_mm", "rmse_mean_mm", "valid_rate_mean"):
            extra[k] = r[f"{k}_nogate"]
            extra[f"extrap_{k}"] = r[f"extrap_{k}_nogate"]
        df = pd.concat([df, pd.DataFrame([extra])], ignore_index=True)
    return df


def pareto_mask(df: pd.DataFrame, x_more_is_better: str, y_less_is_better: str) -> pd.Series:
    """密度（大きいほど良い）と誤差（小さいほど良い）の Pareto 最適点。"""
    x, y = df[x_more_is_better].to_numpy(), df[y_less_is_better].to_numpy()
    keep = []
    for i in range(len(df)):
        dominated = ((x >= x[i]) & (y <= y[i]) & ((x > x[i]) | (y < y[i]))).any()
        keep.append(not dominated)
    return pd.Series(keep, index=df.index)


def main() -> int:
    frames = [f for t in TS if (f := load_band(t)) is not None]
    if not frames:
        print("[ERROR] T<T>/inference_band.csv がありません")
        return 2
    df = pd.concat(frames, ignore_index=True)
    df["pareto_mae_vs_valid"] = pareto_mask(df, "valid_rate_mean", "mae_mean_mm")
    df.to_csv(HERE / "summary_band.csv", index=False, lineterminator="\n")

    # T スイープ（校正帯 (0.45, 7.5) ゲート・ゲートなし）
    sweep = None
    if (HERE / "edge_th" / "settings.csv").is_file():
        sweep = pd.read_csv(HERE / "edge_th" / "settings.csv")
        print("=== T sweep (analyze_disparity_settings, K=25 S=41)")
        cols = [c for c in ("edge_th", "mae_mean_mm_nogate", "mae_mean_mm_bandgate", "abs_bias_max_mm_nogate",
                            "abs_bias_max_mm_bandgate", "valid_rate_mean_nogate", "valid_rate_mean_bandgate",
                            "sigma_median_px", "p95_ms", "edge_px_bench") if c in sweep.columns]
        print(sweep[cols].round(3).to_string(index=False))

    print("=== Pareto-optimal (valid rate vs held-out MAE) over T x inference band")
    p = df[df["pareto_mae_vs_valid"]].sort_values("valid_rate_mean", ascending=False)
    print(p[["edge_th", "calib_low", "calib_high", "inf_low", "inf_high", "mae_mean_mm", "abs_bias_max_mm",
             "valid_rate_mean", "extrap_mae_mean_mm"]].round(3).to_string(index=False))

    panels = [("mae_mean_mm", "held-out MAE [mm] (1500-3100, 9 distances)"),
              ("abs_bias_max_mm", "held-out |bias| max [mm]"),
              ("extrap_mae_mean_mm", "extrapolation MAE [mm] (1000-1300, 3300-4000)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (col, label) in zip(axes, panels):
        for t in TS:
            sub = df[df["edge_th"] == t]
            if sub.empty:
                continue
            band = sub[sub["inf_low"].notna()]
            ax.scatter(band["valid_rate_mean"], band[col], s=22, color=COLORS[t], alpha=0.75, marker=MARKERS[t],
                       label=f"T={t} (calib band ({band['calib_low'].iloc[0]:g}, {band['calib_high'].iloc[0]:g}))")
            nog = sub[sub["inf_low"].isna()]
            ax.scatter(nog["valid_rate_mean"], nog[col], s=90, marker="s", color=COLORS[t], edgecolors="#222222",
                       lw=1.0, zorder=3, label=f"T={t}, no inference gate")
        if col == "mae_mean_mm":
            pp = df[df["pareto_mae_vs_valid"]]
            ax.scatter(pp["valid_rate_mean"], pp[col], s=110, facecolors="none", edgecolors="#222222", lw=1.2,
                       zorder=4, label="Pareto-optimal (density vs MAE)")
            for _, r in pp.iterrows():
                if pd.notna(r["inf_low"]):
                    ax.annotate(f"({r['inf_low']:g},{r['inf_high']:g})", (r["valid_rate_mean"], r[col]),
                                textcoords="offset points", xytext=(5, 3), fontsize=7, color="#444444")
        ax.set_xlabel("valid pixel rate (held-out mean)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Second pass at 528x352, K=25, S=41: edge threshold T x inference band (low', high')  "
                 "[10-distance calibration, 17 sessions x 30 frames]", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "pareto_band.png", dpi=130)
    print(f"wrote {HERE / 'summary_band.csv'} and {HERE / 'pareto_band.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
