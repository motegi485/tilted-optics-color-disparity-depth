"""fig6_parameter_selection.py — パラメータ選定の 3 パネルの図。

入力（読み取りのみ。project_knowledge/records/ 基準）:
  (a) 2026-09-16/B2/predicted_max_disparity.csv（採用係数から予測した画面端の最大視差）と
      distribution.csv（17 距離の実測 |d| 分布・K17/S31/T30）
  (b) 2026-09-18/B3_recheck/summary_recheck.csv（最終 T・推論帯で解像度 3 水準 × K 4 水準）
  (c) 2026-09-18/P2/summary_band.csv（528×352・K25・S41 で T 4 水準 × 推論帯の格子）
自己照合: 記録の値（予測と実測 p99 の一致・528/K25 の MAE と p95・膝の MAE と有効率・Pareto 前線の点数）と一致すること。

実行（リポジトリ root）: uv run python tools/record_repro/2026-10-05/figures/fig6_parameter_selection.py
出力: project_knowledge/records/2026-10-05/figures/ の fig6_parameter_selection.pdf / .png
"""
from __future__ import annotations

import pandas as pd

import figstyle as fs

BUDGET_MS = 33.3                      # 30 fps の時間予算（典型シーンの ZNCC p95）
REACH = {31: 14.5, 41: 19.5}          # ZNCC の到達上限 (S−1)/2 − 0.5 [px]


def main() -> None:
    pred = pd.read_csv(fs.RECORDS / "2026-09-16" / "B2" / "predicted_max_disparity.csv")
    dist = pd.read_csv(fs.RECORDS / "2026-09-16" / "B2" / "distribution.csv").sort_values("gt_mm")
    rec = pd.read_csv(fs.RECORDS / "2026-09-18" / "B3_recheck" / "summary_recheck.csv")
    band = pd.read_csv(fs.RECORDS / "2026-09-18" / "P2" / "summary_band.csv")

    # --- 自己照合 (a): 予測と実測 p99 が 1000〜2300 mm で ±0.5 px 以内 ---
    pred_at = pred.set_index("z_mm")["d_max_abs"]
    fs.check(set(dist.gt_mm.astype(float)) <= set(pred_at.index), "実測 17 距離がすべて予測表の距離に含まれる")
    near = dist[dist.gt_mm <= 2300]
    diff = (near.p99 - near.gt_mm.astype(float).map(pred_at)).abs()
    fs.check(len(dist) == 17 and diff.max() <= 0.5,
             f"1000〜2300 mm の |実測 p99 − 予測| 最大 {diff.max():.2f} px ≤ 0.5（{len(near)} 距離）")
    fs.check(round(pred_at[1000.0], 1) == 13.8 and round(float(dist[dist.gt_mm == 1000].p99.iloc[0]), 1) == 13.4,
             "1000 mm: 予測 13.8 px・実測 p99 13.4 px（記録票 §5.3）")

    # --- 自己照合 (b): 528/K25・528/K29 の MAE と p95 ---
    r = rec.set_index(["W", "baseK"])
    fs.check(len(rec) == 12, "解像度 3 水準 × K 4 水準 = 12 条件")
    fs.check((r.loc[(528, 25), "MAE_band"], r.loc[(528, 25), "p95_typ"]) == (52.6, 23.7)
             and (r.loc[(528, 29), "MAE_band"], r.loc[(528, 29), "p95_typ"]) == (50.4, 30.1),
             "528/K25 = 52.6 mm・23.7 ms、528/K29 = 50.4 mm・30.1 ms")
    fs.check((rec[rec.W == 660].p95_typ > BUDGET_MS).all(), "660 は全条件で予算外")

    # --- 自己照合 (c): 膝の MAE と有効率、Pareto 前線の点数 ---
    knee = band[(band.edge_th == 20) & (band.inf_low == 0.45) & (band.inf_high == 5.0)]
    fs.check(len(knee) == 1 and round(float(knee.mae_mean_mm.iloc[0]), 1) == 50.2
             and round(float(knee.valid_rate_mean.iloc[0]), 3) == 0.313,
             "膝 T20・(0.45, 5): MAE 50.2 mm・有効率 0.313")
    front = band[band.pareto_mae_vs_valid].sort_values("valid_rate_mean")
    mid = front[(front.valid_rate_mean >= 0.17) & (front.valid_rate_mean <= 0.48)]
    fs.check(len(mid) == 29 and int((mid.edge_th == 20).sum()) == 27,
             f"Pareto 前線: 有効率 0.17〜0.48 の {len(mid)} 点中 T=20 が {int((mid.edge_th == 20).sum())} 点（記録票 §6.4: 29 点中 27 点）")

    # --- 描画 ---
    fs.apply_style()
    fig, axes = fs.plt.subplots(1, 3, figsize=(fs.TEXT_WIDTH_IN, 2.45),
                                gridspec_kw={"wspace": 0.34})

    # (a) 探索範囲 S
    ax = axes[0]
    for s, reach in REACH.items():
        ax.axhline(reach, color=fs.MUTED, lw=0.8, ls="--", zorder=1)
        ax.text(4000, reach + 0.3, f"reach of S = {s} ({reach} px)", ha="right", va="bottom",
                fontsize=6.5, color=fs.INK2)
    ax.plot(pred.z_mm, pred.d_max_abs, color=fs.SERIES[0], lw=1.4, zorder=2,
            label="predicted")
    ax.plot(dist.gt_mm, dist.p99, ls="none", marker=fs.MARKERS[1], color=fs.SERIES[1], mfc="white",
            mew=1.0, ms=4, zorder=3, label="measured p99")
    ax.set_xlabel("Distance [mm]")
    ax.set_ylabel("Max. |disparity| at image edge [px]")
    ax.set_xlim(900, 4100)
    ax.set_xticks(range(1000, 4001, 1000))
    ax.set_ylim(0, 21.5)
    ax.legend(loc="lower right", handlelength=1.8)
    fs.panel_label(ax, "(a)", "Search range S")

    # (b) 処理解像度 × K
    ax = axes[1]
    ax.axvline(BUDGET_MS, color=fs.MUTED, lw=0.8, ls="--", zorder=1)
    ax.text(BUDGET_MS + 1.2, 56.0, "30 fps budget", ha="left", va="bottom", rotation=90,
            fontsize=6.5, color=fs.INK2)
    for i, (w, h) in enumerate([(396, 264), (528, 352), (660, 440)]):
        g = rec[rec.W == w].sort_values("baseK")
        ax.plot(g.p95_typ, g.MAE_band, ls=fs.LINESTYLES[i], marker=fs.MARKERS[i], color=fs.SERIES[i],
                mfc=fs.SERIES[i], ms=4, zorder=3, label=f"{w}×{h}")
    sel = r.loc[(528, 25)]
    ax.plot(sel.p95_typ, sel.MAE_band, ls="none", marker="o", ms=9, mfc="none", mec=fs.INK, mew=0.9, zorder=4)
    ax.annotate("adopted\n528×352, K = 25", (sel.p95_typ, sel.MAE_band), xytext=(2, 41.5),
                textcoords="data", fontsize=6.5, color=fs.INK, ha="left", va="bottom",
                arrowprops={"arrowstyle": "-", "color": fs.INK2, "lw": 0.6, "shrinkB": 5})
    ax.set_xlabel("ZNCC time, typical scene, p95 [ms]")
    ax.set_ylabel("Held-out MAE [mm]")
    ax.set_xlim(0, 100)
    ax.set_ylim(40, 80)
    ax.legend(loc="upper right", handlelength=2.2, title="processing size", title_fontsize=6.5)
    fs.panel_label(ax, "(b)", "Resolution and window K")

    # (c) エッジ閾値 T × 推論帯
    ax = axes[2]
    ax.plot(front.valid_rate_mean, front.mae_mean_mm, color=fs.POINT_BG, lw=3.0, zorder=1,
            solid_capstyle="round", label="Pareto front")
    for i, t in enumerate([20, 30, 45, 60]):
        g = band[(band.edge_th == t) & (band.inf_low == 0.45)].sort_values("inf_high")
        ax.plot(g.valid_rate_mean, g.mae_mean_mm, ls=fs.LINESTYLES[i], marker=fs.MARKERS[i],
                color=fs.SERIES[i], mfc=fs.SERIES[i], ms=3.2, zorder=3, label=f"T = {t}")
    kx, ky = float(knee.valid_rate_mean.iloc[0]), float(knee.mae_mean_mm.iloc[0])
    ax.plot(kx, ky, ls="none", marker="o", ms=9, mfc="none", mec=fs.INK, mew=0.9, zorder=4)
    ax.annotate("adopted\nT = 20, band (0.45, 5) px", (kx, ky), xytext=(0.335, 41.0), textcoords="data",
                fontsize=6.5, color=fs.INK, ha="left", va="bottom",
                arrowprops={"arrowstyle": "-", "color": fs.INK2, "lw": 0.6, "shrinkB": 5})
    ax.set_xlabel("Valid-pixel ratio")
    ax.set_ylabel("Held-out MAE [mm]")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(40, 85)
    # T ごとの曲線は推論帯の下限 0.45 px で上限を 3〜14.5 px に振ったもの
    ax.legend(loc="upper left", handlelength=2.2, ncol=1)
    fs.panel_label(ax, "(c)", "Edge threshold T and inference band")

    for p in fs.save(fig, "fig6_parameter_selection"):
        print(f"written: {p}")


if __name__ == "__main__":
    main()
