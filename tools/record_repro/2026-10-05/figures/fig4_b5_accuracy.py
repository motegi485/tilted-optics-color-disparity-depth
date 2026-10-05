"""fig4_b5_accuracy.py — B5 の距離別精度（15 距離）の図。

入力（読み取りのみ）: project_knowledge/records/2026-10-01/distance_accuracy_b5_20261001.csv（推論帯あり＝本番出力）と
distance_accuracy_b5_20261001_bandoff.csv（推論帯なし＝gate1 のみ・保存 PNG から再計算）。
図: (a) 推定値の平均と真値（y = x）(b) MAE (c) 平均誤差（bias）。いずれも推論帯あり／なし。
誤差棒は付けない。公式 CSV の値だけを描く。
自己照合: 2 つの CSV の距離とセッションが一致し、群の要約が記録の表と一致すること。

実行（リポジトリ root）: uv run python tools/record_repro/2026-10-05/figures/fig4_b5_accuracy.py
出力: project_knowledge/records/2026-10-05/figures/ の fig4_b5_accuracy.pdf / .png
"""
from __future__ import annotations

import pandas as pd

import figstyle as fs

SRC = fs.RECORDS / "2026-10-01"
CALIB_RANGE = (1400, 3200)                 # 校正距離の範囲（10 距離 1400〜3200 mm）


def group_summary(df: pd.DataFrame, lo: int, hi: int) -> dict:
    g = df[(df.gt_mm >= lo) & (df.gt_mm <= hi)]
    return {"n": len(g), "mae": g.mae_mm.mean(), "absbias": g.mean_err_mm.abs().max(),
            "valid": g.valid_rate.mean()}


def main() -> None:
    on = pd.read_csv(SRC / "distance_accuracy_b5_20261001.csv").sort_values("gt_mm").reset_index(drop=True)
    off = pd.read_csv(SRC / "distance_accuracy_b5_20261001_bandoff.csv").sort_values("gt_mm").reset_index(drop=True)

    # --- 自己照合（記録の値と一致すること） ---
    fs.check(len(on) == 15 and len(off) == 15, "15 距離ずつ")
    fs.check(list(on.gt_mm) == list(off.gt_mm) and list(on.session) == list(off.session),
             "推論帯あり／なしで距離とセッションが一致")
    fs.check((off.band_on_selfcheck == "exact").all(), "推論帯なし CSV の自己照合列がすべて exact")
    s_on = group_summary(on, 1450, 3050)
    s_off = group_summary(off, 1450, 3050)
    fs.check(s_on["n"] == 9 and round(s_on["mae"], 1) == 48.2 and round(s_on["absbias"], 1) == 11.0
             and round(s_on["valid"], 3) == 0.311,
             f"校正範囲内 9 距離・推論帯あり: MAE 平均 {s_on['mae']:.1f}・|bias| 最大 {s_on['absbias']:.1f}・"
             f"有効率 {s_on['valid']:.3f}（記録票 §6.2: 48.2・11.0・0.311）")
    fs.check(round(s_off["mae"], 1) == 83.8 and round(s_off["absbias"], 1) == 53.9,
             f"校正範囲内 9 距離・推論帯なし: MAE 平均 {s_off['mae']:.1f}・|bias| 最大 {s_off['absbias']:.1f}"
             f"（記録票 §6.2: 83.8・53.9）")
    fs.check(round(group_summary(on, 1000, 1250)["mae"], 1) == 25.9
             and round(group_summary(on, 3450, 4000)["mae"], 1) == 183.7,
             "外挿の MAE 平均（近 25.9・遠 183.7・記録票 §6.2）")

    # --- 描画 ---
    fs.apply_style()
    fig, axes = fs.plt.subplots(3, 1, sharex=True, figsize=(fs.COLUMN_WIDTH_IN, 5.0),
                                gridspec_kw={"height_ratios": [1.25, 1, 1], "hspace": 0.32})
    series = [
        (on, "Inference band on (0.45 < |d| < 5 px)", 0),
        (off, "Inference band off (gate 1 only)", 1),
    ]
    for ax in axes:
        ax.axvspan(*CALIB_RANGE, color=fs.SHADE, lw=0, zorder=0)

    ax = axes[0]
    ax.plot([900, 4100], [900, 4100], color=fs.MUTED, lw=0.8, zorder=1)
    ax.text(3650, 3250, "y = x", ha="left", va="top", fontsize=7, color=fs.INK2)
    # 両系列がほぼ重なるので、白抜きの四角（なし）を大きめに下へ、塗りの丸（あり）を小さめに上へ描く
    for df, label, i in reversed(series):
        ax.plot(df.gt_mm, df.mean_est_mm, ls="none", marker=fs.MARKERS[i], color=fs.SERIES[i],
                mfc=fs.SERIES[i] if i == 0 else "white", mew=1.0, ms=3.2 if i == 0 else 5.5,
                zorder=4 if i == 0 else 3)
    ax.set_ylabel("Estimate [mm]")
    ax.set_xlim(900, 4100)
    ax.set_ylim(900, 4100)
    ax.text(sum(CALIB_RANGE) / 2, 4000, "calibrated range", ha="center", va="top",
            fontsize=7, color=fs.INK2)
    fs.panel_label(ax, "(a)", "Mean of per-pixel estimates")

    ax = axes[1]
    for df, label, i in series:
        ax.plot(df.gt_mm, df.mae_mm, ls=fs.LINESTYLES[i], marker=fs.MARKERS[i], color=fs.SERIES[i],
                mfc=fs.SERIES[i] if i == 0 else "white", mew=1.0, ms=4, zorder=3, label=label)
    ax.set_ylabel("MAE [mm]")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", handlelength=2.2)
    fs.panel_label(ax, "(b)", "Mean absolute error")

    ax = axes[2]
    ax.axhline(0, color=fs.MUTED, lw=0.8, zorder=1)
    for df, label, i in series:
        ax.plot(df.gt_mm, df.mean_err_mm, ls=fs.LINESTYLES[i], marker=fs.MARKERS[i], color=fs.SERIES[i],
                mfc=fs.SERIES[i] if i == 0 else "white", mew=1.0, ms=4, zorder=3)
    ax.set_ylabel("Bias [mm]")
    ax.set_xlabel("Ground-truth distance [mm]")
    ax.set_xticks(range(1000, 4001, 500))
    fs.panel_label(ax, "(c)", "Mean error (bias)")

    for p in fs.save(fig, "fig4_b5_accuracy"):
        print(f"written: {p}")


if __name__ == "__main__":
    main()
