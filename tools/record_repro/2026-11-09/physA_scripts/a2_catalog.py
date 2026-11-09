# -*- coding: utf-8 -*-
"""既知と候補の開口形状のカタログを、同じ指標で評価する（理想フィルタ・幾何光学・アオリなし）。

入力: catalog.py の DESIGNS
出力: physA_out/a2_catalog.csv（指標の表）、a2_catalog_layouts.png（形と重心）、a2_catalog_profiles.png（方向ごとの基線）
指標:
  (a) 方向ごとの『3 チャネルの重心の差の法線成分の最大値』と、その最小（最も苦手な方向）
  (b) Fisher 情報 I(n)=Σa_i((p_i−p̄)·n)² の最小固有値（光子雑音律速・共通のずれは nuisance）。
      幅補正版 Iw: 重みを a_i/w_i（w_i はその方向の LSF の等価幅）にしたもの（ぼけた段差の位置推定の CRLB ∝ w/a を仮定）
  (c) 平均光量・G の光量・G が全開口か
  (d) ぼけ形状の一致: 重心をそろえた透過マスクの正規化相関（2D）、各方向の LSF 相互相関のピークでの正規化相関、
      相互相関ピークの位置と重心差の比（1 なら重心どおりに視差が出る）
  (e) 前後判別: 最も苦手な方向の基線 > 0 なら全方向で符号が取れる（重心差の符号が合焦面の前後で反転する）
仮定: aperture_lib.py の冒頭を参照。寸法の近似は catalog.py の各設計の docstring。
"""
import csv
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aperture_lib as al
import catalog as ct

OUT = os.path.join(os.path.dirname(HERE), "physA_out")
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    grid = al.Grid(1200)
    res = []
    for key, f, label, src in ct.DESIGNS:
        r = al.evaluate(f, grid, name=key)
        r["label"] = label
        r["src"] = src
        res.append(r)
        print(f"{key:13s} 光量 {r['light_mean']:.3f} (R {r['a'][0]:.3f} G {r['a'][1]:.3f} B {r['a'][2]:.3f}) "
              f"縦 {r['vert']:.3f} 横 {r['horiz']:.3f} 最悪 {r['worst']:.3f}@{r['worst_theta']:.0f}° "
              f"Fisher min {r['fisher_min']:.4f} max {r['fisher_max']:.4f} Iw_min {r['Iw_min']:.4f} | "
              f"xcorr 縦 {r['zvert']:.3f} 横 {r['zhoriz']:.3f} 最悪 {r['zworst']:.3f} 比の中央値 {r['ratio_med']:.3f} | "
              f"LSF相関 最小 {r['ncc_best_min']:.3f} 中央 {r['ncc_best_med']:.3f} | 2D形状 RG {r['sim2d'][0]:.3f} RB {r['sim2d'][1]:.3f} GB {r['sim2d'][2]:.3f}")
    # CSV
    fields = ["key", "label", "src", "light_mean", "a_R", "a_G", "a_B", "G_full", "pR_x", "pR_y", "pG_x", "pG_y",
              "pB_x", "pB_y", "vert", "horiz", "worst", "worst_theta", "worst_tri", "fisher_min", "fisher_max",
              "fisher_vert", "fisher_horiz", "Iw_min", "zvert", "zhoriz", "zworst", "ratio_med",
              "ncc_best_min", "ncc_best_med", "sim2d_RG", "sim2d_RB", "sim2d_GB", "sign_all_dirs"]
    with open(os.path.join(OUT, "a2_catalog.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in res:
            p = np.nan_to_num(r["p"])
            w.writerow([r["name"], r["label"], r["src"], r["light_mean"], *r["a"], r["G_full"], *p[0], *p[1], *p[2],
                        r["vert"], r["horiz"], r["worst"], r["worst_theta"], r["worst_tri"], r["fisher_min"],
                        r["fisher_max"], r["fisher_vert"], r["fisher_horiz"], r["Iw_min"], r["zvert"], r["zhoriz"],
                        r["zworst"], r["ratio_med"], r["ncc_best_min"], r["ncc_best_med"], *r["sim2d"],
                        r["worst"] > 1e-6])
    # 形の図
    n = len(res)
    cols = 6
    rows_ = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows_, cols, figsize=(cols * 2.6, rows_ * 3.0))
    for ax, r, (key, f, label, src) in zip(axs.ravel(), res, ct.DESIGNS):
        img = al.render_rgb(f)
        ax.imshow(img, extent=[-1, 1, -1, 1], interpolation="nearest")
        p = r["p"]
        for c, col in zip(range(3), ["#ff3030", "#30c030", "#3060ff"]):
            if r["valid"][c]:
                ax.plot(p[c, 0], p[c, 1], "o", ms=7, mfc=col, mec="black", mew=1.2)
        vv = [c for c in range(3) if r["valid"][c]]
        if len(vv) == 3:
            tri = np.vstack([p[vv], p[vv[0]]])
            ax.plot(tri[:, 0], tri[:, 1], "-", color="black", lw=1.0)
        ax.set_title(f"{label}\n光量 {r['light_mean']:.2f}・最悪 {r['worst']:.2f}", fontsize=8.5)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("開口形状と RGB の重心（赤・緑・青の点。開口半径 1・理想フィルタ・アオリなし）", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "a2_catalog_layouts.png"), dpi=110)
    plt.close(fig)
    # 方向ごとの基線（代表）
    sel = ["cy_lr", "cy_lr_45", "rgb3", "cmy3", "ycg3", "quad", "wheel6", "ywcg4_v", "par_hex", "cy_m60"]
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#52514e", "#0b0b0b"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for k, col in zip(sel, palette):
        r = next(x for x in res if x["name"] == k)
        axs[0].plot(r["thetas"], r["prof"], color=col, lw=2, label=r["label"])
        axs[1].plot(r["thetas"], r["Iprof"], color=col, lw=2, label=r["label"])
    for ax, yl in zip(axs, ["基線（重心差の法線成分の最大・開口半径=1）", "Fisher 情報 I(n)（光量で重み付け）"]):
        ax.set_xlabel("エッジの法線の向き [deg]（0=縦エッジ・90=横エッジ）")
        ax.set_ylabel(yl)
        ax.set_xlim(0, 180)
        ax.set_xticks(range(0, 181, 30))
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)
    axs[1].legend(fontsize=7.5, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "a2_catalog_profiles.png"), dpi=110)
    plt.close(fig)
    # 重心差に対する相互相関ピークの比（方向ごと・現行と 2 つの候補）
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, col in zip(["cy_lr", "wheel6", "ywcg4_v", "rgb3", "mca"], palette):
        r = next(x for x in res if x["name"] == k)
        ax.plot(r["lsf_thetas"], r["zprof"], color=col, lw=2, label=f"{r['label']}（相互相関ピーク）")
        ax.plot(r["thetas"], r["prof"], color=col, lw=1, ls="--")
    ax.set_xlabel("エッジの法線の向き [deg]")
    ax.set_ylabel("基線（実線: LSF 相互相関のピーク／破線: 重心差）")
    ax.set_xlim(0, 180)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "a2_xcorr_vs_centroid.png"), dpi=110)
    plt.close(fig)
    np.save(os.path.join(OUT, "a2_profiles.npy"), {r["name"]: dict(thetas=r["thetas"], prof=r["prof"], Iprof=r["Iprof"],
                                                                   lsf_thetas=r["lsf_thetas"], zprof=r["zprof"],
                                                                   Iw=r["Iw"], label=r["label"]) for r in res},
            allow_pickle=True)


if __name__ == "__main__":
    main()
