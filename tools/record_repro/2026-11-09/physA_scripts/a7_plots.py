# -*- coding: utf-8 -*-
"""メモ用のまとめの図と表を、a2〜a6 の CSV から作る（計算はしない）。

入力: physA_out/a2_catalog.csv, a3_front_*.csv, a3_bound.csv, a3_reps.csv, a4_fronts_real.csv, a6_bounds_table.csv
出力: physA_out/a7_front_summary.png（全構成の前線の和集合・理論上限・カタログ形状）、a7_front_union.csv、
      a7_real_fronts.png（実フィルタの近似での前線）、標準出力（代表形状の表）
"""
import csv
import glob
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

OUT = os.path.join(os.path.dirname(HERE), "physA_out")
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
INK2 = "#52514e"


def read(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pareto(points):
    pts = sorted(points, key=lambda r: -r[0])
    out, best = [], -1
    for p in pts:
        if p[1] > best + 1e-9:
            out.append(p)
            best = p[1]
    return sorted(out, key=lambda r: r[0])


def main():
    cat = read(os.path.join(OUT, "a2_catalog.csv"))
    bound = read(os.path.join(OUT, "a3_bound.csv"))
    fronts = {}
    for p in glob.glob(os.path.join(OUT, "a3_front_*.csv")):
        cfg = os.path.basename(p)[len("a3_front_"):-4]
        for r in read(p):
            fronts.setdefault((r["metric"], r["constraint"]), []).append(
                (float(r["light_mean"]), float(r["value"]), cfg, r["assignment"]))
    union = {k: pareto(v) for k, v in fronts.items()}
    with open(os.path.join(OUT, "a7_front_union.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "constraint", "light_mean", "value", "config", "assignment"])
        for (m, c), lst in union.items():
            for l, v, cfg, a in lst:
                w.writerow([m, c, f"{l:.4f}", f"{v:.4f}", cfg, a])
    print("全構成の前線の和集合（最も苦手な方向の基線）: 平均光量 → 値（構成・割り当て）")
    for c in ("all", "gfull"):
        print(" ", "制約なし" if c == "all" else "G 全開口")
        for l, v, cfg, a in union[("mw", c)]:
            if l >= 0.3:
                print(f"    {l:.3f} → {v:.3f}（{cfg}・{a}）")
    aa = np.linspace(0.01, 0.99, 300)
    gg = al.segment_centroid_dist(aa)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.3))
    for ax, met, yl in zip(axs, ("mw", "fi"), ("最も苦手な方向の基線（開口半径=1）", "Fisher 情報の最小固有値（光量で重み付け）")):
        for con, ls, lab in (("all", "-", "理論上限・制約なし"), ("gfull", "--", "理論上限・G 全開口")):
            b = sorted([(float(r["light_mean"]), float(r["value"])) for r in bound if r["metric"] == met and r["constraint"] == con])
            ax.plot([x for x, _ in b], [y for _, y in b], color="#0b0b0b", ls=ls, lw=2.2, label=lab)
        for con, col, ls, lab in (("all", "#2a78d6", "-", "扇形・リング分割の全探索（制約なし）"),
                                  ("gfull", "#eb6834", "--", "扇形・リング分割の全探索（G 全開口）")):
            u = union[(met, con)]
            ax.step([x for x, *_ in u], [y for _, y, *_ in u], where="pre", color=col, ls=ls, lw=1.8, label=lab)
        if met == "mw":
            ax.plot(aa, 1.5 * (1 - np.sqrt(aa)), color="#1baf7a", ls=":", lw=2, label="形が一致する平行移動の円 3 つ（この族の値・一般の上限ではない）")
        for r in cat:
            x = float(r["light_mean"])
            y = float(r["worst"] if met == "mw" else r["fisher_min"])
            gf = r["G_full"] == "True"
            ax.plot(x, y, "s" if gf else "o", ms=6.5, mfc="white", mec=INK2, mew=1.4)
            if r["key"] in ("cy_lr", "rgb3", "cmy3", "ycg3", "quad", "wheel6", "ywcg4_v", "par_hex", "bando", "mca",
                            "cy_m90", "kaneko_g", "par_cy_open"):
                ax.annotate(r["label"], (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points", color=INK2)
        ax.set_xlabel("平均光量（3 チャネルの平均・開口全体=1）")
        ax.set_ylabel(yl)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("平均光量と全方向の基線のトレードオフ（理想フィルタ・幾何光学・アオリなし。□=G 全開口の形、○=それ以外）", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "a7_front_summary.png"), dpi=120)
    plt.close(fig)
    # 実フィルタ
    p = os.path.join(OUT, "a4_fronts_real.csv")
    if os.path.exists(p):
        rr = read(p)
        cases = []
        for r in rr:
            if r["case"] not in cases:
                cases.append(r["case"])
        pal = ["#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
        fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
        for ax, fam in zip(axs, ("S8 全 8 種", "S12 W/Y/C/G")):
            for cn, col in zip(cases, pal):
                pts = sorted([(float(r["light"]), float(r["value"])) for r in rr
                              if r["case"] == cn and r["family"] == fam and r["metric"] == "mw"])
                ax.step([x for x, _ in pts], [y for _, y in pts], where="pre", color=col, lw=1.8, label=cn)
            ax.set_title(f"{fam}（{'制約なし' if fam.startswith('S8') else 'G を通すフィルタだけ'}）", fontsize=9.5)
            ax.set_xlabel("平均光量")
            ax.set_ylabel("最も苦手な方向の基線（色変換なし）")
            ax.set_xlim(0, 1)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
        axs[0].legend(fontsize=7.5)
        fig.suptitle("実フィルタの近似（透過率・センサ感度はすべて仮定）では全方向の基線がおよそ半分に縮む（3×3 の色変換で灰色の物体なら戻る・a4b）", fontsize=9.5)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "a7_real_fronts.png"), dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
