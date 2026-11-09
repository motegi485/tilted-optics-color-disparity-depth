# -*- coding: utf-8 -*-
"""光学の原理として言えることを数値で確かめる（理想フィルタ・幾何光学・アオリなし）。

入力: なし
出力: 標準出力、physA_out/a6_bounds_table.csv、a6_tradeoff.png
確かめること:
  1. 面積 a の領域の重心の最大距離 g(a)（弦で切った切片）。a が増えると単調に減る＝光量と基線は同時に最大にできない。
  2. 上限の閉じた形: 3 チャネル等光量なら最も苦手な方向の基線 ≤ 1.5 g(a)（正三角形）、
     G 全開口（p_G=0）で R・B が光量 a なら ≤ (√3/2) g(a)（3 点が一辺 g(a) の正三角形）。
  3. 形状一致: G 全開口なら R と G の透過マスクの正規化相関は √a_R 以下（重なり ≤ a_R、ノルム 1 と √a_R）。
     平行移動で同じ形の円 3 つ（半径 r・縁に接する）なら形状は完全一致だが、光量 r²・重心距離 1−r。
  4. 共線の定理: G を通すフィルタ（W/Y/C/G）だけで開口を埋めると p_G=0。さらに Y と C だけで埋めると
     a_R p_R = −a_B p_B なので R・G・B の重心は必ず一直線（全方向の基線は 0）。乱数の分割で確かめる。
  5. RGB の 3 チャネルからは、画素ごとの線形な分解で最大 3 つの部分開口の像しか取り出せない（階数 ≤ 3）。
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

OUT = os.path.join(os.path.dirname(HERE), "physA_out")
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    A = np.round(np.arange(0.05, 1.0, 0.05), 3)
    g = al.segment_centroid_dist(A)
    rows = []
    print("a | g(a) | 等光量 3 切片の最小幅 1.5g | G 全開口の上限 (√3/2)g（平均光量 (1+2a)/3） | √a（G 全開口での R–G 形状一致の上限） | 平行移動の円 3 つ: 最小幅 1.5(1−√a)")
    for a, gg in zip(A, g):
        r = dict(a=a, g=gg, bound_equal3=1.5 * gg, bound_gfull=np.sqrt(3) / 2 * gg, light_gfull=(1 + 2 * a) / 3,
                 sim_rg_max_gfull=np.sqrt(a), disks3_minwidth=1.5 * (1 - np.sqrt(a)))
        rows.append(r)
        print(f"{a:.2f} | {gg:.4f} | {r['bound_equal3']:.4f} | {r['bound_gfull']:.4f}（{r['light_gfull']:.3f}） | {r['sim_rg_max_gfull']:.3f} | {r['disks3_minwidth']:.4f}")
    with open(os.path.join(OUT, "a6_bounds_table.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # 円 3 つの最小幅が切片の上限に対してどれだけ低いか
    for a in (0.1, 0.2, 0.25, 0.3333):
        print(f"光量 {a:.3f}: 平行移動の円 3 つ {1.5 * (1 - np.sqrt(a)):.3f} / 切片 3 つ {1.5 * al.segment_centroid_dist(a):.3f}")
    # G 全開口で形状一致 √a ≥ s を求めたときの基線の上限
    print("G 全開口で R–G の形状一致を s 以上にしたいときの、最も苦手な方向の基線の上限 (√3/2)·g(s²)")
    for s in (0.7, 0.8, 0.9, 0.95):
        print(f"  s={s:.2f}: a≥{s * s:.3f} → 上限 {np.sqrt(3) / 2 * al.segment_centroid_dist(s * s):.4f}")
    # 共線の定理を乱数で確かめる（扇形 24 分割にランダムに割り当て）
    rng = np.random.default_rng(1)
    import a3_design_search as a3
    regions = a3.sectors(24)
    Ak, Mk = a3.region_moments(regions)
    for alpha in ("YC", "YCW", "YCG", "WYCG"):
        tv = np.array([al.FILTERS[f] for f in alpha])
        n = 20000
        idx = rng.integers(0, len(alpha), size=(n, len(regions)))
        T = tv[idx]  # (n,K,3)
        Asum = np.einsum("nkc,k->nc", T, Ak)
        Msum = np.einsum("nkc,kd->ncd", T, Mk)
        mw, fi = a3.metrics_vec(Asum, Msum)
        print(f"  フィルタ {alpha:4s}（扇形 24 分割・乱数 {n} 通り）: G の光量の最小 {Asum[:, 1].min():.4f}・最小幅の最大 {mw.max():.2e}・Fisher 最小固有値の最大 {fi.max():.2e}")
    # 図: 三つの量の関係
    aa = np.linspace(0.01, 0.99, 200)
    gg = al.segment_centroid_dist(aa)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axs[0]
    ax.plot(aa, 1.5 * gg, color="#2a78d6", lw=2, label="切片 3 つ（等光量・形状は不一致）: 1.5·g(a)")
    ax.plot(aa, 1.5 * (1 - np.sqrt(aa)), color="#eb6834", lw=2, label="平行移動の円 3 つ（形状が一致）: 1.5·(1−√a)")
    ax.plot((1 + 2 * aa) / 3, np.sqrt(3) / 2 * gg, color="#1baf7a", lw=2, ls="--",
            label="G 全開口（横軸は平均光量 (1+2a)/3）: (√3/2)·g(a)")
    ax.set_xlabel("平均光量（G 全開口の線は (1+2a)/3、他は各チャネルの光量 a）")
    ax.set_ylabel("最も苦手な方向の基線の上限（開口半径=1）")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax = axs[1]
    ax.plot(np.sqrt(aa), np.sqrt(3) / 2 * gg, color="#1baf7a", lw=2)
    for s in (0.7, 0.8, 0.9):
        y = np.sqrt(3) / 2 * al.segment_centroid_dist(s * s)
        ax.plot(s, y, "o", color="#1baf7a", ms=7)
        ax.annotate(f"一致 {s:.1f} → 基線 ≤ {y:.2f}", (s, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("G 全開口での R–G の形状一致の上限 √a_R（重心をそろえたマスクの正規化相関）")
    ax.set_ylabel("最も苦手な方向の基線の上限")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    fig.suptitle("光量・全方向の基線・ぼけ形状の一致は同時に最大化できない（理想フィルタ・幾何光学の上限）", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "a6_tradeoff.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
