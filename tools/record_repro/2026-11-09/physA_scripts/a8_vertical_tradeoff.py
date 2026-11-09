# -*- coding: utf-8 -*-
"""同じ光量のまま全方向の基線を得ると、得意な方向（縦エッジ）の基線をどれだけ手放すか。

入力: なし（a3_design_search.py の領域とベクトル化した指標を使う）
出力: physA_out/a8_vert_tradeoff.csv、a8_vert_tradeoff.png、標準出力
方法: 扇形・リング分割の全探索で、平均光量が閾値以上の設計について
      (1) 最も得意な方向の基線 = 3 つの重心の最大距離（三角形の最長辺。開口を回せば縦エッジに向けられる）
      (2) 最も苦手な方向の基線 = 三角形の最小幅
      のパレート前線（(1) を固定したとき (2) の最大）を、G 全開口の制約あり・なしで出す。現行は (1)=0.849・(2)=0。
仮定: 理想フィルタ・幾何光学・回折なし・アオリなし。開口は自由に回せる（回転で (1) を縦エッジに合わせる）。
"""
import csv
import itertools
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
import a3_design_search as a3

OUT = os.path.join(os.path.dirname(HERE), "physA_out")
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
DB = 0.005  # 最長辺のビン幅


def diam_minwidth(A, M):
    valid = A > 1e-9
    p = np.where(valid[..., None], M / np.where(valid, A, 1.0)[..., None], 0.0)
    s01 = np.hypot(*(p[:, 1] - p[:, 0]).T) * (valid[:, 0] & valid[:, 1])
    s02 = np.hypot(*(p[:, 2] - p[:, 0]).T) * (valid[:, 0] & valid[:, 2])
    s12 = np.hypot(*(p[:, 2] - p[:, 1]).T) * (valid[:, 1] & valid[:, 2])
    diam = np.maximum(np.maximum(s01, s02), s12)
    mw, _ = a3.metrics_vec(A, M)
    return diam, mw


def run(name, regions, alpha, thresholds, msuf):
    A_k, M_k = a3.region_moments(regions)
    K, F = len(regions), len(alpha)
    tv = np.array([al.FILTERS[f] for f in alpha])
    cA = A_k[:, None, None] * tv[None]
    cM = M_k[:, None, None, :] * tv[None, :, :, None]
    m = min(K, msuf)
    SA, SM = np.zeros((1, 3)), np.zeros((1, 3, 2))
    for k in range(K - m, K):
        SA = (SA[:, None] + cA[k][None]).reshape(-1, 3)
        SM = (SM[:, None] + cM[k][None]).reshape(-1, 3, 2)
    nb = int(2.2 / DB) + 1
    best = {(t, c): np.full(nb, -1.0) for t in thresholds for c in ("all", "gfull")}
    code = {(t, c): np.full((nb, 2), -1, np.int64) for t in thresholds for c in ("all", "gfull")}
    sidx = np.arange(SA.shape[0])
    for pre, digits in enumerate(itertools.product(range(F), repeat=K - m)):
        pA = sum((cA[k, d] for k, d in zip(range(K - m), digits)), np.zeros(3))
        pM = sum((cM[k, d] for k, d in zip(range(K - m), digits)), np.zeros((3, 2)))
        A = SA + pA
        M = SM + pM
        light = A.mean(1)
        diam, mw = diam_minwidth(A, M)
        b = np.clip((diam / DB).astype(np.int64), 0, nb - 1)
        gfull = A[:, 1] > 1 - 1e-9
        for t in thresholds:
            for c in ("all", "gfull"):
                ok = (light >= t - 1e-9) & (gfull if c == "gfull" else True)
                if not ok.any():
                    continue
                bb, vv, ii = b[ok], mw[ok], sidx[ok]
                order = np.lexsort((-vv, bb))
                bu, first = np.unique(bb[order], return_index=True)
                j = order[first]
                better = vv[j] > best[(t, c)][bu]
                best[(t, c)][bu[better]] = vv[j][better]
                code[(t, c)][bu[better], 0] = pre
                code[(t, c)][bu[better], 1] = ii[j][better]
    meta = (list(range(K - m)), list(range(K - m, K)), F)
    return best, code, meta


def front_from(best, code):
    # 最長辺が D 以上で取れる最小幅の最大（右から累積最大）
    ok = np.where(best >= 0)[0]
    D = (ok + 0.5) * DB
    V = best[ok]
    C = code[ok]
    out, run_max = [], -1
    for i in range(len(D) - 1, -1, -1):
        if V[i] > run_max + 1e-9:
            out.append((D[i], V[i], C[i]))
            run_max = V[i]
    return out[::-1]


def main():
    thresholds = [0.5, 0.6, 2.0 / 3.0, 0.75]
    cfgs = {"S8_all8": (a3.sectors(8), "WYCMRGBO", 6), "S12_WYCG": (a3.sectors(12), "WYCG", 8),
            "C050_S6_all8": ([(0, 0.5, -180, 180)] + a3.sectors(6, 0.5, 1.0), "WYCMRGBO", 6),
            "S6_all8": (a3.sectors(6), "WYCMRGBO", 6)}
    rows = []
    res = {}
    for name, (reg, alpha, msuf) in cfgs.items():
        best, code, meta = run(name, reg, alpha, thresholds, msuf)
        res[name] = (best, code, meta, reg, alpha)
        for (t, c), b in best.items():
            for D, V, C in front_from(b, code[(t, c)]):
                rows.append(dict(config=name, light_min=round(t, 4), constraint=c, diameter=D, minwidth=V,
                                 assignment=a3.decode(C, *meta, alpha)))
    with open(os.path.join(OUT, "a8_vert_tradeoff.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # 和集合の前線を表示: 光量 ≥ 2/3 と ≥ 0.6
    for t in thresholds:
        for c in ("all", "gfull"):
            pts = [(r["diameter"], r["minwidth"], r["config"], r["assignment"]) for r in rows
                   if abs(r["light_min"] - round(t, 4)) < 1e-6 and r["constraint"] == c]
            pts.sort(key=lambda x: -x[0])
            fr, bestv = [], -1
            for p in pts:
                if p[1] > bestv + 1e-9:
                    fr.append(p)
                    bestv = p[1]
            print(f"平均光量 ≥ {t:.3f}・{'制約なし' if c == 'all' else 'G 全開口'}: 最長辺（得意な方向の基線）→ 最小幅（苦手な方向の基線）")
            for D, V, cfg, a in sorted(fr):
                print(f"    {D:.3f} → {V:.3f}（{cfg}・{a}）")
    # 図
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    pal = {0.5: "#2a78d6", 0.6: "#1baf7a", round(2 / 3, 4): "#eb6834", 0.75: "#6250d6"}
    for ax, c in zip(axs, ("all", "gfull")):
        for t in thresholds:
            pts = [(r["diameter"], r["minwidth"]) for r in rows if abs(r["light_min"] - round(t, 4)) < 1e-6 and r["constraint"] == c]
            pts.sort(key=lambda x: -x[0])
            fr, bestv = [], -1
            for p in pts:
                if p[1] > bestv + 1e-9:
                    fr.append(p)
                    bestv = p[1]
            fr.sort()
            ax.step([x for x, _ in fr], [y for _, y in fr], where="post", color=pal[round(t, 4)], lw=2,
                    label=f"平均光量 ≥ {t:.2f}")
        ax.plot(0.849, 0.0, "s", ms=8, mfc="white", mec="#0b0b0b", mew=1.5)
        ax.annotate("現行 C/Y 左右（光量 0.67）", (0.849, 0.0), xytext=(-60, 12), textcoords="offset points", fontsize=8)
        ax.set_title("制約なし" if c == "all" else "G 全開口", fontsize=10)
        ax.set_xlabel("得意な方向の基線（三角形の最長辺・開口を回して縦エッジに向ける）")
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 1.3)
    axs[0].set_ylabel("最も苦手な方向の基線（三角形の最小幅）")
    axs[0].legend(fontsize=8)
    fig.suptitle("全方向の基線を得る代わりに得意な方向の基線をどれだけ手放すか（扇形・リング分割の全探索・理想フィルタ）", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "a8_vert_tradeoff.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
