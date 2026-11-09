# -*- coding: utf-8 -*-
"""現行 C/Y 左右からの「最小の改変」の族と、全探索の前線に出た代表形状を、同じ指標で評価する。

目的: 現行の形からどれだけ変えると、どれだけ全方向の基線が得られ、何を失うかを連続的に示す。
  族 A（G を遮る）: 現行の下側に中心角 w の M 扇形（R+B を通し G を遮る）。w=120° で CMY 三扇形（を回した形）と同じ幾何になる。
                    physB のメモの「C/Y＋下 M 扇形」は w=120°。平均光量は w によらず 2/3。
  族 B（R と B を遮る）: 現行の下側に中心角 w の G 扇形（G だけ通す）。G は全開口のまま、R・B の光量が減る。
  代表形状: a8_vertical_tradeoff.py の前線（平均光量 ≥ 0.5 / 2/3・G 全開口）に出た割り当て。
入力: なし（catalog.py・aperture_lib.py・a3_design_search.py・a4_spectral.py を import）
出力: physA_out/a10_families.csv、a10_reps.csv、a10_families.png、a10_reps.png、標準出力
仮定: 理想フィルタ（実フィルタの近似は「基準」ケースだけ並記・a4_spectral.py の仮定）・幾何光学・回折なし・ケラレなし・アオリなし。
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
import a3_design_search as a3
import a4_spectral as a4

OUT = os.path.join(os.path.dirname(HERE), "physA_out")
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
BASE_CASE = "基準(Y500/C580,幅7)"


def make_cy_g(width_deg):
    """現行の左右 Y/C の下側に中心角 width の G 扇形（R と B を遮る）を入れる"""
    def f(x, y):
        return al.paint(x, y, [(x < 0, "Y"), (x >= 0, "C"), (al.in_sector(x, y, 270, width_deg), "G")])
    return f


def row_of(name, w, e, e_real):
    p = np.nan_to_num(e["p"])
    return dict(family=name, width_deg=w, light_mean=e["light_mean"], a_R=e["a"][0], a_G=e["a"][1], a_B=e["a"][2],
                pR_x=p[0, 0], pR_y=p[0, 1], pG_x=p[1, 0], pG_y=p[1, 1], pB_x=p[2, 0], pB_y=p[2, 1],
                vert=e["vert"], horiz=e["horiz"], worst=e["worst"], fisher_min=e["fisher_min"],
                fisher_vert=e["fisher_vert"], zworst=e["zworst"], ratio_med=e["ratio_med"],
                sim2d_min=float(np.nanmin(e["sim2d"])), sim2d_RB=e["sim2d"][1],
                vert_real=e_real["vert"], worst_real=e_real["worst"], fisher_min_real=e_real["fisher_min"])


def main():
    grid = al.Grid(600)
    Wt = a4.eff_weights(a4.CASES[BASE_CASE])
    rows = []
    print("族 | w[deg] | 平均光量 | R,G,B 光量 | 縦 | 横 | 最悪 | Fisher最小 | 縦の Fisher | xcorr最悪 | 2D形状の最小 | 実フィルタ近似(基準): 縦/最悪")
    fams = {"A: 下に M（G を遮る）": (ct.make_cy_m, np.arange(0, 121, 10)),
            "B: 下に G（R・B を遮る）": (make_cy_g, np.arange(0, 181, 15))}
    for name, (mk, ws) in fams.items():
        for w in ws:
            f = mk(float(w)) if w > 0 else ct.cy_lr
            e = al.evaluate(f, grid, lsf_step_deg=6.0)
            er = al.evaluate(a4.realize(f, Wt), grid, lsf_step_deg=6.0)
            r = row_of(name, int(w), e, er)
            rows.append(r)
            print(f"{name} | {w:3d} | {r['light_mean']:.3f} | {r['a_R']:.2f},{r['a_G']:.2f},{r['a_B']:.2f} | {r['vert']:.3f} | "
                  f"{r['horiz']:.3f} | {r['worst']:.3f} | {r['fisher_min']:.4f} | {r['fisher_vert']:.4f} | {r['zworst']:.3f} | "
                  f"{r['sim2d_min']:.3f} | {r['vert_real']:.3f}/{r['worst_real']:.3f}")
    with open(os.path.join(OUT, "a10_families.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    # 図: 族の曲線
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, name, col in zip(axs, fams, ("#e87ba4", "#1baf7a")):
        rr = [r for r in rows if r["family"] == name]
        x = [r["width_deg"] for r in rr]
        ax.plot(x, [r["vert"] for r in rr], "-o", color="#2a78d6", lw=2, ms=4, label="縦エッジの基線")
        ax.plot(x, [r["worst"] for r in rr], "-o", color="#eb6834", lw=2, ms=4, label="最も苦手な方向の基線")
        ax.plot(x, [r["light_mean"] for r in rr], "--", color="#0b0b0b", lw=1.6, label="平均光量")
        ax.plot(x, [r["a_G"] for r in rr], ":", color="#008300", lw=2, label="G の光量")
        ax.plot(x, [r["sim2d_min"] for r in rr], "-.", color="#52514e", lw=1.4, label="ぼけ形状の一致（2D・最小の対）")
        ax.plot(x, [r["worst_real"] for r in rr], "-", color="#eb6834", lw=1, alpha=0.6,
                label="最も苦手な方向の基線（実フィルタ近似・色変換なし）")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("下側の扇形の中心角 w [deg]（0 = 現行 C/Y 左右）")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=7.5, loc="center right")
    fig.suptitle("現行からの最小の改変（理想フィルタ・幾何光学・アオリなし。族 A の w=120° は CMY 三扇形と同じ幾何）", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "a10_families.png"), dpi=120)
    plt.close(fig)
    # 代表形状（a8 の前線・G 全開口）
    reps = [("S8", "YYCCGGGG", "光量≥0.5・G 全開口で縦 0.85 を保つ"),
            ("S8", "YWWCCGGY", "光量≥2/3・G 全開口・縦 0.60"),
            ("S12", "YYWWWWCCGGGG", "光量≥2/3・G 全開口・最悪の最大"),
            ("C050_S6", "GYGGGCW", "光量≥0.5・G 全開口・最悪の最大")]
    regs = {"S8": a3.sectors(8), "S12": a3.sectors(12),
            "C050_S6": [(0, 0.5, -180, 180)] + a3.sectors(6, 0.5, 1.0)}
    out, evals = [], []
    print("\n代表形状 | 平均光量 | R,G,B | 縦/横/最悪（最長辺の向きを縦エッジに回した値） | Fisher最小 | xcorr最悪 | 2D形状の最小 | 実フィルタ近似(基準) 最悪")
    for cfg, asg, note in reps:
        f0 = a3.design_from_assignment(regs[cfg], asg)
        e0 = al.evaluate(f0, grid, lsf_step_deg=6.0)
        p = np.nan_to_num(e0["p"])
        # 三角形の最長辺を水平に（縦エッジの基線を最大に）回す
        pairs = [(0, 1), (0, 2), (1, 2)]
        d = max(pairs, key=lambda ij: np.hypot(*(p[ij[1]] - p[ij[0]])))
        v = p[d[1]] - p[d[0]]
        ang = np.degrees(np.arctan2(v[1], v[0]))
        f = al.rotate_design(f0, -ang)
        e = al.evaluate(f, grid, lsf_step_deg=6.0)
        er = al.evaluate(a4.realize(f, Wt), grid, lsf_step_deg=6.0)
        r = dict(config=cfg, assignment=asg, note=note, rot_deg=-ang, light_mean=e["light_mean"], a_R=e["a"][0],
                 a_G=e["a"][1], a_B=e["a"][2], vert=e["vert"], horiz=e["horiz"], worst=e["worst"],
                 fisher_min=e["fisher_min"], fisher_vert=e["fisher_vert"], zworst=e["zworst"], ratio_med=e["ratio_med"],
                 sim2d_min=float(np.nanmin(e["sim2d"])), worst_real=er["worst"], vert_real=er["vert"])
        out.append(r)
        evals.append((f, e, r))
        print(f"{cfg}・{asg}（{note}） | {r['light_mean']:.3f} | {r['a_R']:.2f},{r['a_G']:.2f},{r['a_B']:.2f} | "
              f"{r['vert']:.3f}/{r['horiz']:.3f}/{r['worst']:.3f} | {r['fisher_min']:.4f} | {r['zworst']:.3f} | {r['sim2d_min']:.3f} | {r['worst_real']:.3f}")
    with open(os.path.join(OUT, "a10_reps.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)
    # 図: 族の形と代表形状
    shows = [("現行 C/Y 左右", ct.cy_lr), ("族 A: 下に M 60°", ct.make_cy_m(60)), ("族 A: 下に M 120°（=CMY 三扇形）", ct.make_cy_m(120)),
             ("族 B: 下に G 90°", make_cy_g(90)), ("族 B: 下に G 180°", make_cy_g(180))]
    shows += [(f"{r['config']}・{r['assignment']}", f) for f, e, r in evals]
    fig, axs = plt.subplots(2, 5, figsize=(14, 6.2))
    for ax, (lab, f) in zip(axs.ravel(), shows):
        e = al.evaluate(f, al.Grid(300), lsf_step_deg=30.0)
        ax.imshow(al.render_rgb(f), extent=[-1, 1, -1, 1], interpolation="nearest")
        p = e["p"]
        for c, colr in zip(range(3), ["#ff3030", "#30c030", "#3060ff"]):
            if e["valid"][c]:
                ax.plot(p[c, 0], p[c, 1], "o", ms=7, mfc=colr, mec="black", mew=1.2)
        ax.set_title(f"{lab}\n光量 {e['light_mean']:.2f}・G {e['a'][1]:.2f}・縦 {e['vert']:.2f}・最悪 {e['worst']:.2f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axs.ravel()[len(shows):]:
        ax.axis("off")
    fig.suptitle("最小の改変の族と、全探索の前線に出た G 全開口の形（点は R・G・B の重心。開口半径 1）", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "a10_reps.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
