# -*- coding: utf-8 -*-
"""実フィルタの近似: フィルタの透過率とセンサ感度を仮定して、各チャネルの実効重心が理想からどれだけ中心へ縮むか、
前線（平均光量 対 最も苦手な方向の基線）の結論が変わるかを感度として示す。すべて [仮定]。

入力: なし
出力: physA_out/a4_weights.csv（フィルタ×チャネルの実効透過率）、a4_catalog_real.csv（カタログ形状の指標・ケース別）、
      a4_fronts_real.csv / a4_fronts_real.png（S8 の全探索の前線をケース別に）
モデル:
  - センサ感度は開発リポ tools/record_repro/2026-11-07/theme1_physics/spectral.py の SENS をそのまま写した [仮定]
    （R: 600 nm・半値幅 75、G: 535・85、B: 462・65 のガウス × IR カット 680 nm）。光源・物体は平坦な白（灰色）。
  - フィルタ（遷移幅 w のロジスティック・漏れ ε）[仮定]: Y=長波長通過（端 y_e）、C=短波長通過（端 c_e）、
    M=y_e より短い側と c_e より長い側を通す（G 帯を遮る）、R=c_e より長い側、G=y_e〜c_e、B=y_e より短い側、W=1、O=0。
  - 実効透過率 w[f,c] = ∫S_c T_f dλ / ∫S_c dλ。各チャネルの面積・モーメントは領域ごとに w[f,c] で重み付けした和。
    ホワイトバランスのゲインは重心と Fisher の比較を変えない（チャネルごとの定数倍なので）。
  - 濃度勾配の設計（Paramonov の 2 つ）は点ごとのフィルタ種類が決まらないので除く。
仮定（幾何）: aperture_lib.py の冒頭（回折なし・ケラレなし・アオリなし）。
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

OUT = os.path.join(os.path.dirname(HERE), "physA_out")
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

LAM = np.arange(380.0, 731.0, 1.0)


def gauss(mu, fwhm):
    s = fwhm / 2.3548
    return np.exp(-0.5 * ((LAM - mu) / s) ** 2)


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


IRCUT = expit((680.0 - LAM) / 8.0)
SENS = dict(R=gauss(600.0, 75.0) * IRCUT, G=gauss(535.0, 85.0) * IRCUT, B=gauss(462.0, 65.0) * IRCUT)  # spectral.py と同じ

CASES = {
    "理想": None,
    "基準(Y500/C580,幅7)": dict(y_e=500.0, c_e=580.0, w=7.0, leak=0.0),
    "重なり狭(Y520/C560)": dict(y_e=520.0, c_e=560.0, w=7.0, leak=0.0),
    "重なり広(Y480/C610)": dict(y_e=480.0, c_e=610.0, w=7.0, leak=0.0),
    "遷移幅20nm": dict(y_e=500.0, c_e=580.0, w=20.0, leak=0.0),
    "漏れ3%": dict(y_e=500.0, c_e=580.0, w=7.0, leak=0.03),
    "漏れ10%": dict(y_e=500.0, c_e=580.0, w=7.0, leak=0.10),
}
LETTERS = "WYCMRGBO"


def filter_curves(y_e, c_e, w, leak):
    lp = lambda e: expit((LAM - e) / w)
    sp = lambda e: expit((e - LAM) / w)
    base = dict(W=np.ones_like(LAM), Y=lp(y_e), C=sp(c_e), M=np.clip(sp(y_e) + lp(c_e), 0, 1),
                R=lp(c_e), G=lp(y_e) * sp(c_e), B=sp(y_e), O=np.zeros_like(LAM))
    return {k: (v if k in "WO" else leak + (1 - leak) * v) for k, v in base.items()}


def eff_weights(case):
    if case is None:
        return np.array([al.FILTERS[f] for f in LETTERS])
    T = filter_curves(**case)
    Wt = np.zeros((len(LETTERS), 3))
    for i, f in enumerate(LETTERS):
        for j, c in enumerate("RGB"):
            Wt[i, j] = (SENS[c] * T[f]).sum() / SENS[c].sum()
    return Wt


CODE2IDX = {(1, 1, 1): 0, (1, 1, 0): 1, (0, 1, 1): 2, (1, 0, 1): 3, (1, 0, 0): 4, (0, 1, 0): 5, (0, 0, 1): 6, (0, 0, 0): 7}


def realize(func, Wt):
    """理想の設計関数 → 実フィルタの近似の設計関数（点ごとのフィルタ種類を理想の透過ベクトルから復元）"""
    lut = np.zeros(8, dtype=np.int64)
    for code, idx in CODE2IDX.items():
        lut[code[0] * 4 + code[1] * 2 + code[2]] = idx

    def f(x, y):
        T = func(x, y)
        inside = al.in_disk(x, y)
        b = np.rint(T).astype(np.int64)
        idx = lut[b[:, 0] * 4 + b[:, 1] * 2 + b[:, 2]]
        out = Wt[idx]
        out[~inside] = 0.0
        return out
    return f


def main():
    # 実効透過率
    with open(os.path.join(OUT, "a4_weights.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "filter", "R", "G", "B"])
        for cn, case in CASES.items():
            Wt = eff_weights(case)
            for i, f in enumerate(LETTERS):
                w.writerow([cn, f, *[f"{v:.4f}" for v in Wt[i]]])
            print(cn, " ".join(f"{f}:({Wt[i,0]:.2f},{Wt[i,1]:.2f},{Wt[i,2]:.2f})" for i, f in enumerate(LETTERS)))
    # カタログ
    skip = {"par_cmy_grad", "par_cy_grad"}
    grid = al.Grid(700)
    rows = []
    ideal = {}
    print("\n形状 | ケース | 光量 | R,G,B の重心距離（理想比） | 縦 | 横 | 最悪 | Fisher最小")
    for key, func, label, src in ct.DESIGNS:
        if key in skip:
            continue
        for cn, case in CASES.items():
            Wt = eff_weights(case)
            e = al.evaluate(realize(func, Wt), grid, lsf_step_deg=6.0, name=key)
            pn = np.hypot(*np.nan_to_num(e["p"]).T)
            if cn == "理想":
                ideal[key] = (pn, e)
            pi = ideal[key][0]
            shrink = np.where(pi > 1e-3, pn / np.where(pi > 1e-3, pi, 1), np.nan)
            rows.append(dict(key=key, label=label, case=cn, light=e["light_mean"], a_R=e["a"][0], a_G=e["a"][1],
                             a_B=e["a"][2], r_R=pn[0], r_G=pn[1], r_B=pn[2], shrink_R=shrink[0], shrink_G=shrink[1],
                             shrink_B=shrink[2], vert=e["vert"], horiz=e["horiz"], worst=e["worst"],
                             worst_ratio=e["worst"] / ideal[key][1]["worst"] if ideal[key][1]["worst"] > 1e-6 else np.nan,
                             vert_ratio=e["vert"] / ideal[key][1]["vert"] if ideal[key][1]["vert"] > 1e-6 else np.nan,
                             fisher_min=e["fisher_min"]))
            if key in ("cy_lr", "rgb3", "cmy3", "ycg3", "quad", "wheel6", "ywcg4_v", "cy_m90", "par_hex", "bando", "kaneko_g"):
                print(f"{label} | {cn} | {e['light_mean']:.3f} | {pn[0]:.3f},{pn[1]:.3f},{pn[2]:.3f} "
                      f"({shrink[0]:.2f},{shrink[1]:.2f},{shrink[2]:.2f}) | {e['vert']:.3f} | {e['horiz']:.3f} | {e['worst']:.3f} | {e['fisher_min']:.4f}")
    with open(os.path.join(OUT, "a4_catalog_real.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # 順位の変化（最悪の基線・ケース別）
    print("\n最も苦手な方向の基線の順位（上位 10）")
    for cn in CASES:
        rr = sorted([r for r in rows if r["case"] == cn], key=lambda r: -r["worst"])[:10]
        print(f"  {cn}: " + " > ".join(f"{r['label']}({r['worst']:.2f})" for r in rr))
    # 前線（S8・8 種）と G 系（S12・WYCG）
    regions8 = a3.sectors(8)
    regions12 = a3.sectors(12)
    fr_rows = []
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    pal = ["#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    for (cn, case), col in zip(CASES.items(), pal):
        Wt = eff_weights(case)
        best8, meta8 = a3.search(f"S8_{cn}", regions8, LETTERS, 6, tv=Wt)
        idxG = [LETTERS.index(c) for c in "WYCG"]
        best12, meta12 = a3.search(f"S12WYCG_{cn}", regions12, "WYCG", 8, tv=Wt[idxG], gfull_th=-1.0)
        for ax, met in zip(axs, ("mw", "fi")):
            for (b, meta, alpha, lab, ls) in ((best8, meta8, LETTERS, "S8 全 8 種", "-"),
                                              (best12, meta12, "WYCG", "S12 W/Y/C/G", "--")):
                L, V, C = b[(met, "all")].front()
                o = np.argsort(L)
                ax.step(L[o], V[o], where="pre", color=col, ls=ls, lw=1.5, label=f"{cn}・{lab}")
                for l, v, c in zip(L, V, C):
                    fr_rows.append(dict(case=cn, family=lab, metric=met, light=l, value=v,
                                        assignment=a3.decode(c, *meta, alpha)))
    for ax, yl in zip(axs, ("最も苦手な方向の基線", "Fisher 最小固有値")):
        ax.set_xlabel("平均光量")
        ax.set_ylabel(yl)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=6, ncol=2)
    fig.suptitle("実フィルタの近似（すべて仮定）で前線がどう動くか（実線: 扇形 8 分割・8 種、破線: 扇形 12 分割・W/Y/C/G）", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "a4_fronts_real.png"), dpi=110)
    plt.close(fig)
    with open(os.path.join(OUT, "a4_fronts_real.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fr_rows[0].keys()))
        w.writeheader()
        w.writerows(fr_rows)
    # 代表光量での前線の値（ケース比較）
    print("\n前線の値（平均光量が目標以上の設計の中で最も良い値）: ケース | S8 全 8 種 L≥0.5, ≥0.667 | S12 W/Y/C/G L≥0.667, ≥0.75")
    for cn in CASES:
        def val(fam, met, t):
            vv = [r["value"] for r in fr_rows if r["case"] == cn and r["family"] == fam and r["metric"] == met and r["light"] >= t - 1e-6]
            return max(vv) if vv else np.nan
        t23 = 2.0 / 3.0
        print(f"  {cn}: 最悪基線 {val('S8 全 8 種','mw',0.5):.3f}, {val('S8 全 8 種','mw',t23):.3f} | {val('S12 W/Y/C/G','mw',t23):.3f}, {val('S12 W/Y/C/G','mw',0.75):.3f}"
              f" || Fisher {val('S8 全 8 種','fi',0.5):.4f}, {val('S8 全 8 種','fi',t23):.4f} | {val('S12 W/Y/C/G','fi',t23):.4f}, {val('S12 W/Y/C/G','fi',0.75):.4f}")


if __name__ == "__main__":
    main()
