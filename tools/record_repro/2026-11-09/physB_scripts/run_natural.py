"""自然画像に近い被写体（1/f 雑音）で、開口形状ごとの推定を比べる（physB・2026-11-09）

目的: 等方的な 1/f 雑音と、横向きの構造が多い 1/f 雑音（塀・横板のような場面のまね）で、
      形状 × 推定法 × ぼけ量 × 雑音（等しい雑音／光量に応じた雑音）の成績を出す。局所のエッジの向きごとの内訳も出す。
      あわせて、色のある被写体（チャネルごとに模様が少し違う）での崩れ方を 1 条件だけ見る。
入力: なし
出力: physB_out/natural.csv, physB_out/natural.txt, physB_out/natural.png（UTF-8）
推定法: run_orient.py と同じ（i_hzncc は R–B の基線が横の形状だけ）。
雑音: eqS = 全チャネル σ=S 階調（S=0.5, 1.5, 4）／light = σ_k = 1.5·sqrt(0.5/a_k)（a_k はチャネル k の平均透過。
      露光一定で光子雑音が支配的、と仮定し、現行の R/B 半開口で σ=1.5 になるよう規格化 [仮定]）。
被写体: iso = 振幅 ∝ 1/f の等方な雑音、aniso = 周波数ベクトルが縦軸に近い成分（横向きの構造）を強めた 1/f 雑音
      （振幅の重み 0.15 + exp(−ψ²/(2·15²))）、color = iso の明るさに、R と B だけ独立な 1/f の色むら（±25 %・σ）を掛けたもの。
校正の模擬: すべての値は、別の種（9001, 9002）の等方 1/f・σ=1.5 で求めた比 k=median(ĉ/c) で割った ĉ/k で評価する
      （apsim.iso_gain。向きによらない一定の比は実機の校正で吸収される、とみなす）。
評価: 全開口で描いた画像の |∇I|>20 の画素（中央 112×112）。向きの内訳は全開口画像の窓内構造テンソルの法線角で
      縦エッジ寄り（法線 0±22.5°）・斜め・横エッジ寄り（法線 90±22.5°）に分ける。
仮定: apsim.py 冒頭のとおり。
"""
import csv
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import scipy.fft as sfft

import apsim as ap

OUT = os.path.join(os.path.dirname(HERE), "physB_out")
H = W = 192
SS = 4
ROI = (slice(40, 152), slice(40, 152))
CS = [3.0, 6.0, 12.0]
SHAPES = ["cy_lr", "cy_rot45", "rgb3", "cmy3", "ycg3", "kaneko", "cy_msec", "cy_topW"]
LR = {"cy_lr", "kaneko", "cy_msec", "cy_topW"}
CANDS = np.arange(-47, 48, 1.0)
NOISES = ["eq1.5", "eq0.5", "eq4.0", "light"]

t0 = time.time()
scene_defs = [("iso", 101), ("iso", 102), ("aniso", 201), ("aniso", 202), ("color", 301)]
scenes = {}
for kind, seed in scene_defs:
    rng = np.random.default_rng(seed)
    if kind == "aniso":
        L = ap.scene_1overf(H, W, SS, rng, aniso_sigma_deg=15.0)
    else:
        L = ap.scene_1overf(H, W, SS, rng)
    if kind == "color":
        chi_r = ap.scene_1overf(H, W, SS, rng, mean=0.0, std=1.0, clip=False)
        chi_b = ap.scene_1overf(H, W, SS, rng, mean=0.0, std=1.0, clip=False)
        imgs = dict(R=np.clip(L * (1 + 0.25 * chi_r), 0.02, 0.9), G=L, B=np.clip(L * (1 + 0.25 * chi_b), 0.02, 0.9))
    else:
        imgs = dict(R=L, G=L, B=L)
    scenes[(kind, seed)] = {k: ap.Scene(imgs[k], SS) for k in ap.CH}


def render_color(sc, fn, c):
    ks = ap.psf_kernels(fn, c, SS)
    out = {}
    for k in ap.CH:
        kf = ap.kernel_fft(ks[k], sc[k].shape)
        img = sfft.irfft2(sc[k].F * kf, s=sc[k].shape, workers=-1)
        out[k] = ap.box_down(img, SS)
    return out


# 評価画素と向き
masks, bins = {}, {}
for key, sc in scenes.items():
    for c in CS:
        m, gx, gy = ap.eval_mask_from_full(sc["G"], c, roi=ROI)
        ang, coh = ap.local_orientation(gx, gy)
        masks[(key, c)] = m
        v = (ang <= 22.5) | (ang >= 157.5)
        h = (ang >= 67.5) & (ang <= 112.5)
        bins[(key, c)] = dict(all=m, v=m & v, o=m & ~v & ~h, h=m & h)

LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


log("=== 評価画素の向きの内訳（全開口画像の窓内構造テンソル・c=6）===")
for key in scenes:
    b = bins[(key, 6.0)]
    n = b["all"].sum()
    log(f"{key}: 画素 {n}, 縦エッジ寄り {b['v'].sum() / n:.2f}, 斜め {b['o'].sum() / n:.2f}, 横エッジ寄り {b['h'].sum() / n:.2f}")

rows = []
gains = {}
for shape in SHAPES:
    fn = ap.SHAPES[shape][0]
    st = ap.shape_stats(fn)
    p = ap.centroid_img(st)
    for c in CS:
        gains[(shape, c)] = ap.iso_gain(fn, p, c, CANDS, shape in LR)
    for key, sc in scenes.items():
        for c in CS:
            lin = render_color(sc, fn, c)
            for nz in NOISES:
                if key[0] == "color" and nz != "eq1.5":
                    continue
                if nz == "light":
                    sig = {k: 1.5 * np.sqrt(0.5 / st[k]["area"]) for k in ap.CH}
                else:
                    sig = float(nz[2:])
                u8 = ap.to_u8(lin, sig, np.random.default_rng(int(c * 10) + key[1]))
                ests = ap.run_methods(u8, p, CANDS, shape in LR)
                for mt, e in ests.items():
                    kk = gains[(shape, c)][mt]
                    for bn, bm in bins[(key, c)].items():
                        rows.append(dict(scene=key[0], seed=key[1], shape=shape, c=c, noise=nz, method=mt, bin=bn, k=kk, **ap.stats(e / kk, c, bm)))
    print(f"{shape} 済 ({time.time() - t0:.0f} s)")

with open(os.path.join(OUT, "natural.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)


def agg(scene, shape, c, noise, method, bn):
    rr = [r for r in rows if r["scene"] == scene and r["shape"] == shape and r["c"] == c and r["noise"] == noise
          and r["method"] == method and r["bin"] == bn and r["n"] > 0]
    if not rr:
        return None
    n = sum(r["n"] for r in rr)
    good = sum(r["good"] * r["n"] for r in rr) / n
    valid = sum(r["valid"] * r["n"] for r in rr) / n
    bias = np.nanmean([r["bias"] for r in rr])
    sd = np.nanmean([r["sd"] for r in rr])
    return dict(good=good, valid=valid, bias=bias, sd=sd)


methods = ["i_hzncc", "ii_zsum", "ii_bando", "iii_zw"]
for scene in ["iso", "aniso"]:
    log()
    log(f"=== {scene}: 良好率 [縦寄り/斜め/横寄り] と全体のばらつき（σ=1.5 等しい雑音）===")
    for c in CS:
        log(f"--- c={c:g}")
        log("形状 | " + " | ".join(methods))
        for shape in SHAPES:
            cells = []
            for mt in methods:
                a = agg(scene, shape, c, "eq1.5", mt, "all")
                if a is None:
                    cells.append("—")
                    continue
                parts = [agg(scene, shape, c, "eq1.5", mt, b_) for b_ in ["v", "o", "h"]]
                cells.append(f"{a['good']:.2f} [" + "/".join(f"{x['good']:.2f}" if x else "-" for x in parts) + f"] sd {a['sd']:.2f} 偏 {a['bias']:+.2f}")
            log(f"{ap.SHAPES[shape][1]} | " + " | ".join(cells))

log()
log("=== 雑音と光量（iso と aniso・c=6・iii_zw と ii_bando・良好率／ばらつき）===")
for scene in ["iso", "aniso"]:
    for mt in ["iii_zw", "ii_bando", "i_hzncc"]:
        log(f"--- {scene} {mt}: " + " | ".join(NOISES))
        for shape in SHAPES:
            cells = []
            for nz in NOISES:
                a = agg(scene, shape, 6.0, nz, mt, "all")
                cells.append("—" if a is None else f"{a['good']:.2f}/{a['sd']:.2f}")
            log(f"  {ap.SHAPES[shape][1]:<14s}: " + " | ".join(cells))

log()
log("=== 色のある被写体（color・σ=1.5）vs 無彩色（iso）: 良好率／ばらつき ===")
for c in CS:
    log(f"--- c={c:g}")
    for shape in SHAPES:
        cells = []
        for mt in methods:
            a = agg("color", shape, c, "eq1.5", mt, "all")
            b = agg("iso", shape, c, "eq1.5", mt, "all")
            cells.append("—" if a is None else f"{mt}: {a['good']:.2f}/{a['sd']:.2f}（iso {b['good']:.2f}/{b['sd']:.2f}）")
        log(f"  {ap.SHAPES[shape][1]:<14s}: " + " | ".join(cells))
log(f"所要 {time.time() - t0:.0f} s")
with open(os.path.join(OUT, "natural.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(LOG) + "\n")

# 図: aniso と iso の良好率（iii_zw と i）を形状ごとに棒で
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
fig, axs = plt.subplots(2, 3, figsize=(14, 6.5), sharey=True)
xs = np.arange(len(SHAPES))
for i, scene in enumerate(["iso", "aniso"]):
    for j, c in enumerate(CS):
        a = axs[i, j]
        for k, (mt, off, col) in enumerate([("i_hzncc", -0.27, "#999999"), ("ii_bando", 0.0, "#1f77b4"), ("iii_zw", 0.27, "#d62728")]):
            vals = []
            for shape in SHAPES:
                r = agg(scene, shape, c, "eq1.5", mt, "all")
                vals.append(np.nan if r is None else r["good"])
            a.bar(xs + off, vals, width=0.26, color=col, label=mt)
        a.set_xticks(xs)
        a.set_xticklabels([ap.SHAPES[s][1] for s in SHAPES], rotation=40, ha="right", fontsize=7)
        a.set_title(f"{scene}  c={c:g} px", fontsize=9)
        a.grid(axis="y", alpha=0.3)
axs[0, 0].set_ylabel("良好率（|ĉ−c|≤1 px）")
axs[1, 0].set_ylabel("良好率（|ĉ−c|≤1 px）")
axs[0, 0].legend(fontsize=7)
fig.suptitle("[計算] 1/f 雑音の被写体（上: 等方・下: 横向きの構造が多い）での良好率（σ=1.5・幾何光学・理想フィルタ）", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "natural.png"), dpi=120)
