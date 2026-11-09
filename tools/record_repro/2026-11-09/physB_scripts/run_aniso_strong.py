"""横向きの構造の強さで結果が変わるか: 弱い異方性（run_natural.py の aniso）と強い異方性の 1/f 雑音（physB・2026-11-09）

目的: run_natural.py の aniso（振幅の重み 0.15+exp(−ψ²/(2·15²))）では、窓の支配方向はすべて横エッジ寄りなのに、
      現行 C/Y でも (ii) は良好率 1.00 だった（(i) は 0.21）。横の筋が短く（縦横比 4:1 程度）端や曲がりで横方向の変化が
      残るためと考えられる [推論]。筋を長くした強い異方性（重み 0.03+exp(−ψ²/(2·4²))）で同じ比較をし、(iv) も加える。
入力: なし
出力: physB_out/aniso_strong.txt, physB_out/aniso_scenes.png（UTF-8）
推定法: (i)（R–B 横・Rust）、(ii) color-line、(ii) zsum、(iv) PSF 合わせ（候補 −30〜30）。(i)(ii) は等方 1/f で求めた校正の比で割る。
      (iv) は割らない（raw）。σ=1.5。
仮定: apsim.py 冒頭のとおり。
"""
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import apsim as ap

OUT = os.path.join(os.path.dirname(HERE), "physB_out")
H = W = 192
SS = 4
ROI = (slice(40, 152), slice(40, 152))
CS = [3.0, 6.0, 12.0]
SHAPES = ["cy_lr", "rgb3", "cmy3", "ycg3", "cy_msec", "kaneko"]
CANDS = np.arange(-47, 48, 1.0)
CANDS4 = np.arange(-30, 31, 1.0)
LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


t0 = time.time()
scn = {}
for name, sig, fl, seeds in [("弱い異方性(σ15°,床0.15)", 15.0, 0.15, (201,)), ("強い異方性(σ4°,床0.03)", 4.0, 0.03, (211, 212))]:
    scn[name] = [ap.Scene(ap.scene_1overf(H, W, SS, np.random.default_rng(s), aniso_sigma_deg=sig, floor=fl), SS) for s in seeds]

# 向きの内訳
for name, lst in scn.items():
    for sc in lst:
        m, gx, gy = ap.eval_mask_from_full(sc, 6.0, roi=ROI)
        ang, coh = ap.local_orientation(gx, gy)
        h = (ang >= 67.5) & (ang <= 112.5)
        # 画素ごとの勾配の向き（窓をとらない）
        pang = np.degrees(np.arctan2(gy, gx)) % 180.0
        ph = (pang >= 67.5) & (pang <= 112.5)
        log(f"{name}: 評価画素 {m.sum()}, 窓の支配方向が横エッジ寄り {h[m].mean():.2f}, 画素の勾配が横エッジ寄り {ph[m].mean():.2f}, コヒーレンスの中央値 {np.median(coh[m]):.2f}")

log()
log("良好率 / ばらつき [px]（σ=1.5。(i)(ii) は校正の比で割った値、(iv) は raw）")
for c in CS:
    log(f"--- c={c:g}")
    for shape in SHAPES:
        fn = ap.SHAPES[shape][0]
        p = ap.centroid_img(ap.shape_stats(fn))
        use_i = shape in ap.LR_SHAPES
        k = ap.iso_gain(fn, p, c, CANDS, use_i)
        cells = []
        for name, lst in scn.items():
            acc = {}
            for j, sc in enumerate(lst):
                m, _, _ = ap.eval_mask_from_full(sc, c, roi=ROI)
                u8 = ap.to_u8(ap.render(sc, fn, c), 1.5, np.random.default_rng(j + int(c)))
                ests = ap.run_methods(u8, p, CANDS, use_i)
                e4, _, _ = ap.est_crossblur(u8, fn, CANDS4)
                for mt in (["i_hzncc"] if use_i else []) + ["ii_bando", "ii_zsum"]:
                    acc.setdefault(mt, []).append(ap.stats(ests[mt] / k[mt], c, m))
                acc.setdefault("iv_xblur", []).append(ap.stats(e4, c, m))
            parts = []
            for mt, sts in acc.items():
                n = sum(s["n"] for s in sts)
                g = sum(s["good"] * s["n"] for s in sts) / n
                sd = np.nanmean([s["sd"] for s in sts])
                parts.append(f"{mt} {g:.2f}/{sd:.2f}")
            cells.append(f"[{name}] " + ", ".join(parts))
        log(f"  {ap.SHAPES[shape][1]:<12s}: " + " | ".join(cells))
    print(f"c={c} 済 ({time.time() - t0:.0f} s)")

with open(os.path.join(OUT, "aniso_strong.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(LOG) + "\n")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
iso = ap.Scene(ap.scene_1overf(H, W, SS, np.random.default_rng(101)), SS)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.6))
for a, (t, sc) in zip(axs, [("等方 1/f", iso), ("弱い異方性（σ15°・床0.15）", scn["弱い異方性(σ15°,床0.15)"][0]), ("強い異方性（σ4°・床0.03）", scn["強い異方性(σ4°,床0.03)"][0])]):
    a.imshow(ap.box_down(sc.img, SS), cmap="gray", vmin=0, vmax=0.85)
    a.set_title(t, fontsize=9)
    a.set_xticks([])
    a.set_yticks([])
fig.suptitle("[計算] 被写体（線形の明るさ・ぼけ前）", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "aniso_scenes.png"), dpi=120)
