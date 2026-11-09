"""ぼけ形状の不一致（G 全開口と R/B 部分開口など）が ZNCC の偏りとピークの鋭さに与える影響（physB・2026-11-09）

目的: チャネル対の PSF の形が違う（例: 現行の R=半円 と G=全円）と、照合のピークが重心の差からずれる・鈍る。
      形の一致した対（同じ形を平行移動しただけの「三つの小円」や、鏡像の R–B）と比べる。
入力: なし
出力: physB_out/mismatch.txt, physB_out/mismatch_lsf.png（UTF-8）
内容:
  A. 連続の 1 次元 LSF（開口マスクをエッジの法線へ射影したもの）の相互相関のピーク位置 τ* と、重心の差 Δμ の比。
     雑音・窓・画素なし。1 本の段差エッジで |∇I| を照合するときの偏りの理論値に当たる [計算]。
  B. (i) 横方向 ZNCC の Python 再実装（Rust と一致を確認済み）で、現行形状の R–B・R–G・G–B を比べる
     （縦縞のバーコードと等方 1/f・c=3,6,12・σ=0 と 1.5）。d/(重心予測)、ピークの ZNCC 値、曲率、変位のばらつき。
  C. (ii) の 2 次元推定を 1 つのチャネル対だけで使い（|∇I| の ZNCC）、各形状の各対で同じ量を出す（等方 1/f）。
仮定: apsim.py 冒頭のとおり。「三つの小円」は比較用の仮想形状（半径 0.35 の円を半径 0.6 の位置に 120° おき・R/G/B）。
"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import apsim as ap

OUT = os.path.join(os.path.dirname(HERE), "physB_out")
LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


def shape_holes3(U, V, r0=0.35, R0=0.6):
    out = {}
    for k, a in zip(ap.CH, [210.0, 90.0, 330.0]):
        cx, cy = R0 * np.cos(np.deg2rad(a)), R0 * np.sin(np.deg2rad(a))
        out[k] = ((U - cx) ** 2 + (V - cy) ** 2 <= r0 ** 2).astype(float)
    return out


ap.SHAPES["holes3"] = (shape_holes3, "三つの小円（形一致の対照）")

# A. 連続 LSF -------------------------------------------------------------
log("=== A. 連続 LSF の相互相関ピーク τ* と重心の差 Δμ の比（1 本の段差エッジ・雑音なし・窓なし）===")
log("φ: エッジ法線の角度（0=縦エッジ）。比 τ*/Δμ が 1 なら重心どおり。Δμ が 0 に近い向きは省略")
n = 1201
U, V, inside = ap.aperture_grid(n)
X, Y = -U, V  # 像面（c>0）の座標: x=-U, y=V（y 下）
dt = 2.0 / 600
tb = np.arange(-1.2, 1.2 + dt, dt)
pairs_by_shape = {
    "cy_lr": [("R", "B"), ("R", "G"), ("G", "B")],
    "kaneko": [("R", "B"), ("R", "G")],
    "rgb3": [("R", "B"), ("R", "G"), ("G", "B")],
    "ycg3": [("R", "B"), ("R", "G")],
    "cmy3": [("R", "B"), ("R", "G")],
    "cy_msec": [("R", "B"), ("R", "G"), ("G", "B")],
    "holes3": [("R", "B"), ("R", "G")],
}
lsf_store = {}
for shape, pairs in pairs_by_shape.items():
    fn = ap.SHAPES[shape][0]
    T = fn(U, V)
    for i, j in pairs:
        line = []
        for phi in range(0, 180, 15):
            nv = np.array([np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))])
            t = X * nv[0] + Y * nv[1]
            Li = np.histogram(t[inside], bins=tb, weights=T[i][inside])[0]
            Lj = np.histogram(t[inside], bins=tb, weights=T[j][inside])[0]
            tc = 0.5 * (tb[1:] + tb[:-1])
            mu_i = (Li * tc).sum() / Li.sum()
            mu_j = (Lj * tc).sum() / Lj.sum()
            dmu = mu_i - mu_j
            # C(τ) = Σ L_i(t) L_j(t−τ): np.correlate(Li, Lj, 'full') のラグ
            C = np.correlate(Li, Lj, mode="full")
            lags = (np.arange(len(C)) - (len(Lj) - 1)) * dt
            k = int(np.argmax(C))
            if 0 < k < len(C) - 1:
                den = C[k - 1] - 2 * C[k] + C[k + 1]
                sub = 0.5 * (C[k - 1] - C[k + 1]) / den if den != 0 else 0.0
            else:
                sub = 0.0
            tau = lags[k] + sub * dt
            if (shape, i, j, phi) not in lsf_store and phi in (0, 45, 90):
                lsf_store[(shape, i, j, phi)] = (tc, Li / Li.sum() / dt, Lj / Lj.sum() / dt)
            line.append((phi, dmu, tau))
        txt = " ".join(f"{phi}:{(tau / dmu if abs(dmu) > 0.05 else np.nan):.2f}" for phi, dmu, tau in line)
        log(f"{ap.SHAPES[shape][1]} {i}–{j}: τ*/Δμ  {txt}")

# B. (i) 横方向 ZNCC で現行形状の対を比べる ----------------------------------------
log()
log("=== B. 現行 C/Y 左右・(i) 横 ZNCC（|Sobel x|・K25 S41 T20・Python 再実装）でチャネル対を比べる ===")
log("値: d/(重心予測) の中央値 / ピーク ZNCC の中央値 / 曲率 2p0−pn−pp の中央値 / 変位のばらつき 1.4826·MAD [px] / ĉ のばらつき [px]")
H = W = 192
SS = 4
ROI = (slice(40, 152), slice(40, 152))
fn = ap.SHAPES["cy_lr"][0]
p = ap.centroid_img(ap.shape_stats(fn))
for sc_name in ["barcode0", "iso"]:
    rng = np.random.default_rng(77)
    img = ap.scene_barcode(H, W, SS, 0.0, rng) if sc_name == "barcode0" else ap.scene_1overf(H, W, SS, rng)
    sc = ap.Scene(img, SS)
    for c in [3.0, 6.0, 12.0]:
        lin = ap.render(sc, fn, c)
        m, _, _ = ap.eval_mask_from_full(sc, c, roi=ROI)
        for sig in [0.0, 1.5]:
            u8 = ap.to_u8(lin, sig, np.random.default_rng(3))
            cells = []
            for a, b in [("R", "B"), ("R", "G"), ("G", "B")]:
                d, pk, cu = ap.est_i_py(u8, a, b)
                pred = c / 2 * (p[a][0] - p[b][0])
                mm = m & np.isfinite(d)
                if mm.sum() == 0:
                    cells.append(f"{a}–{b}: —")
                    continue
                dv = d[mm]
                med = np.median(dv)
                sdd = 1.4826 * np.median(np.abs(dv - med))
                cells.append(f"{a}–{b}: {med / pred:.3f} / {np.median(pk[mm]):.3f} / {np.median(cu[mm]):.4f} / {sdd:.3f} / {sdd * 2 / abs(p[a][0] - p[b][0]):.3f}")
            log(f"{sc_name} c={c:g} σ={sig}: " + " | ".join(cells))

# C. 2 次元推定・1 対だけ -------------------------------------------------------
log()
log("=== C. (ii) 2 次元推定を 1 対だけで（|∇I| ZNCC・等方 1/f・候補 c=−47..47）===")
log("値: ĉ/c の中央値 / ピーク ZNCC の中央値 / 変位のばらつき=ĉ のばらつき×|b_ij|/2 [px]（σ=1.5）/ 良好率")
CANDS = np.arange(-47, 48, 1.0)
rng = np.random.default_rng(78)
sc = ap.Scene(ap.scene_1overf(H, W, SS, rng), SS)
for shape, pairs in pairs_by_shape.items():
    fn = ap.SHAPES[shape][0]
    p = ap.centroid_img(ap.shape_stats(fn))
    for c in [3.0, 6.0, 12.0]:
        lin = ap.render(sc, fn, c)
        m, _, _ = ap.eval_mask_from_full(sc, c, roi=ROI)
        cells = []
        for i, j in pairs:
            bij = np.linalg.norm(p[i] - p[j])
            out = []
            for sig in [0.0, 1.5]:
                u8 = ap.to_u8(lin, sig, np.random.default_rng(5))
                res, _ = ap.est_scalar2d(u8, p, CANDS, pairs=((i, j),), use_bando=False, metrics=("zsum",))
                e, pk, cu = res["zsum"]
                mm = m & np.isfinite(e)
                ev = e[mm]
                med = np.median(ev)
                sd = 1.4826 * np.median(np.abs(ev - med))
                out.append((med / c, np.median(pk[mm]), sd * bij / 2, (np.abs(e[m] - c) <= 1).mean()))
            cells.append(f"{i}–{j}(|b|={bij:.2f}): {out[0][0]:.3f} / {out[0][1]:.3f} / {out[1][2]:.3f} / {out[1][3]:.2f}")
        log(f"{ap.SHAPES[shape][1]} c={c:g}: " + " | ".join(cells))

with open(os.path.join(OUT, "mismatch.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(LOG) + "\n")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
show = [("cy_lr", "R", "B", 0), ("cy_lr", "R", "G", 0), ("cy_lr", "R", "B", 45), ("kaneko", "R", "G", 0), ("holes3", "R", "B", 0)]
fig, axs = plt.subplots(1, len(show), figsize=(3.0 * len(show), 2.6))
for a, key in zip(axs, show):
    if key not in lsf_store:
        continue
    tc, Li, Lj = lsf_store[key]
    a.plot(tc, Li, color="#c00", label=key[1])
    a.plot(tc, Lj, color="#00c" if key[2] == "B" else "#080", label=key[2])
    a.set_title(f"{ap.SHAPES[key[0]][1]} {key[1]}–{key[2]} φ={key[3]}°", fontsize=8)
    a.set_xlabel("法線方向の位置（ぼけ半径=1）", fontsize=7)
    a.legend(fontsize=7)
fig.suptitle("[計算] エッジの法線へ射影した LSF（c>0・面積で正規化）", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "mismatch_lsf.png"), dpi=130)
