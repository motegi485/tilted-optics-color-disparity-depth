"""形状の幾何の表・合成器の自己検査・アオリの PSF の光線追跡による確認（physB・2026-11-09）

目的:
  1. 各開口形状の光量・重心・エッジの向きごとの基線を出し、研究方針.md §3 の表と照合する（形状の定義の確認）。
  2. 合成器の PSF の重心が (c/2)·p_k になっているかを確かめる。
  3. (i) の Python 再実装（est_i_py）が開発リポの Rust 実装（core.disparity）と一致するかを確かめる。
  4. 開発リポ theme1_physics/optics.py の光線追跡（読み取りのみで import）で、アオリの PSF の縦横比と、
     上下の半開口の重心の差（縦の基線）を θ=30°（現行）と θ≈3.97°（路面合焦）で求める。
  5. (ii)/(iii) の 1 枚あたりの計算時間を測る。
入力: なし（開発リポのコードを読み取りで import）
出力: physB_out/geometry.txt（UTF-8）、physB_out/shape_masks.png
仮定: apsim.py 冒頭のとおり。光線追跡は optics.py の薄レンズモデル（回折・収差なし）。
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
os.makedirs(OUT, exist_ok=True)
LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


# 1. 形状の表 -------------------------------------------------------------
log("=== 1. 形状の幾何（開口半径 1・理想フィルタ・重心のみ）===")
log("基線は (p_i−p_j)·n の最大（研究方針.md §7 と同じ定義・半径 1 単位）。RSS は 3 ペアの二乗和の平方根")
log("形状 | 光量 R,G,B（平均） | 重心 R,G,B（像面 x 右・y 下, c>0） | 縦エッジ(φ=0) | 横エッジ(φ=90) | 最悪（角度） | RSS 最悪")
angles = np.arange(0, 180, 0.5)
geo = {}
for name, (fn, label) in ap.SHAPES.items():
    st = ap.shape_stats(fn)
    p = ap.centroid_img(st)
    # 半径 1 の開口で「重心の差」= p_i − p_j（centroid_img は半径 1 あたり）
    mx, rs = ap.baseline_table(p, angles)
    a = [st[k]["area"] for k in ap.CH]
    geo[name] = (st, p, mx, rs)
    iw = int(np.argmin(mx))
    log(f"{label} | {a[0]:.2f},{a[1]:.2f},{a[2]:.2f} ({np.mean(a):.2f}) | "
        + ", ".join(f"({p[k][0]:+.3f},{p[k][1]:+.3f})" for k in ap.CH)
        + f" | {mx[0]:.3f} | {mx[180]:.3f} | {mx[iw]:.3f} ({angles[iw]:.1f}°) | {rs.min():.3f}")
log("研究方針.md §3 の表: 現行 0.85/0/0, 45°回転 0.60/0.60/0, RGB 0.95/0.83/0.83, CMY 0.48/0.41/0.41, Y/C/G 0.95/0.28/0.28")

# 2. PSF の重心の検査 ------------------------------------------------------
log()
log("=== 2. PSF（ss=4 格子）の重心と (c/2)·p_k の比較（c=±10, sy=1 と sy=cos30°）===")
for name in ["cy_lr", "rgb3", "kaneko", "cy_msec"]:
    fn = ap.SHAPES[name][0]
    st = ap.shape_stats(fn)
    for sy in [1.0, np.cos(np.deg2rad(30))]:
        p = ap.centroid_img(st, sy)
        for c in [10.0, -10.0]:
            ks = ap.psf_kernels(fn, c, ss=4, sy=sy)
            msg = []
            for k in ap.CH:
                kk = ks[k]
                h, w = kk.shape
                yy, xx = np.mgrid[0:h, 0:w]
                cx = ((xx - w // 2) * kk).sum() / 4.0
                cy = ((yy - h // 2) * kk).sum() / 4.0
                ex = c / 2 * p[k]
                msg.append(f"{k}: ({cx:+.3f},{cy:+.3f}) 期待 ({ex[0]:+.3f},{ex[1]:+.3f})")
            log(f"{name} sy={sy:.3f} c={c:+.0f}: " + " | ".join(msg))

# 3. (i) の Python 再実装と Rust の照合 ----------------------------------------
log()
log("=== 3. (i) Python 再実装 vs 開発リポの Rust 実装（core.disparity.calcDisparityByZNCC, K25 S41 T20）===")
rng = np.random.default_rng(1)
H = W = 192
ss = 4
for sc_name in ["1/f", "barcode0", "barcode30"]:
    if sc_name == "1/f":
        img = ap.scene_1overf(H, W, ss, rng)
    elif sc_name == "barcode0":
        img = ap.scene_barcode(H, W, ss, 0.0, rng)
    else:
        img = ap.scene_barcode(H, W, ss, 30.0, rng)
    sc = ap.Scene(img, ss)
    for c in [4.0, 10.0]:
        lin = ap.render(sc, ap.shape_cy_lr, c)
        u8 = ap.to_u8(lin, 1.5, np.random.default_rng(2))
        dr = ap.est_i_rust(u8)
        dp, pk, cu = ap.est_i_py(u8)
        both = np.isfinite(dr) & np.isfinite(dp)
        only_r = np.isfinite(dr) & ~np.isfinite(dp)
        only_p = ~np.isfinite(dr) & np.isfinite(dp)
        diff = np.abs(dr[both] - dp[both])
        log(f"{sc_name} c={c}: 両方有効 {both.sum()}, Rust のみ {only_r.sum()}, Py のみ {only_p.sum()}, "
            f"|差| 最大 {diff.max() if diff.size else np.nan:.2e}, |差|>1e-3 の画素 {(diff > 1e-3).sum()}, "
            f"d の中央値 {np.nanmedian(dr):.3f}（βc={ap.BETA * c:.3f}）")

# 4. アオリの光線追跡 -------------------------------------------------------
log()
log("=== 4. 光線追跡（optics.py の _trace・薄レンズ）: アオリの PSF の縦横比と上下・左右の半開口の重心差 ===")
sys.path.insert(0, os.path.join(ap.ROOT, "tools", "record_repro", "2026-11-07", "theme1_physics"))
import optics as op

rigs = [op.Rig("現行 θ=30°", 30.0, 109.3, Hc=1330.0), op.rig_parallel("路面合焦 θ≈3.97°", 0.0)]
for rig in rigs:
    log(f"[{rig.name}] θ={rig.theta_deg:.2f}°, A={rig.A:.2f} mm, F={rig.F:.0f} px(1080 幅), A/J={rig.row_coef:.4f}")
    ax, at = op._aperture_grid(rig.A, 241)
    ex, et = op._aperture_edge(rig.A)
    for Zc in [1500.0, 2500.0, 4000.0, 20000.0]:
        for Yc_frac in [-0.03, 0.0, 0.03]:
            Yc = Yc_frac * Zc
            X = 0.0
            try:
                col_e, row_e = op._trace(rig, X, Yc, Zc, ex, et)
            except ValueError:
                continue
            cx_ = col_e.max() - col_e.min()
            cy_ = row_e.max() - row_e.min()
            col, row = op._trace(rig, X, Yc, Zc, ax, at)
            d_lr = col[ax > 0].mean() - col[ax < 0].mean()
            d_tb = row[at > 0].mean() - row[at < 0].mean()
            # 上下半開口の重心差の横成分（せん断の有無）
            d_tb_x = col[at > 0].mean() - col[at < 0].mean()
            if cx_ < 1.0:
                continue
            log(f"  Z={Zc:.0f} Y={Yc:+.0f}: c_x={cx_:.2f} px, c_y/c_x={cy_ / cx_:.4f} (cosθ={np.cos(rig.theta):.4f}), "
                f"|d_LR|/c_x={abs(d_lr) / cx_:.4f}, |d_TB|/c_x={abs(d_tb) / cx_:.4f} (β·cosθ={ap.BETA * np.cos(rig.theta):.4f}), "
                f"d_TB の横成分/c_x={d_tb_x / cx_:+.4f}")

# 5. 計算時間 -------------------------------------------------------------
log()
log("=== 5. 計算時間（192×192, 候補 c=-47..47 の 95 個, K=25）===")
img = ap.scene_1overf(H, W, ss, rng)
sc = ap.Scene(img, ss)
fn = ap.SHAPES["rgb3"][0]
p = ap.centroid_img(ap.shape_stats(fn))
t0 = time.time()
lin = ap.render(sc, fn, 6.0)
t1 = time.time()
u8 = ap.to_u8(lin, 1.5, rng)
res, sel = ap.est_scalar2d(u8, p, np.arange(-47, 48, 1.0))
t2 = time.time()
m, _, _ = ap.eval_mask_from_full(sc, 6.0, roi=(slice(40, 152), slice(40, 152)))
for k in res:
    log(f"  {k}: {ap.stats(res[k][0], 6.0, m)}")
log(f"  描画 {t1 - t0:.2f} s, 推定（3 指標）{t2 - t1:.2f} s")

# マスクの図 --------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
names = [n for n in ap.SHAPES if n != "full"]
fig, axs = plt.subplots(1, len(names), figsize=(2.0 * len(names), 2.4))
U, V, inside = ap.aperture_grid(301)
for a, n in zip(axs, names):
    T = ap.SHAPES[n][0](U, V)
    rgb = np.stack([np.where(inside, T[k], 1.0) for k in ap.CH], axis=-1)
    rgb[~inside] = 1.0
    a.imshow(rgb, extent=(-1, 1, -1, 1))
    a.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="k", lw=0.8))
    st = ap.shape_stats(ap.SHAPES[n][0])
    for k, col in zip(ap.CH, ["#c00", "#080", "#00c"]):
        a.plot(st[k]["cU"], st[k]["cV"], "o", ms=5, mfc=col, mec="k")
    a.set_title(ap.SHAPES[n][1], fontsize=8)
    a.set_xticks([])
    a.set_yticks([])
fig.suptitle("開口座標（左右・上下は研究方針.md §7 の表記）。色＝通すチャネル（R,G,B）、点＝各チャネルの重心", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "shape_masks.png"), dpi=130)

with open(os.path.join(OUT, "geometry.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(LOG) + "\n")
