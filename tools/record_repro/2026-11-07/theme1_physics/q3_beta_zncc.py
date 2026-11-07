"""ZNCC（|∂x| の照合）が返す d は、重心の差 βc と一致するか。実装の関数をそのまま使う。

入力: 1 本の縦エッジ／ランダム四角模様（校正板に近い）に半円 PSF を掛けて 8bit にしたもの。
偏り（pixel-locking を含む）を、d を細かく振って測る。枠内で平均しても下がらない成分。
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
ROOT = "C:/Users/t5424001/dev/tilted_optics_color_disparity_depth"
sys.path.insert(0, ROOT)
import cv2
import numpy as np

import optics as op
import spectral as sp
import synth as sy
from core.disparity import calcDisparityByZNCC

BETA = op.BETA
SS = 4
rng = np.random.default_rng(20261106)

ideal_white = {"R": (1.0, 0.0), "G": (0.5, 0.5), "B": (0.0, 1.0)}
real_white = {"R": (1.0, 0.0), "G": (0.7, 0.3), "B": (0.0, 1.0)}


def lsf_peak_beta():
    """連続の LSF で、相互相関のピーク位置を出す（窓なし・理想）"""
    r = 1.0
    x = np.linspace(-1.5, 1.5, 6001)
    dx = x[1] - x[0]
    Lb = np.where((x >= 0) & (x <= r), np.sqrt(np.maximum(r * r - x * x, 0)), 0.0)
    Lr = np.where((x <= 0) & (x >= -r), np.sqrt(np.maximum(r * r - x * x, 0)), 0.0)
    Lg = Lr + Lb
    def peak(a, b):
        xc = np.correlate(a, b, mode="same")
        lags = (np.arange(len(a)) - len(a) // 2) * dx
        return lags[np.argmax(xc)]
    print("連続 LSF の相互相関ピークの位置（開口半径 1）: R-B", abs(peak(Lr, Lb)), " 重心差 =", 2 * 4 / (3 * np.pi), "  → β_peak=", abs(peak(Lr, Lb)) / 2)
    print("  R-G", abs(peak(Lr, Lg)), " 重心差 =", 4 / (3 * np.pi))
    Lg73 = 0.7 * Lr + 0.3 * Lb
    print("  G を Y:C=0.7:0.3 とした R-G", abs(peak(Lr, Lg73)), " 重心差 =", (1 - 0.4) * 4 / (3 * np.pi), "  G-B", abs(peak(Lg73, Lb)), " 重心差 =", (1 + 0.4) * 4 / (3 * np.pi))


lsf_peak_beta()


def scene_edge(H, W, x0, ss):
    m = np.zeros((H * ss, W * ss), np.float32)
    m[:, int(x0 * ss) :] = 1.0
    return m


def scene_squares(H, W, grain_px, ss, seed=0):
    r = np.random.default_rng(seed)
    ny, nx = int(H / grain_px) + 2, int(W / grain_px) + 2
    g = (r.random((ny, nx)) > 0.5).astype(np.float32)
    m = cv2.resize(g, (W * ss, H * ss), interpolation=cv2.INTER_NEAREST)
    return m


def render_pair(mask, c, W, ss, hi=0.7, lo=0.05):
    kY, kC = sy.half_kernels(c, abs(c) * 0.998, ss)
    a = sy.conv(mask, kY) * 2
    b = sy.conv(mask, kC) * 2
    out = {}
    for k in ("R", "G", "B"):
        out[k] = lo + (hi - lo) * (W[k][0] * a + W[k][1] * b)
    return {k: sy.box_down(np.asarray(v), ss) for k, v in out.items()}


def measure(lin, pair=("R", "B"), noise=1.5, use420=False, k=25, s=41, t=20, roi=None, seed=1):
    r_ = np.random.default_rng(seed)
    bgr = sy.to_u8(lin, noise=noise, rng=r_)
    if use420:
        bgr = sy.chroma420(bgr)
    idx = {"B": 0, "G": 1, "R": 2}
    d = calcDisparityByZNCC(bgr[:, :, idx[pair[0]]], bgr[:, :, idx[pair[1]]], k, s, t)
    if roi is not None:
        d = d[roi]
    return d[np.isfinite(d)]


print()
print("=== 1. 1 本の縦エッジ: d_ZNCC / c（理想の半円 PSF・ガンマ後ノイズ σ=1.5 階調 [仮定]）。重心なら 0.424 ===")
print("c[px] | R-B: 中央値, d/c, 有効画素 | 4:2:0 あり | R-G (G半々) d/c | G-B (G半々) d/c | R-G (G 0.7:0.3) d/c | G-B (G 0.7:0.3) d/c")
H, W_ = 120, 320
for c in [1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
    m = scene_edge(H, W_, 160, SS)
    lin = render_pair(m, c, ideal_white, SS)
    lin2 = render_pair(m, c, real_white, SS)
    res = {}
    roi = (slice(20, 100), slice(100, 220))
    a = measure(lin, ("R", "B"), roi=roi)
    a4 = measure(lin, ("R", "B"), use420=True, roi=roi)
    b = measure(lin, ("R", "G"), roi=roi)
    e = measure(lin, ("G", "B"), roi=roi)
    b2 = measure(lin2, ("R", "G"), roi=roi)
    e2 = measure(lin2, ("G", "B"), roi=roi)
    f = lambda v: (np.median(v) / c if v.size else np.nan)
    print(f"{c} | {np.median(a) if a.size else np.nan:.3f}, {f(a):.3f}, {a.size} | {f(a4):.3f} ({a4.size}) | {f(b):.3f} | {f(e):.3f} | {f(b2):.3f} | {f(e2):.3f}")

print()
print("=== 2. ランダム四角模様（粒の幅 15 px@1080≒校正板 5.6 mm を 2 m）: d/c ===")
print("c[px] | R-B d/c, |d−βc| の中央値 | 4:2:0 あり d/c | R-G(0.7:0.3) d/c | G-B(0.7:0.3) d/c")
H, W_ = 240, 480
roi = (slice(40, 200), slice(70, 410))
for c in [1.5, 3, 6, 12, 24]:
    m = scene_squares(H, W_, 15, SS)
    lin = render_pair(m, c, ideal_white, SS)
    lin2 = render_pair(m, c, real_white, SS)
    a = measure(lin, ("R", "B"), roi=roi)
    a4 = measure(lin, ("R", "B"), use420=True, roi=roi)
    b2 = measure(lin2, ("R", "G"), roi=roi)
    e2 = measure(lin2, ("G", "B"), roi=roi)
    print(f"{c} | {np.median(a)/c:.3f}, {np.median(np.abs(a-BETA*c)):.3f} | {np.median(a4)/c:.3f} | {np.median(b2)/c:.3f} | {np.median(e2)/c:.3f}")

print()
print("=== 3. 真の d を細かく振ったときの偏り（縦エッジ・1080 幅・K=25,S=83,T=20／528 幅・K=25,S=41,T=20） ===")
print("偏りは枠内で平均しても下がらない（同じ枠の d はほぼ同じ小数部を持つ）。")
H, W_ = 120, 320
dvals = np.arange(0.2, 6.01, 0.1)
for label, scale, k, s, t in [("1080幅 K25 S83 T20", 1.0, 25, 83, 20), ("528幅 K25 S41 T20", op.R_1080_TO_528, 25, 41, 20)]:
    for use420 in [False, True]:
        err = []
        vals = []
        for dtrue in dvals:
            c = dtrue * scale / BETA  # 1080 幅の c
            m = scene_edge(H * 2, W_ * 2, 320, SS)
            lin = render_pair(m, c, ideal_white, SS)
            bgr = sy.to_u8(lin, noise=1.5, rng=np.random.default_rng(int(dtrue * 1000)))
            if use420:
                bgr = sy.chroma420(bgr)
            if scale != 1.0:
                bgr = cv2.resize(bgr, (int(round(bgr.shape[1] / scale)), int(round(bgr.shape[0] / scale))), interpolation=cv2.INTER_AREA)
            dm = calcDisparityByZNCC(bgr[:, :, 2], bgr[:, :, 0], k, s, t)
            dm = dm[np.isfinite(dm)]
            if dm.size == 0:
                err.append(np.nan)
                vals.append(0)
                continue
            err.append(np.median(dm) - dtrue)
            vals.append(dm.size)
        err = np.array(err)
        ok = np.isfinite(err)
        print(f"{label} 420={use420}: 偏り 平均 {np.nanmean(err):+.3f}, RMS {np.sqrt(np.nanmean(err**2)):.3f}, 最大 {np.nanmax(np.abs(err)):.3f} px | "
              f"d<1: 平均 {np.nanmean(err[dvals<1]):+.3f} | 1≤d<3: 平均 {np.nanmean(err[(dvals>=1)&(dvals<3)]):+.3f}, 振れ {np.nanstd(err[(dvals>=1)&(dvals<3)]):.3f} | "
              f"d≥3: 平均 {np.nanmean(err[dvals>=3]):+.3f}, 振れ {np.nanstd(err[dvals>=3]):.3f} | 欠損 {int((~ok).sum())}/{len(err)}")
        print("   d=" + " ".join(f"{d:.1f}:{e:+.2f}" for d, e in zip(dvals[::6], err[::6])))

print()
print("=== 4. 実効の ZNCC 基線: 重心の値と比べる（ランダム模様・c=6・G の重みを振る） ===")
print("G の Y 側の重み | R-G 重心比 | R-G ZNCC 比 d_RG/d_RB | G-B ZNCC 比 d_GB/d_RB")
H, W_ = 240, 480
roi = (slice(40, 200), slice(70, 410))
for c in [3, 6, 12, 24]:
    print(f"c={c}")
    for wy in [0.5, 0.6, 0.7, 0.8, 0.9]:
        W = {"R": (1.0, 0.0), "G": (wy, 1 - wy), "B": (0.0, 1.0)}
        m = scene_squares(H, W_, 15, SS)
        lin = render_pair(m, c, W, SS)
        a = measure(lin, ("R", "B"), roi=roi)
        b = measure(lin, ("R", "G"), roi=roi)
        e = measure(lin, ("G", "B"), roi=roi)
        print(f"  {wy} | {(1-wy):.2f} | {np.median(b)/np.median(a):.3f} | {np.median(e)/np.median(a):.3f}")
