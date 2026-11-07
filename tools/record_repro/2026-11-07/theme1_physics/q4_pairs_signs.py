"""Q4 の続き: チャネル対ごとの灯火からの視差、標識の色ごとの使えるチャネル対、消灯レンズの反射を消したときの感度、実写の赤いレンズ。
Q3 の補足: PSF ごとの MTF50（輝度の PSF は全円か）。
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
import q3_signal_render as q3

SS = 4
BETA = op.BETA
SP_M = q3.SP_M
SP_N = q3.SP_N
Ra = op.rig_parallel("(a)", 0.0)
R10 = op.rig_parallel("(b1.0)", 1000.0)
IDX = {"B": 0, "G": 1, "R": 2}


def zncc(u, a, b, K, S, T):
    return calcDisparityByZNCC(u[:, :, IDX[a]], u[:, :, IDX[b]], K, S, T)


print("=== 1. 灯火の縁だけから取れるチャネル対（1080 幅 4:4:4・K25 S83 T20・ノイズ 1.5・消灯レンズと本体の差を消した場合 unlit=0.03 も比較） ===")
print("灯色 | z | c | unlit | 対 | 期待（実効基線 × βc） | 灯火の縁の有効px | 中央値 | |d-期待|")
zs = [30e3, 50e3, 100e3]
for lit, idx in [("LED赤630", 0), ("LED黄590", 1), ("LED青505", 2)]:
    for z in zs:
        for unlit in [0.05, 0.03]:
            lin, ms, c, info = q3.build(Ra, z, idx, lit, SP_M, unlit=unlit)
            W = SP_M.led[lit]
            for a, b in [("R", "B"), ("R", "G"), ("G", "B")]:
                ea = sp.centroid_e(*W[a])
                eb = sp.centroid_e(*W[b])
                exp = (ea - eb) / 2 * BETA * c if np.isfinite(ea) and np.isfinite(eb) else np.nan
                res = []
                for seed in range(6):
                    rng = np.random.default_rng(seed + 500)
                    u = sy.to_u8(lin, noise=1.5, rng=rng)
                    dm = zncc(u, a, b, 25, 83, 20)
                    H_, W_ = dm.shape
                    yy, xx = np.mgrid[0:H_, 0:W_]
                    lc = ms["centers"][idx]
                    rr = np.hypot(xx + 0.5 - lc[0], yy + 0.5 - lc[1])
                    end = (np.abs(yy + 0.5 - ms["cy"]) <= ms["hh"]) & (np.abs(np.abs(xx + 0.5 - ms["cx"]) - ms["hw"]) <= max(6, abs(c)))
                    m = (np.abs(rr - info["lamp_px"] / 2) <= max(3.0, abs(c) / 2)) & ~end & np.isfinite(dm)
                    x = dm[m]
                    res.append((x.size, np.median(x) if x.size else np.nan))
                res = np.array(res, dtype=float)
                med = np.nanmean(res[:, 1]) if np.isfinite(res[:, 1]).any() else np.nan
                print(f"{lit} | {z/1000:.0f} | {c:.1f} | {unlit} | {a}-{b} | {exp:+.2f} | {res[:,0].mean():.0f} | {med:+.2f} | {abs(med-exp):.2f}")

print()
print("=== 2. 標識（路面合焦 (a)・空 0.6 の前）。使えるチャネル対と視差（実効基線は白の重みで R-B=1 に対する値） ===")
S = SP_M.src
exp_scale = 0.8 / 0.85


def sign_scene(kind, z, rig, H_c, size_mm, sky=0.6):
    c = float(rig.blur_signed(H_c, z))
    cy = abs(c) * np.cos(rig.theta)
    d_px = rig.F * size_mm / z
    margin = int(max(70, abs(c) * 1.2 + 62))
    Wd = int(np.ceil(d_px + 2 * margin))
    H = W = Wd * SS
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx = cy0 = W / 2.0
    r = np.hypot(xx - cx, yy - cy0)
    R_ = d_px * SS / 2
    comps = []
    if kind == "白地赤枠":
        disc = r <= R_
        ring = disc & (r >= R_ * 0.72)
        inner = disc & ~ring
        bar = inner & ((np.abs(xx - cx + 0.28 * R_) <= 0.08 * R_) | (np.abs(xx - cx - 0.28 * R_) <= 0.08 * R_)) & (np.abs(yy - cy0) <= 0.45 * R_)
        comps = [(~disc, SP_M.white, sky), (ring, S["赤塗料"], exp_scale), (inner & ~bar, S["白"], exp_scale), (bar, S["青塗料"], exp_scale)]
    elif kind == "青地白記号":
        sq = (np.abs(xx - cx) <= R_) & (np.abs(yy - cy0) <= R_)
        arrow = sq & (np.abs(xx - cx) <= 0.12 * R_) & (np.abs(yy - cy0) <= 0.7 * R_)
        comps = [(~sq, SP_M.white, sky), (sq & ~arrow, S["青塗料"], exp_scale), (arrow, S["白"], exp_scale)]
    elif kind == "緑地白字":
        sq = (np.abs(xx - cx) <= R_) & (np.abs(yy - cy0) <= R_)
        s_ = sq & (np.abs(((xx - cx) / (0.3 * R_)) % 1.0 - 0.5) <= 0.17) & (np.abs(yy - cy0) <= 0.6 * R_)
        comps = [(~sq, SP_M.white, sky), (sq & ~s_, S["緑塗料"], exp_scale), (s_, S["白"], exp_scale)]
    elif kind == "黄地黒":
        sq = (np.abs(xx - cx) <= R_) & (np.abs(yy - cy0) <= R_)
        bar = sq & (np.abs(xx - cx) <= 0.18 * R_) & (np.abs(yy - cy0) <= 0.6 * R_)
        comps = [(~sq, SP_M.white, sky), (sq & ~bar, S["黄塗料"], exp_scale), (bar, S["黒"], exp_scale)]
    lin = sy.render(comps, c, cy, SP_M, ss=SS)
    return lin, c, d_px, Wd / 2.0, margin


print("標識 | 高さ | 大きさ | z | c | 処理 | R-B: 有効, d/(βc) | R-G: 有効, d/(βc) | G-B: 有効, d/(βc)")
for kind, H_c, size in [("白地赤枠", 2100.0, 600.0), ("青地白記号", 2100.0, 600.0), ("黄地黒", 2100.0, 600.0), ("緑地白字", 5450.0, 900.0)]:
    for z in [20e3, 30e3, 50e3, 70e3]:
        lin, c, d_px, cen, margin = sign_scene(kind, z, Ra, H_c, size)
        for label, down, K, Sx, T, use420 in [("1080 444", False, 25, 83, 20, False), ("528 420", True, 25, 41, 20, True)]:
            scale = op.R_1080_TO_528 if down else 1.0
            dexp = BETA * c / scale
            outs = []
            for a, b in [("R", "B"), ("R", "G"), ("G", "B")]:
                res = []
                for seed in range(5):
                    rng = np.random.default_rng(seed + 700)
                    u = sy.to_u8(lin, noise=1.5, rng=rng)
                    if use420:
                        u = sy.chroma420(u)
                    if down:
                        u = cv2.resize(u, (int(round(u.shape[1] / scale)), int(round(u.shape[0] / scale))), interpolation=cv2.INTER_AREA)
                    dm = zncc(u, a, b, K, Sx, T)
                    H_, W_ = dm.shape
                    yy, xx = np.mgrid[0:H_, 0:W_]
                    half = (d_px / 2 + max(6, abs(c))) / scale
                    m = (np.abs(xx + 0.5 - cen / scale) <= half) & (np.abs(yy + 0.5 - cen / scale) <= half) & np.isfinite(dm)
                    x = dm[m]
                    res.append((x.size, np.median(x) if x.size else np.nan))
                res = np.array(res, dtype=float)
                med = np.nanmean(res[:, 1]) if np.isfinite(res[:, 1]).any() else np.nan
                outs.append(f"{res[:,0].mean():.0f}, {med/dexp:+.2f}")
            print(f"{kind} | {H_c/1000:.2f} | {size/1000:.1f} | {z/1000:.0f} | {c:.1f} | {label} | " + " | ".join(outs))

print()
print("=== 3. PSF の MTF50（直径 c=1 に対する周波数 [cy/c]）と等価幅 p_eq=1/(2 MTF50)／c ===")
N = 2048
c_px = 160.0
yy, xx = np.mgrid[-N // 2 : N // 2, -N // 2 : N // 2].astype(float)
disc = xx**2 + yy**2 <= (c_px / 2) ** 2
left = disc & (xx < 0)
right = disc & (xx >= 0)


def mtf50(psf, axis):
    psf = psf / psf.sum()
    F_ = np.abs(np.fft.fftshift(np.fft.fft2(psf)))
    if axis == "x":
        prof = F_[N // 2, N // 2 :]
    else:
        prof = F_[N // 2 :, N // 2]
    fr = np.arange(prof.size) / N
    i = int(np.argmax(prof < 0.5))
    f = fr[i - 1] + (prof[i - 1] - 0.5) / (prof[i - 1] - prof[i]) * (fr[i] - fr[i - 1])
    return f * c_px


cases = {
    "全円（G 理想・輝度の理想）": disc.astype(float),
    "半円（R または B）": left.astype(float),
    "G 実測合わせ Y0.7:C0.3": 0.7 * left + 0.3 * right,
    "輝度 Y=.299R+.587G+.114B（G 実測合わせ）": (0.299 + 0.587 * 0.7) * left + (0.114 + 0.587 * 0.3) * right,
    "輝度（G 半々・理想）": (0.299 + 0.587 * 0.5) * left + (0.114 + 0.587 * 0.5) * right,
}
print("PSF | MTF50 横 [cy/c] | 横の等価幅/c | MTF50 縦 | 縦の等価幅/c")
for k, v in cases.items():
    fx, fy = mtf50(v, "x"), mtf50(v, "y")
    print(f"{k} | {fx:.3f} | {1/(2*fx):.2f} | {fy:.3f} | {1/(2*fy):.2f}")

print()
print("=== 4. 実写（data/test/2300.JPG・静止画・車の背面の写真/実物は記録未確認）: 赤いレンズの周りのチャネル対 ===")
img = cv2.imread(os.path.join(ROOT, "data", "test", "2300.JPG"))
p = cv2.resize(img, (528, 352), interpolation=cv2.INTER_AREA)
b, g, r = cv2.split(p)
drb = calcDisparityByZNCC(r, b, 25, 41, 20)
drg = calcDisparityByZNCC(r, g, 25, 41, 20)
dgb = calcDisparityByZNCC(g, b, 25, 41, 20)
# 目視した位置: 左の赤いレンズはおおよそ x=30〜90, y=50〜110（540 幅基準）
for name, (x0, x1, y0, y1) in {"左の赤レンズ(上)": (28, 92, 50, 110), "右の赤レンズ(上)": (445, 515, 55, 115), "同じ行の白い車体": (110, 210, 95, 160), "ナンバー帯(同じ行帯)": (210, 330, 100, 160)}.items():
    s = 528 / 540
    X0, X1, Y0, Y1 = int(x0 * s), int(x1 * s), int(y0 * s), int(y1 * s)
    roi = lambda a: a[Y0:Y1, X0:X1]
    vals = [roi(r).mean(), roi(g).mean(), roi(b).mean()]
    out = []
    for nm, d in [("R-B", drb), ("R-G", drg), ("G-B", dgb)]:
        x = roi(d)
        x = x[np.isfinite(x)]
        out.append(f"{nm}: {x.size}px med {np.median(x) if x.size else np.nan:+.2f}")
    print(f"{name}: 平均RGB {vals[0]:.0f},{vals[1]:.0f},{vals[2]:.0f} | " + " | ".join(out))
