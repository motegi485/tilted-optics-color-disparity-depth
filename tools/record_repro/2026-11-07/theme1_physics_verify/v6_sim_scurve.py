"""実データの S 字（合焦行の近くで d の傾きが落ちる）が何で生じるかを合成で切り分ける。
実装の calcDisparityByZNCC を使う。行ごとに c を変える θ=30° 相当の画像を作り、実データと同じ解析をする。
切り分ける要素: 色の混ざり（R が C 側を、B が Y 側を通した光も受ける）、基底のぼけ（収差の代わり）、4:2:0、Bayer のデモザイク。
入力は合成のみ。
"""
import os
import sys

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "theme1_physics"))
ROOT = "C:/Users/t5424001/dev/tilted_optics_color_disparity_depth"
sys.path.insert(0, ROOT)
import cv2
import numpy as np
from scipy.signal import fftconvolve

import synth as sy
from core.disparity import calcDisparityByZNCC

BETA = 4 / (3 * np.pi)
SS = 2
H, W = 720, 1080
SLOPE = np.sin(np.deg2rad(30.0)) / 5.6  # c の行あたりの変化（px/行、スケール不変）


def scene(grain, seed=0):
    r = np.random.default_rng(seed)
    ny, nx = int(H / grain) + 4, int(W / grain) + 4
    g = (r.random((ny, nx)) > 0.5).astype(np.float32)
    return cv2.resize(g, (W * SS, H * SS), interpolation=cv2.INTER_NEAREST)


def render_rows(mask, r0, slope=SLOPE, leak=0.0, base=0.0, step=0.25):
    """行 r の c=(r-r0)·slope（下で +）の像を行ごとに作る。step ごとに全面を畳み込んで行を選ぶ。"""
    rows = np.arange(H)
    cs = (rows - r0) * slope
    levels = np.arange(np.floor(cs.min() / step) * step, cs.max() + step, step)
    Y = np.zeros((H, W), np.float32)
    C = np.zeros((H, W), np.float32)
    m = mask
    if base > 0:
        m = cv2.GaussianBlur(mask, (0, 0), base * SS)
    for lv in levels:
        sel = np.abs(cs - lv) <= step / 2 + 1e-9
        if not sel.any():
            continue
        c = float(lv)
        kY, kC = sy.half_kernels(c, abs(c) * 0.866, SS)
        a = sy.box_down(sy.conv(m, kY) * 2, SS)
        b = sy.box_down(sy.conv(m, kC) * 2, SS)
        Y[sel] = a[sel]
        C[sel] = b[sel]
    lo, hi = 0.05, 0.7
    Y = lo + (hi - lo) * Y
    C = lo + (hi - lo) * C
    R = (1 - leak) * Y + leak * C
    B = (1 - leak) * C + leak * Y
    G = 0.7 * Y + 0.3 * C
    return dict(R=R, G=G, B=B)


def bayer_demosaic(bgr):
    """BGGR の Bayer へ間引いて OpenCV の EA デモザイクで戻す（カメラ内の処理の代わり。実機の処理とは違う）"""
    h, w = bgr.shape[:2]
    raw = np.zeros((h, w), np.uint8)
    raw[0::2, 0::2] = bgr[0::2, 0::2, 0]  # B
    raw[0::2, 1::2] = bgr[0::2, 1::2, 1]  # G
    raw[1::2, 0::2] = bgr[1::2, 0::2, 1]  # G
    raw[1::2, 1::2] = bgr[1::2, 1::2, 2]  # R
    return cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR_EA)


def row_profile(d, min_n=30):
    med = np.full(d.shape[0], np.nan)
    for y in range(d.shape[0]):
        v = d[y][np.isfinite(d[y])]
        if v.size >= min_n:
            med[y] = np.median(v)
    return med


def analyse(lin, label, r0, use420=False, demosaic=False, noise=1.5, K=25, S=41, T=20, seed=1):
    bgr = sy.to_u8(lin, noise=noise, rng=np.random.default_rng(seed))
    if demosaic:
        bgr = bayer_demosaic(bgr)
    if use420:
        bgr = sy.chroma420(bgr)
    p = cv2.resize(bgr, (528, 352), interpolation=cv2.INTER_AREA)
    d = calcDisparityByZNCC(p[:, :, 2], p[:, :, 0], K, S, T)
    med = row_profile(d)
    y = np.arange(med.size)
    out = {}
    for name, sign in (("pos", +1), ("neg", -1)):
        m = np.isfinite(med) & (sign * med > 1.5) & (sign * med < 5)
        if m.sum() < 8:
            out[name] = (np.nan, np.nan)
            continue
        a, b = np.polyfit(med[m], y[m], 1)
        out[name] = (a, b)
    # d の符号は実装の定義。行が増えると d が減る向きになるように c の符号を合わせる
    gap = out["neg"][1] - out["pos"][1]
    r528 = r0 / (1080 / 528)
    # r0 の前後での傾き（d の中央値の差）
    w = 10
    i0, i1 = int(r528 - w), int(r528 + w)
    gain = np.nan
    if 0 <= i0 and i1 < med.size and np.isfinite(med[i0]) and np.isfinite(med[i1]):
        slope_near = (med[i1] - med[i0]) / (i1 - i0)
        gain = abs(slope_near) / (BETA * SLOPE)
    print(f"{label}: a_pos={out['pos'][0]:.2f}, a_neg={out['neg'][0]:.2f}（理論 {-1/(BETA*SLOPE):.2f}）, 切片差={gap:.1f} 行@528（d 換算 {abs(gap)*BETA*SLOPE:.2f} px）, 合焦行の前後 ±{w} 行の利得={gain:.2f}（d の傾き/理論）")
    return med


GRAIN = 11.0  # 校正板の粒 5.6 mm を 2 m で見た大きさの目安（1080 幅）。実データと同じではない
r0 = 340.0
mask = scene(GRAIN)
print(f"理論の行あたりの視差 = {BETA*SLOPE:.4f} px/行（528 でも同じ）。行 r0={r0}@1080 が合焦。粒 {GRAIN} px@1080。ガンマ後雑音 σ=1.5 階調")
print()
print("--- 実データ（2000 mm・528 K25）の目安: a_pos=-26.2, a_neg=-22.8, 切片差 37 行（d 換算 1.4 px）、合焦行の前後の利得 ≈0.4 ---")
print()
print("=== 1. 要素を 1 つずつ足す ===")
lin = render_rows(mask, r0)
m_ideal = analyse(lin, "理想（半円 PSF のみ）", r0)
analyse(lin, "＋4:2:0", r0, use420=True)
analyse(lin, "＋デモザイク", r0, demosaic=True)
analyse(lin, "＋デモザイク＋4:2:0", r0, demosaic=True, use420=True)
for lk in [0.1, 0.2, 0.3]:
    analyse(render_rows(mask, r0, leak=lk), f"色の混ざり {lk}", r0)
for b in [1.0, 2.0, 4.0, 8.0]:
    analyse(render_rows(mask, r0, base=b), f"基底ぼけ σ={b} px@1080", r0)
for lk, b in [(0.2, 2.0), (0.2, 4.0), (0.3, 4.0), (0.3, 8.0)]:
    analyse(render_rows(mask, r0, leak=lk, base=b), f"混ざり {lk}＋基底 σ={b}", r0)
print()
print("=== 2. 窓の中で d が変わる効果（θ の効果）: 行の傾きを変えて比べる（要素なし／混ざり 0.2＋基底 4） ===")
for th in [10.0, 19.11, 30.0, 40.0]:
    sl = np.sin(np.deg2rad(th)) / 5.6
    r0_ = 360.0
    for lk, b in [(0.0, 0.0), (0.2, 4.0)]:
        lin = render_rows(mask, r0_, slope=sl, leak=lk, base=b)
        # analyse は SLOPE で利得を出すので、ここだけ別に計算する
        bgr = sy.to_u8(lin, noise=1.5, rng=np.random.default_rng(1))
        p = cv2.resize(bgr, (528, 352), interpolation=cv2.INTER_AREA)
        d = calcDisparityByZNCC(p[:, :, 2], p[:, :, 0], 25, 41, 20)
        med = row_profile(d)
        y = np.arange(med.size)
        res = []
        for sign in (+1, -1):
            mm = np.isfinite(med) & (sign * med > 1.5) & (sign * med < 5)
            res.append(np.polyfit(med[mm], y[mm], 1) if mm.sum() >= 8 else (np.nan, np.nan))
        rc = r0_ / (1080 / 528)
        w = 10
        gain = abs(med[int(rc + w)] - med[int(rc - w)]) / (2 * w) / (BETA * sl)
        gap = res[1][1] - res[0][1]
        print(f"θ={th}（{BETA*sl:.4f} px/行）混ざり{lk} 基底{b}: a_pos={res[0][0]:.1f}, a_neg={res[1][0]:.1f}, 切片差={gap:.1f} 行, d 換算 {abs(gap)*BETA*sl:.2f} px, 利得={gain:.2f}")
print()
print("=== 3. 粒の大きさを変える（理想／混ざり 0.2＋基底 4） ===")
for g in [5.0, 8.0, 15.0, 30.0]:
    ms = scene(g, seed=3)
    analyse(render_rows(ms, r0), f"粒 {g} 理想", r0)
    analyse(render_rows(ms, r0, leak=0.2, base=4.0), f"粒 {g} 混ざり 0.2＋基底 4", r0)
print()
print("=== 4. 処理解像度を上げたとき平坦域の幅が変わるか（混ざり 0.2＋基底 4・1080 幅で K=51 S=83）。実データは幅がほぼ変わらない ===")
lin = render_rows(mask, r0, leak=0.2, base=4.0)
bgr = sy.to_u8(lin, noise=1.5, rng=np.random.default_rng(1))
d10 = calcDisparityByZNCC(bgr[:, :, 2], bgr[:, :, 0], 51, 83, 20)
med = row_profile(d10)
y = np.arange(med.size)
sc = 1080 / 528
res = []
for sign in (+1, -1):
    mm = np.isfinite(med) & (sign * med > 1.5 * sc) & (sign * med < 5 * sc)
    res.append(np.polyfit(med[mm], y[mm], 1) if mm.sum() >= 8 else (np.nan, np.nan))
print(f"1080 K51: a_pos={res[0][0]:.1f}, a_neg={res[1][0]:.1f}, 切片差={(res[1][1]-res[0][1])/sc:.1f} 行@528 換算")
