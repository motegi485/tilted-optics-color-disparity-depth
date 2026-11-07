"""v8 の合成の幅を 1080 に直して、クロマの低域通過で実データの S 字と帯域ごとのずれが再現するかを見る。
処理は実データと同じ 1080×720 → 528×352（INTER_AREA）→ ZNCC(K25,S41,T20)。1080 幅（K51,S83）でも比べる。
θ を変えたとき（行の傾きが変わるとき）に平坦域が「視差の幅」で一定か「行の幅」で一定かも見る。入力は合成のみ。
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

import synth as sy
from core.disparity import calcDisparityByZNCC

BETA = 4 / (3 * np.pi)
SS = 2
H, W = 720, 1080
BANDS = [(0.02, 0.0625), (0.0625, 0.125), (0.125, 0.1875), (0.1875, 0.25), (0.25, 0.375), (0.375, 0.5001)]


def xcorr_band(a, b, fl, fh, maxlag=30):
    a = a.astype(np.float64) - a.mean(axis=1, keepdims=True)
    b = b.astype(np.float64) - b.mean(axis=1, keepdims=True)
    n = a.shape[1]
    fr = np.fft.rfftfreq(2 * n)
    mask = (fr >= fl) & (fr < fh)
    Fa = np.fft.rfft(a, n=2 * n, axis=1) * mask
    Fb = np.fft.rfft(b, n=2 * n, axis=1) * mask
    xc = np.fft.irfft(Fa * np.conj(Fb), axis=1).mean(axis=0)
    aa = np.fft.irfft(np.abs(Fa) ** 2, axis=1).mean(axis=0)[0]
    bb = np.fft.irfft(np.abs(Fb) ** 2, axis=1).mean(axis=0)[0]
    xc = np.concatenate([xc[-maxlag:], xc[: maxlag + 1]]) / np.sqrt(aa * bb)
    lag = np.arange(-maxlag, maxlag + 1)
    i = int(np.argmax(xc))
    sub = 0.0
    if 0 < i < len(xc) - 1:
        den = xc[i - 1] - 2 * xc[i] + xc[i + 1]
        if abs(den) > 1e-12:
            sub = 0.5 * (xc[i - 1] - xc[i + 1]) / den
    return lag[i] + sub, xc[i]


def scene(grain, seed=0):
    r = np.random.default_rng(seed)
    ny, nx = int(H / grain) + 4, int(W / grain) + 4
    g = (r.random((ny, nx)) > 0.5).astype(np.float32)
    return cv2.resize(g, (W * SS, H * SS), interpolation=cv2.INTER_NEAREST)


def render_rows(mask, r0, slope, step=0.5):
    rows = np.arange(H)
    cs = (rows - r0) * slope
    levels = np.arange(np.floor(cs.min() / step) * step, cs.max() + step, step)
    Y = np.zeros((H, W), np.float32)
    C = np.zeros((H, W), np.float32)
    for lv in levels:
        sel = np.abs(cs - lv) <= step / 2 + 1e-9
        if not sel.any():
            continue
        kY, kC = sy.half_kernels(float(lv), abs(float(lv)) * 0.866, SS)
        Y[sel] = sy.box_down(sy.conv(mask, kY) * 2, SS)[sel]
        C[sel] = sy.box_down(sy.conv(mask, kC) * 2, SS)[sel]
    lo, hi = 0.05, 0.7
    Y = lo + (hi - lo) * Y
    C = lo + (hi - lo) * C
    return dict(R=Y, G=0.7 * Y + 0.3 * C, B=C)


def chroma_lp(bgr, sigma):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    if sigma > 0:
        ycc[:, :, 1] = cv2.GaussianBlur(ycc[:, :, 1], (0, 0), sigma)
        ycc[:, :, 2] = cv2.GaussianBlur(ycc[:, :, 2], (0, 0), sigma)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def row_profile(d, min_n=30):
    med = np.full(d.shape[0], np.nan)
    for y in range(d.shape[0]):
        v = d[y][np.isfinite(d[y])]
        if v.size >= min_n:
            med[y] = np.median(v)
    return med


def analyse_d(d, label, r0_528, slope, scale=1.0, w=15):
    med = row_profile(d)
    y = np.arange(med.size)
    res = {}
    for name, s in (("pos", 1), ("neg", -1)):
        m = np.isfinite(med) & (s * med > 1.5 * scale) & (s * med < 5 * scale)
        res[name] = np.polyfit(med[m], y[m], 1) if m.sum() >= 8 else (np.nan, np.nan)
    gap = abs(res["neg"][1] - res["pos"][1]) / scale
    rc = r0_528 * scale
    W_ = int(w * scale)
    gain = abs(med[int(rc + W_)] - med[int(rc - W_)]) / (2 * W_) / (BETA * slope)
    print(f"{label}: a_pos={res['pos'][0]:.1f}, a_neg={res['neg'][0]:.1f}（理論 ±{1/(BETA*slope):.1f}）, 切片差={gap:.1f} 行@528（d 換算 {gap*BETA*slope:.2f} px@528）, 合焦行 ±{w} 行の利得={gain:.2f}")
    return med


slope30 = 0.5 / 5.6
r0 = 340.0
r0_528 = r0 * 528 / 1080
mask = scene(11.0)
lin = render_rows(mask, r0, slope30)
base = sy.to_u8(lin, noise=1.5, rng=np.random.default_rng(1))
print("実データの目安（2000 mm・528 K25）: a_pos −26.2、a_neg −22.8、切片差 37 行（d 換算 1.4〜1.6 px）。行 147→187 の傾きから利得 ≈0.41")
print("=== 1. θ=30° 相当でクロマの低域通過を振る（粒 11 px@1080） ===")
for sg in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    b = chroma_lp(base, sg)
    p = cv2.resize(b, (528, 352), interpolation=cv2.INTER_AREA)
    d = calcDisparityByZNCC(p[:, :, 2], p[:, :, 0], 25, 41, 20)
    analyse_d(d, f"σ={sg} 528 K25", r0_528, slope30)
print()
print("=== 2. 同じ入力を 1080 幅のまま（K51,S83）。実データは切片差が 528 と同程度（33〜54 行@528 換算） ===")
for sg in [0.0, 2.0, 3.0]:
    b = chroma_lp(base, sg)
    d = calcDisparityByZNCC(b[:, :, 2], b[:, :, 0], 51, 83, 20)
    analyse_d(d, f"σ={sg} 1080 K51", r0_528, slope30, scale=1080 / 528)
print()
print("=== 3. 帯域ごとのずれ（合成・σ=2.5）。実データは 0.125 cy/px@1080 より上でずれが 0、相関が高い ===")
b = chroma_lp(base, 2.5)
bb, _, rr = cv2.split(b)
print("行帯 | c[px] | βc | " + " | ".join(f"{a:.3f}-{c:.3f}" for a, c in BANDS))
for y0 in range(0, 720, 120):
    cmid = (y0 + 60 - r0) * slope30
    out = [("%+.1f(%.2f)" % xcorr_band(rr[y0 : y0 + 120, 40:-40], bb[y0 : y0 + 120, 40:-40], fl, fh)) for fl, fh in BANDS]
    print(f"{y0} | {cmid:+.1f} | {BETA*cmid:+.1f} | " + " | ".join(out))
print()
print("=== 4. θ を変えて平坦域の幅を見る（σ=2.5。528 K25）。視差の幅が一定なら、θ が小さいほど行の幅が広がる ===")
for th in [4.0, 10.0, 19.11, 30.0]:
    sl = np.sin(np.deg2rad(th)) / 5.6
    # 行あたりの c が小さいので、全行で帯に届かない θ では切片差を出せない。d の中央値と理論を表で示す
    lin_ = render_rows(mask, 360.0, sl)
    b_ = chroma_lp(sy.to_u8(lin_, noise=1.5, rng=np.random.default_rng(1)), 2.5)
    p = cv2.resize(b_, (528, 352), interpolation=cv2.INTER_AREA)
    d = calcDisparityByZNCC(p[:, :, 2], p[:, :, 0], 25, 41, 20)
    med = row_profile(d)
    rows = [30, 60, 90, 120, 150, 176, 202, 232, 262, 292, 322]
    # 理論の d（528）: 行は 528 換算。c は 1080 の px で (r1080-360)*sl、d@528=βc/2.045
    print(f"θ={th}: 行@528 | 理論 d@528 | 測定 d の中央値")
    print("   " + " ".join(f"{r}:{BETA*((r*1080/528-360)*sl)/(1080/528):+.2f}/{med[r]:+.2f}" for r in rows))
