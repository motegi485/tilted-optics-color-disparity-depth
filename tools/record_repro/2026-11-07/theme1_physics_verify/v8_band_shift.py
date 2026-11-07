"""S 字の候補「輝度の高周波を R,B が共有する（クロマの帯域が狭い）」を確かめる。
1. 実データで、帯域ごとの R-B のずれ（相互相関の山）を比べる。高い帯域だけ 0 に張り付くなら共有。
2. 合成で、クロマに低域通過を掛けたときの切片差と利得を測り、実データの値と比べる。
入力: data/calib_raw/*.png、data/test/2300.JPG（読み取りのみ）と合成。
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

RAW = os.path.join(ROOT, "data", "calib_raw")
BETA = 4 / (3 * np.pi)
SS = 2
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
    return lag[i] + sub, xc[i], xc[maxlag]


print("=== 1. 実データ: 帯域ごとの R-B のずれ[px@1080]（相関）。最初の行帯は上が pos、下が neg ===")
print("画像 行帯 | " + " | ".join(f"{a:.3f}-{b:.3f}" for a, b in BANDS))
for label, img in [(f"{d}", cv2.imread(os.path.join(RAW, f"{d}.png"))[:, 100:1180]) for d in (1400, 2000, 3200)] + [("静止画2300", cv2.resize(cv2.imread(os.path.join(ROOT, "data", "test", "2300.JPG")), (1080, 720), interpolation=cv2.INTER_AREA))]:
    b, _, r = cv2.split(img)
    for y0 in range(0, 720, 120):
        out = []
        for fl, fh in BANDS:
            s, c, c0 = xcorr_band(r[y0 : y0 + 120, 40:-40], b[y0 : y0 + 120, 40:-40], fl, fh)
            out.append(f"{s:+.1f}({c:.2f})")
        print(f"{label} {y0} | " + " | ".join(out))


def scene(H, W, grain, seed=0):
    r = np.random.default_rng(seed)
    ny, nx = int(H / grain) + 4, int(W / grain) + 4
    g = (r.random((ny, nx)) > 0.5).astype(np.float32)
    return cv2.resize(g, (W * SS, H * SS), interpolation=cv2.INTER_NEAREST)


def render_rows(mask, H, W, r0, slope, step=0.5):
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


def analyse_img(bgr, label, r0, slope):
    p = cv2.resize(bgr, (528, 352), interpolation=cv2.INTER_AREA)
    d = calcDisparityByZNCC(p[:, :, 2], p[:, :, 0], 25, 41, 20)
    med = row_profile(d)
    y = np.arange(med.size)
    res = {}
    for name, s in (("pos", 1), ("neg", -1)):
        m = np.isfinite(med) & (s * med > 1.5) & (s * med < 5)
        res[name] = np.polyfit(med[m], y[m], 1) if m.sum() >= 8 else (np.nan, np.nan)
    gap = res["neg"][1] - res["pos"][1]
    rc = r0 / (1080 / 528)
    w = 15
    gain = abs(med[int(rc + w)] - med[int(rc - w)]) / (2 * w) / (BETA * slope)
    print(f"{label}: a_pos={res['pos'][0]:.1f}, a_neg={res['neg'][0]:.1f}, |切片差|={abs(gap):.1f} 行@528（d 換算 {abs(gap)*BETA*slope:.2f} px）, 合焦行 ±{w} 行の利得={gain:.2f}")


print()
print("=== 2. 合成: クロマの低域通過の強さを振る（θ=30° 相当の行の傾き、1080×720 で作って 528 で ZNCC。粒 11 px@1080） ===")
print("実データ: |切片差| 37〜49 行（d 換算 1.4〜1.9 px）、±15 行の利得 約 0.45（行 147→187 で +0.32→−0.30 の傾きから）")
H, W = 720, 800
slope = 0.5 / 5.6
r0 = 340.0
mask = scene(H, W, 11.0)
lin = render_rows(mask, H, W, r0, slope)
base = sy.to_u8(lin, noise=1.5, rng=np.random.default_rng(1))
for sg in [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
    analyse_img(chroma_lp(base, sg), f"クロマ低域通過 σ={sg} px@1080", r0, slope)
print()
print("=== 3. 同じクロマ低域通過を粒 22 px でも（実データの近距離に近い） ===")
mask2 = scene(H, W, 22.0, seed=2)
lin2 = render_rows(mask2, H, W, r0, slope)
base2 = sy.to_u8(lin2, noise=1.5, rng=np.random.default_rng(1))
for sg in [0.0, 2.0, 4.0, 6.0, 8.0]:
    analyse_img(chroma_lp(base2, sg), f"粒 22 クロマ低域通過 σ={sg} px@1080", r0, slope)
