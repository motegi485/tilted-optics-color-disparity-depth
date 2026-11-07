"""G の位置の非対称の原因を切り分ける: 動画経路（4:2:0）と静止画（4:2:2・6000×4000）で、帯域ごとの R–G・G–B・R–B のずれを比べる。読み取りのみ。"""
import os
import sys

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
ROOT = "C:/Users/t5424001/dev/tilted_optics_color_disparity_depth"
sys.path.insert(0, ROOT)
import cv2
import numpy as np

from core.disparity import calcDisparityByZNCC

RAW = os.path.join(ROOT, "data", "calib_raw")
K, S, T = 25, 41, 20


def band_shift(a, b, fl, fh, maxlag=40):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
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


def report(bgr, label, rows):
    b, g, r = cv2.split(bgr)
    print(f"--- {label} ---")
    print("行帯 | 低域 0.02-0.08: R-B, R-G, G-B（ずれ[px]・相関）, 比 RG/RB, GB/RB | 中域 0.08-0.125 同 | 0.125-0.25: R-B ずれ,相関,lag0 | 0.25-0.5: R-B ずれ,相関,lag0")
    for (y0, y1) in rows:
        R_, G_, B_ = r[y0:y1, 40:-40], g[y0:y1, 40:-40], b[y0:y1, 40:-40]
        out = []
        for fl, fh in [(0.02, 0.08), (0.08, 0.125)]:
            rb = band_shift(R_, B_, fl, fh)
            rg = band_shift(R_, G_, fl, fh)
            gb = band_shift(G_, B_, fl, fh)
            ratio = (rg[0] / rb[0], gb[0] / rb[0]) if abs(rb[0]) > 1.0 else (np.nan, np.nan)
            out.append(f"{rb[0]:+.2f}({rb[1]:.2f}) {rg[0]:+.2f}({rg[1]:.2f}) {gb[0]:+.2f}({gb[1]:.2f}) {ratio[0]:.2f},{ratio[1]:.2f}")
        m = band_shift(R_, B_, 0.125, 0.25)
        h = band_shift(R_, B_, 0.25, 0.5001)
        print(f"{y0}-{y1} | {out[0]} | {out[1]} | {m[0]:+.2f},{m[1]:.2f},{m[2]:.2f} | {h[0]:+.2f},{h[1]:.2f},{h[2]:.2f}")


rows720 = [(y, y + 90) for y in range(0, 720, 90)]
for dist in [1400, 2000, 3200]:
    img = cv2.imread(os.path.join(RAW, f"{dist}.png"))[:, 100:1180]
    report(img, f"動画経路 {dist} mm（1080×720・4:2:0）", rows720)

path = os.path.join(ROOT, "data", "test", "2300.JPG")
img = cv2.imread(path)
print("静止画", img.shape)
down = cv2.resize(img, (1080, 720), interpolation=cv2.INTER_AREA)
report(down, "静止画 2300.JPG を 1080×720 へ INTER_AREA（元は 6000×4000・4:2:2）", rows720)
down2 = cv2.resize(img, (3000, 2000), interpolation=cv2.INTER_AREA)
report(down2, "静止画を 3000×2000（クロマの横の解像度）へ", [(y, y + 250) for y in range(0, 2000, 250)])

print()
print("=== ZNCC の比（静止画を 528×352 へ。動画経路と同じ K,S,T） ===")
p = cv2.resize(img, (528, 352), interpolation=cv2.INTER_AREA)
b, g, r = cv2.split(p)
d_rb = calcDisparityByZNCC(r, b, K, S, T)
d_rg = calcDisparityByZNCC(r, g, K, S, T)
d_gb = calcDisparityByZNCC(g, b, K, S, T)
print("行帯 | d_RB | d_RG | d_GB | RG/RB | GB/RB")
for y0 in range(12, 340, 44):
    band = slice(y0, y0 + 44)
    a, c, e = d_rb[band], d_rg[band], d_gb[band]
    if np.isfinite(a).sum() < 50:
        print(f"{y0}-{y0+43} | 画素不足")
        continue
    ma, mc, me = np.nanmedian(a), np.nanmedian(c), np.nanmedian(e)
    print(f"{y0}-{y0+43} | {ma:+.2f} | {mc:+.2f} | {me:+.2f} | {mc/ma if abs(ma)>0.3 else np.nan:.2f} | {me/ma if abs(ma)>0.3 else np.nan:.2f}")

# 画像の中身を確認するための縮小版を scratchpad に保存（目視用）
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out, exist_ok=True)
cv2.imwrite(os.path.join(out, "still_2300_small.png"), cv2.resize(img, (540, 360), interpolation=cv2.INTER_AREA))
cv2.imwrite(os.path.join(out, "calib_raw_2000_small.png"), cv2.resize(cv2.imread(os.path.join(RAW, "2000.png"))[:, 100:1180], (540, 360), interpolation=cv2.INTER_AREA))
