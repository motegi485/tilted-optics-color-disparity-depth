"""実データ（読み取りのみ）で確かめる:
 1. R–G / G–B / R–B の視差の比（G は開口の中央か）
 2. R・G・B の横方向のぼけ幅の比（R,B は半円、G は全円か）
 3. 4:2:0 の影響: Y・Cb・Cr の横方向スペクトル、高周波で R と B が同じ成分を共有しているか
 4. 静止画（data/test/2300.JPG）の画素数とクロマの間引き
入力: data/calib_raw/*.png（ノイズ板・1280×720・32枚平均）。ファイルは変更しない。
"""
import os
import struct
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


def load_crop(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    assert img is not None and img.shape[:2] == (720, 1280), (path, None if img is None else img.shape)
    return img[:, 100:1180]


def to528(crop):
    return cv2.resize(crop, (528, 352), interpolation=cv2.INTER_AREA)


print("=== 1. チャネル対の視差（528×352・K=25・S=41・T=20・ノイズ板） ===")
print("距離 | 画素数 | 傾き d_RG/d_RB, d_GB/d_RB (pos枝) | (neg枝) | 閉じ d_RG+d_GB-d_RB の中央値/IQR")
for dist in [1000, 1400, 2000, 2600, 3200, 4000]:
    crop = load_crop(os.path.join(RAW, f"{dist}.png"))
    p = to528(crop)
    b, g, r = cv2.split(p)
    d_rb = calcDisparityByZNCC(r, b, K, S, T)
    d_rg = calcDisparityByZNCC(r, g, K, S, T)
    d_gb = calcDisparityByZNCC(g, b, K, S, T)
    m = np.isfinite(d_rb) & np.isfinite(d_rg) & np.isfinite(d_gb)
    out = []
    for sgn in (1, -1):
        mm = m & (sgn * d_rb > 0.75) & (sgn * d_rb < 6)
        x = d_rb[mm]
        if x.size < 100:
            out.append("n<100")
            continue
        s_rg = np.sum(x * d_rg[mm]) / np.sum(x * x)
        s_gb = np.sum(x * d_gb[mm]) / np.sum(x * x)
        # 中央値の比（外れ値に頑健）
        rg_med = np.median(d_rg[mm] / x)
        gb_med = np.median(d_gb[mm] / x)
        out.append(f"LS {s_rg:.3f},{s_gb:.3f} med {rg_med:.3f},{gb_med:.3f} (n={x.size})")
    clo = d_rg[m] + d_gb[m] - d_rb[m]
    print(f"{dist} | {int(m.sum())} | {out[0]} | {out[1]} | {np.median(clo):+.3f} / {np.subtract(*np.percentile(clo,[75,25])):.3f}")

print()
print("=== 1b. 行で束ねた比: 行ごとの d の中央値（2000 mm・行帯 22 行刻み） ===")
crop = load_crop(os.path.join(RAW, "2000.png"))
p = to528(crop)
b, g, r = cv2.split(p)
d_rb = calcDisparityByZNCC(r, b, K, S, T)
d_rg = calcDisparityByZNCC(r, g, K, S, T)
d_gb = calcDisparityByZNCC(g, b, K, S, T)
print("行帯 | d_RB | d_RG | d_GB | RG/RB | GB/RB")
for y0 in range(12, 340, 44):
    band = slice(y0, y0 + 44)
    a, c, e = d_rb[band], d_rg[band], d_gb[band]
    if np.isfinite(a).sum() < 50:
        continue
    ma, mc, me = np.nanmedian(a), np.nanmedian(c), np.nanmedian(e)
    print(f"{y0}-{y0+43} | {ma:+.2f} | {mc:+.2f} | {me:+.2f} | {mc/ma if abs(ma)>0.3 else np.nan:.2f} | {me/ma if abs(ma)>0.3 else np.nan:.2f}")


def autocorr_width(ch, y0, y1, xmargin=60, maxlag=40):
    """横方向の自己相関（平均除去）の半値半幅を、行帯で求める"""
    a = ch[y0:y1, xmargin:-xmargin].astype(np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    n = a.shape[1]
    F_ = np.fft.rfft(a, n=2 * n, axis=1)
    ac = np.fft.irfft(np.abs(F_) ** 2, axis=1).mean(axis=0)[: maxlag + 1]
    ac = ac / ac[0]
    i = np.argmax(ac < 0.5)
    if i == 0:
        return np.nan
    return (i - 1) + (ac[i - 1] - 0.5) / (ac[i - 1] - ac[i])


print()
print("=== 2. 横方向のぼけ幅（自己相関の半値半幅 px@528・ノイズ板）。R,B は半円で幅が約半分のはず ===")
print("距離 行帯 | d_RB中央値 | R | G | B | G/R | G/B | 縦方向の幅 R,G,B")
def autocorr_width_v(ch, x0, x1, ymargin=4, maxlag=40, y0=0, y1=None):
    a = ch[y0:y1, x0:x1].astype(np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    n = a.shape[0]
    F_ = np.fft.rfft(a, n=2 * n, axis=0)
    ac = np.fft.irfft(np.abs(F_) ** 2, axis=0).mean(axis=1)[: maxlag + 1]
    ac = ac / ac[0]
    i = np.argmax(ac < 0.5)
    if i == 0:
        return np.nan
    return (i - 1) + (ac[i - 1] - 0.5) / (ac[i - 1] - ac[i])
for dist in [1400, 2000, 3200]:
    p = to528(load_crop(os.path.join(RAW, f"{dist}.png")))
    b, g, r = cv2.split(p)
    d = calcDisparityByZNCC(r, b, K, S, T)
    for y0 in range(0, 352 - 32, 64):
        y1 = y0 + 32
        wr, wg, wb = autocorr_width(r, y0, y1), autocorr_width(g, y0, y1), autocorr_width(b, y0, y1)
        vr, vg, vb = [autocorr_width_v(c, 60, 468, y0=y0, y1=y1) for c in (r, g, b)]
        print(f"{dist} {y0}-{y1} | {np.nanmedian(d[y0:y1]):+.2f} | {wr:.2f} | {wg:.2f} | {wb:.2f} | {wg/wr:.2f} | {wg/wb:.2f} | {vr:.2f},{vg:.2f},{vb:.2f}")

print()
print("=== 3. 4:2:0: 1080 幅での横方向パワー（2000 mm と 3200 mm・行帯ごと・相対値） ===")
print("cv2 の YCrCb へ変換して横方向の FFT を取る。band: [0-0.125],[0.125-0.25],[0.25-0.5] cy/px")
for dist in [2000, 3200]:
    crop = load_crop(os.path.join(RAW, f"{dist}.png"))
    ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    for (y0, y1) in [(0, 80), (280, 360), (640, 720)]:
        s = []
        for idx, nm in [(0, "Y"), (1, "Cr"), (2, "Cb")]:
            a = ycc[y0:y1, 40:-40, idx]
            a = a - a.mean(axis=1, keepdims=True)
            Pw = (np.abs(np.fft.rfft(a, axis=1)) ** 2).mean(axis=0)
            fr = np.fft.rfftfreq(a.shape[1])
            e = [Pw[(fr >= l) & (fr < h)].sum() for (l, h) in [(0.0, 0.125), (0.125, 0.25), (0.25, 0.5001)]]
            tot = sum(e)
            s.append(f"{nm}: {e[0]/tot:.3f},{e[1]/tot:.3f},{e[2]/tot:.4f} (total {tot:.2e})")
        print(f"{dist} 行{y0}-{y1}: " + " | ".join(s))


def band_shift(a, b, fl, fh, maxlag=30):
    """行ごとに帯域制限した a,b の横方向相互相関を平均し、ピーク位置と相関値を返す。

    a を基準に b をずらしたとき a(x) ~ b(x - s) となる s>0 ⇔ b が左。
    """
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
    ea = (np.abs(Fa) ** 2).sum(axis=1).mean() / (2 * n)
    eb = (np.abs(Fb) ** 2).sum(axis=1).mean() / (2 * n)
    xc = np.concatenate([xc[-maxlag:], xc[: maxlag + 1]]) / np.sqrt(ea * eb)
    lag = np.arange(-maxlag, maxlag + 1)
    i = int(np.argmax(xc))
    sub = 0.0
    if 0 < i < len(xc) - 1:
        den = xc[i - 1] - 2 * xc[i] + xc[i + 1]
        if abs(den) > 1e-12:
            sub = 0.5 * (xc[i - 1] - xc[i + 1]) / den
    i0 = maxlag
    return lag[i] + sub, xc[i], xc[i0]


print()
print("=== 3b. 帯域ごとの R–B の横ずれ（1080 幅の行帯平均）: 高周波で 0 に張り付くなら R と B は Y を共有している ===")
print("距離 行帯 | d_RB@528→1080換算 | 低域(0.02-0.125): ずれ,相関 | 中域(0.125-0.25): ずれ,相関,lag0相関 | 高域(0.25-0.5): ずれ,相関,lag0相関")
for dist in [1400, 2000, 3200]:
    crop = load_crop(os.path.join(RAW, f"{dist}.png"))
    p = to528(crop)
    b5, _, r5 = cv2.split(p)
    d5 = calcDisparityByZNCC(r5, b5, K, S, T)
    b, g, r = cv2.split(crop)
    for y0_ in [0, 90, 180, 270, 360, 450, 540, 630]:
        y1_ = y0_ + 90
        d_ref = np.nanmedian(d5[int(y0_ / 2.045) : int(y1_ / 2.045)]) * 2.045
        lo = band_shift(r[y0_:y1_, 40:-40], b[y0_:y1_, 40:-40], 0.02, 0.125)
        mi = band_shift(r[y0_:y1_, 40:-40], b[y0_:y1_, 40:-40], 0.125, 0.25)
        hi = band_shift(r[y0_:y1_, 40:-40], b[y0_:y1_, 40:-40], 0.25, 0.5001)
        print(f"{dist} {y0_}-{y1_} | {d_ref:+.2f} | {lo[0]:+.2f},{lo[1]:.3f} | {mi[0]:+.2f},{mi[1]:.3f},{mi[2]:.3f} | {hi[0]:+.2f},{hi[1]:.3f},{hi[2]:.3f}")

print()
print("=== 3c. 実際の ZNCC を 1080 幅で（K=51,S=83,T≈10）と 528 幅で（K=25,S=41,T=20）比べる（中央値は 528 換算） ===")
dist = 2000
crop = load_crop(os.path.join(RAW, f"{dist}.png"))
b, g, r = cv2.split(crop)
p = to528(crop)
b5, _, r5 = cv2.split(p)
d5 = calcDisparityByZNCC(r5, b5, K, S, T)
sm = cv2.GaussianBlur
d10 = calcDisparityByZNCC(r, b, 51, 83, 10)
d10s = calcDisparityByZNCC(r, b, 25, 83, 10)
print("行帯(720基準) | d@528 | d@1080(K51)/2.045 | d@1080(K25)/2.045 | |d|<0.45@528 換算の画素割合 528, 1080K51, 1080K25")
for y0_ in range(0, 720, 90):
    y1_ = y0_ + 90
    a = d5[int(y0_ / 2.045) : int(y1_ / 2.045)]
    c = d10[y0_:y1_] / 2.045
    e = d10s[y0_:y1_] / 2.045
    fr = [np.mean(np.abs(x[np.isfinite(x)]) < 0.45) for x in (a, c, e)]
    print(f"{y0_}-{y1_} | {np.nanmedian(a):+.2f} | {np.nanmedian(c):+.2f} | {np.nanmedian(e):+.2f} | {fr[0]:.3f},{fr[1]:.3f},{fr[2]:.3f}")

print()
print("=== 4. 静止画 data/test/2300.JPG ===")
path = os.path.join(ROOT, "data", "test", "2300.JPG")
with open(path, "rb") as fh:
    buf = fh.read()
i = 2
found = False
while i < len(buf) - 9:
    if buf[i] != 0xFF:
        i += 1
        continue
    mk = buf[i + 1]
    if mk in (0xC0, 0xC1, 0xC2):
        L = struct.unpack(">H", buf[i + 2 : i + 4])[0]
        P, H, W, Nf = buf[i + 4], *struct.unpack(">HH", buf[i + 5 : i + 9]), buf[i + 9]
        comps = [(buf[i + 10 + 3 * j], buf[i + 11 + 3 * j] >> 4, buf[i + 11 + 3 * j] & 15) for j in range(Nf)]
        print(f"SOF{mk-0xC0}: {W}×{H}, comps(id,H,V)={comps}")
        found = True
        # 埋め込みサムネイルの SOF を拾わないよう、最後まで走査して最大のものを採る
    if mk == 0xD9 or mk == 0x00 or (0xD0 <= mk <= 0xD7):
        i += 2
    elif mk == 0xDA:
        L = struct.unpack(">H", buf[i + 2 : i + 4])[0]
        i += 2 + L
        # エントロピー符号データを飛ばす
        while i < len(buf) - 1:
            if buf[i] == 0xFF and buf[i + 1] not in (0x00,) and not (0xD0 <= buf[i + 1] <= 0xD7):
                break
            i += 1
    elif mk == 0xE1:
        L = struct.unpack(">H", buf[i + 2 : i + 4])[0]
        i += 2 + L  # EXIF は飛ばす（埋め込みサムネイルを読まない）
    else:
        L = struct.unpack(">H", buf[i + 2 : i + 4])[0]
        i += 2 + L
if not found:
    print("SOF が見つからなかった")
