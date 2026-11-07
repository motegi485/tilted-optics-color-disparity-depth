"""実データ（読み取りのみ）で、合焦行の近くの視差の振る舞いと、|d| ごとの視差のばらつきを調べる。
入力: data/calib_raw/*.png（ノイズ板・1280×720・32 枚平均）、data/test/2300.JPG（静止画）。ファイルは変更しない。

1. 行ごとの d の中央値を、528 幅と 1080 幅で求め、pos/neg の直線の切片の差（平坦域の幅）を比べる。
   平坦域が同じ「行数（528 換算）」なら光学または色の帯域の原因、半分になるなら画素領域（pixel-locking）の原因が疑われる。
2. |d| の区間ごとに、行の直線からの残差のばらつき（MAD）を枝別に出す（同じ距離で pos と neg を比べる）。
"""
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
SC = 1080.0 / 528.0


def row_profile(d, min_n=30):
    """行ごとの中央値（NaN 除外）。"""
    H = d.shape[0]
    med = np.full(H, np.nan)
    n = np.zeros(H, dtype=int)
    for y in range(H):
        v = d[y][np.isfinite(d[y])]
        n[y] = v.size
        if v.size >= min_n:
            med[y] = np.median(v)
    return med, n


def fit_branch(med, lo, hi, sign):
    y = np.arange(med.size)
    m = np.isfinite(med) & (sign * med > lo) & (sign * med < hi)
    if m.sum() < 8:
        return None
    a, b = np.polyfit(med[m], y[m], 1)  # y = a d + b
    return a, b, int(m.sum())


def analyse(d, scale, label):
    """scale: d と行の単位を 528 換算する倍率（1 または SC）"""
    med, n = row_profile(d)
    lo, hi = 1.5 * scale, 5.0 * scale  # 528 換算で 1.5〜5 px の範囲で直線を当てる
    fp = fit_branch(med, lo, hi, +1)
    fn = fit_branch(med, lo, hi, -1)
    if fp is None or fn is None:
        return f"{label}: 枝が足りない pos={fp} neg={fn}", None
    ap, bp, np_ = fp
    an, bn, nn = fn
    gap_rows = (bn - bp) / scale  # 528 換算の行
    y = np.arange(med.size)
    # 切片の間の行で、測定 d の中央値がどれだけ 0 に貼りつくか
    i0, i1 = int(min(bp, bn)), int(max(bp, bn))
    i0c, i1c = max(i0, 0), min(i1, med.size - 1)
    inner = med[i0c : i1c + 1]
    inner = inner[np.isfinite(inner)]
    flat = np.nan if inner.size == 0 else np.max(np.abs(inner)) / scale
    frac_small = np.nan if inner.size == 0 else np.mean(np.abs(inner) < 0.25 * scale)
    # 2 枝の直線を別々に当てたときの傾き（528 換算で行/px は scale に依らないはず）
    return (
        f"{label}: a_pos={ap:.2f}, a_neg={an:.2f}（行/px）, 切片 pos={bp/scale:.1f}, neg={bn/scale:.1f}（528 換算の行）, "
        f"差={gap_rows:.1f} 行（視差換算 {gap_rows/abs(ap):.2f}/{gap_rows/abs(an):.2f} px@528）, 切片間の |d| 中央値の最大={flat:.2f} px@528, |d|<0.25 の行の割合={frac_small:.2f}",
        (ap, bp, an, bn, med),
    )


print("=== 1. 切片の差（平坦域）を 528 幅と 1080 幅で比べる（ノイズ板・動画経路 4:2:0） ===")
store = {}
for dist in [1400, 1600, 2000, 2400, 2800, 3200]:
    img = cv2.imread(os.path.join(RAW, f"{dist}.png"))[:, 100:1180]
    p = cv2.resize(img, (528, 352), interpolation=cv2.INTER_AREA)
    b5, _, r5 = cv2.split(p)
    d5 = calcDisparityByZNCC(r5, b5, K, S, T)
    b, _, r = cv2.split(img)
    d10 = calcDisparityByZNCC(r, b, 51, 83, 20)
    d10s = calcDisparityByZNCC(r, b, 25, 83, 20)
    s5, r5_ = analyse(d5, 1.0, f"{dist} 528 K25")
    s10, r10_ = analyse(d10, SC, f"{dist} 1080 K51")
    s10s, r10s_ = analyse(d10s, SC, f"{dist} 1080 K25")
    print(s5)
    print(s10)
    print(s10s)
    store[dist] = (d5, d10)

print()
print("=== 2. 静止画 2300.JPG（6000×4000・4:2:2。縮小でクロマの帯域は足りる）で同じことをする ===")
path = os.path.join(ROOT, "data", "test", "2300.JPG")
still = cv2.imread(path)
print("still shape", still.shape)
for (w, h, k, s, sc) in [(528, 352, 25, 41, 1.0), (1080, 720, 51, 83, SC), (1080, 720, 25, 83, SC), (2160, 1440, 51, 167, 2 * SC)]:
    pp = cv2.resize(still, (w, h), interpolation=cv2.INTER_AREA)
    bb, _, rr = cv2.split(pp)
    dd = calcDisparityByZNCC(rr, bb, k, s, 20)
    st, _ = analyse(dd, sc, f"静止画 {w} K{k}")
    print(st)

print()
print("=== 3. 行の中央値（2000 mm・528 換算。y0 の前後 60 行）。S 字の形を見る ===")
d5, d10 = store[2000]
m5, n5 = row_profile(d5)
m10, n10 = row_profile(d10)
fp = fit_branch(m5, 1.5, 5, 1)
fn = fit_branch(m5, 1.5, 5, -1)
print(f"pos 直線 y={fp[0]:.2f} d + {fp[1]:.1f}; neg 直線 y={fn[0]:.2f} d + {fn[1]:.1f}")
print("行@528 | d@528 中央値 | d@1080/2.045 中央値 | pos 直線の d | neg 直線の d")
for y in range(int(fp[1]) - 30, int(fn[1]) + 31, 5):
    if 0 <= y < 352:
        y10 = int(round(y * SC))
        v10 = np.nanmedian(m10[max(y10 - 2, 0) : y10 + 3]) / SC
        print(f"{y} | {m5[y]:+.3f} | {v10:+.3f} | {(y-fp[1])/fp[0]:+.3f} | {(y-fn[1])/fn[0]:+.3f}")

print()
print("=== 4. |d| の区間ごとの視差のばらつき: 行の中央値からの残差（行内の散らばり）の MAD×1.4826。枝は符号で分ける（528・K25） ===")
print("距離 | 枝 | " + " | ".join(["0-0.45", "0.45-1", "1-2", "2-3", "3-4", "4-5", "5-7", "7-10", "10-14.5"]))
bins = [0, 0.45, 1, 2, 3, 4, 5, 7, 10, 14.5]
agg = {+1: [[] for _ in bins[:-1]], -1: [[] for _ in bins[:-1]]}
for dist in [1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3600, 4000]:
    img = cv2.imread(os.path.join(RAW, f"{dist}.png"))[:, 100:1180]
    p = cv2.resize(img, (528, 352), interpolation=cv2.INTER_AREA)
    b5, _, r5 = cv2.split(p)
    d = calcDisparityByZNCC(r5, b5, K, S, T)
    med, _ = row_profile(d)
    res = d - med[:, None]
    for sgn in (+1, -1):
        out = []
        for j in range(len(bins) - 1):
            m = np.isfinite(d) & np.isfinite(res) & (sgn * med[:, None] >= bins[j]) & (sgn * med[:, None] < bins[j + 1])
            if m.sum() < 200:
                out.append("-")
                continue
            r_ = res[m]
            mad = 1.4826 * np.median(np.abs(r_ - np.median(r_)))
            out.append(f"{mad:.2f}")
            agg[sgn][j].append(mad)
        print(f"{dist} | {'pos' if sgn>0 else 'neg'} | " + " | ".join(out))
print()
print("枝別の中央値（距離をまたいで）:")
for sgn in (+1, -1):
    print(("pos" if sgn > 0 else "neg") + " | " + " | ".join(f"{np.median(v):.2f}" if len(v) else "-" for v in agg[sgn]))
