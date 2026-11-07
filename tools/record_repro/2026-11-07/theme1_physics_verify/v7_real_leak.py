"""S 字の候補「R,B の色の混ざり」を実データから直接確かめる。
大きくぼけた行では R と B の相互相関の主な山はずれ d の位置に出る。混ざり x があると lag=0 にも山が出る
（R⋆B = (1-x)² h_Y⋆h_C + x² h_C⋆h_Y + x(1-x)(h_Y⋆h_Y + h_C⋆h_C)）。
クロマの間引きの影響を避けるため、低い帯域（0.03〜0.125 cy/px@1080）だけで見る。合成と同じ指標で比べる。
入力: data/calib_raw/*.png、data/test/2300.JPG（読み取りのみ）。
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

RAW = os.path.join(ROOT, "data", "calib_raw")
SS = 2
SLOPE = 0.5 / 5.6
BETA = 4 / (3 * np.pi)


def xcorr_band(a, b, fl=0.03, fh=0.125, maxlag=40):
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
    return np.arange(-maxlag, maxlag + 1), xc


def metrics(lag, xc):
    i = int(np.argmax(xc))
    return lag[i], xc[i], xc[lag == 0][0], xc[lag == -lag[i]][0] if abs(lag[i]) <= lag.max() else np.nan


def real_table():
    print("=== 実データ（動画経路 1080×720・ノイズ板）: 行帯ごとの R-B 相互相関 ===")
    print("距離 行帯@720 | 山の位置[px@1080] | 山の相関 | lag=0 の相関 | -山の位置の相関 | lag0/山")
    for dist in [1400, 2000, 2600, 3200]:
        img = cv2.imread(os.path.join(RAW, f"{dist}.png"))[:, 100:1180]
        b, _, r = cv2.split(img)
        for y0 in range(0, 720, 90):
            lag, xc = xcorr_band(r[y0 : y0 + 90], b[y0 : y0 + 90])
            m = metrics(lag, xc)
            print(f"{dist} {y0}-{y0+90} | {m[0]:+d} | {m[1]:.3f} | {m[2]:.3f} | {m[3]:.3f} | {m[2]/m[1]:.2f}")
    path = os.path.join(ROOT, "data", "test", "2300.JPG")
    still = cv2.imread(path)
    img = cv2.resize(still, (1080, 720), interpolation=cv2.INTER_AREA)
    b, _, r = cv2.split(img)
    print("--- 静止画 2300.JPG（4:2:2 を 1080 へ。車の写真の板）---")
    for y0 in range(0, 720, 90):
        lag, xc = xcorr_band(r[y0 : y0 + 90], b[y0 : y0 + 90])
        m = metrics(lag, xc)
        print(f"still {y0}-{y0+90} | {m[0]:+d} | {m[1]:.3f} | {m[2]:.3f} | {m[3]:.3f} | {m[2]/m[1]:.2f}")


def render_uniform(mask, c, leak):
    kY, kC = sy.half_kernels(c, abs(c) * 0.866, SS)
    a = sy.box_down(sy.conv(mask, kY) * 2, SS)
    b = sy.box_down(sy.conv(mask, kC) * 2, SS)
    lo, hi = 0.05, 0.7
    Y = lo + (hi - lo) * a
    C = lo + (hi - lo) * b
    return dict(R=(1 - leak) * Y + leak * C, G=0.7 * Y + 0.3 * C, B=(1 - leak) * C + leak * Y)


def sim_table():
    print()
    print("=== 合成（一様な c・ランダム四角模様・ガンマ後 σ=1.5・4:2:0 あり）: 同じ指標 ===")
    print("粒[px@1080] | 混ざり | c[px@1080] | 山の位置 | 山の相関 | lag=0 の相関 | -山の相関 | lag0/山")
    H, W = 180, 800
    rng = np.random.default_rng(5)
    for grain in [11.0, 22.0]:
        ny, nx = int(H / grain) + 4, int(W / grain) + 4
        g = (rng.random((ny, nx)) > 0.5).astype(np.float32)
        mask = cv2.resize(g, (W * SS, H * SS), interpolation=cv2.INTER_NEAREST)
        for leak in [0.0, 0.1, 0.2, 0.3]:
            for c in [12.0, 24.0, 40.0]:
                lin = render_uniform(mask, c, leak)
                bgr = sy.chroma420(sy.to_u8(lin, noise=1.5, rng=np.random.default_rng(1)))
                lag, xc = xcorr_band(bgr[20:-20, :, 2], bgr[20:-20, :, 0])
                m = metrics(lag, xc)
                print(f"{grain} | {leak} | {c} | {m[0]:+d} | {m[1]:.3f} | {m[2]:.3f} | {m[3]:.3f} | {m[2]/m[1]:.2f}")


real_table()
sim_table()
