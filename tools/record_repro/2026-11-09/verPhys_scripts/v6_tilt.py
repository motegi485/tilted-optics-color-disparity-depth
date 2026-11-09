"""verPhys v6: アオリで行ごとにぼけが変わる平面での、縦の基線と横の基線のばらつきの比（2026-11-09）

目的: physB §9 の「θ=30° で縦の基線（横エッジ）のばらつきが横の基線（縦エッジ）の約 3 倍、θ≈4° ではほぼ差がない」を、
      別の描画法で確かめる。physB は「一定のぼけで描いた画像を出力の行ごとに補間」する近似。ここでは
      「被写体の各行（光源の行）がその行のぼけ c(y) でぼける」厳密な描画（1 次元の被写体なので計算できる）にした。
      あわせて、physB 型の近似でも描いて差を見る。誤差の原因が「窓の中のぼけの平均を窓の中心に割り当てること」かを、
      窓内の勾配で重み付けした真値との差で分ける。
入力: なし
出力: verPhys_out/v6_log.txt
仮定: c(y) = c0 + (sinθ/N)·(y − H/2)、N=5.6（研究方針.md §7 の式・physB と同じ）。PSF の縦は cosθ 倍（推定側も知っている）。
      推定は R–B の 1 対だけの color-line（＝窓内の明るさの相関の最大化）を基線の向きに 1 次元で探索。σ=1.5。種 2 つ。
"""
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import vp_lib as vp

OUT = os.path.join(os.path.dirname(HERE), "verPhys_out")
LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(str(s))


H = W = 192
SS = 4
M = 24
NF = 5.6
CANDS = np.arange(-24, 25, 1.0)
K = 25
t0 = time.time()


def barcode1d(n_ss, rng, wr=(2, 10)):
    t = (np.arange(n_ss) + 0.5) / SS
    e = [t.min() - 1]
    while e[-1] < t.max() + 1:
        e.append(e[-1] + rng.uniform(*wr))
    e = np.array(e)
    vals = rng.uniform(0.05, 0.70, len(e))
    return vals[np.clip(np.searchsorted(e, t) - 1, 0, len(vals) - 1)]


def c_of_y(y1x, c0, g):
    return c0 + g * (y1x - H / 2.0)


def render_exact(fn, s1d, axis, c0, g, sy):
    """axis='y': 被写体は y だけで変わる（横エッジ）。axis='x': x だけで変わる（縦エッジ）。
    光源の行 y_s のぼけは c(y_s)。戻り値は線形 RGB（1×・余白を落とした H×W）"""
    Hs = (H + 2 * M) * SS
    Ws = (W + 2 * M) * SS
    ys1x = (np.arange(Hs) + 0.5) / SS - M  # 光源の行の 1× 座標（切り出し後の原点）
    out = {}
    if axis == "y":
        # I_k(y) = Σ_ys s(ys) LSF_k(y − ys; c(ys))
        acc = {k: np.zeros(Hs) for k in vp.CH}
        cache = {}
        for j in range(Hs):
            c = round(float(c_of_y(ys1x[j], c0, g)), 4)
            if c not in cache:
                ks = vp.psf(fn, c, h=1.0 / SS, sy=sy, nsub=3)
                cache[c] = {k: ks[k].sum(1) for k in vp.CH}  # 縦の線広がり
            for k in vp.CH:
                l = cache[c][k]
                R = (len(l) - 1) // 2
                lo, hi = j - R, j + R + 1
                a0, a1 = max(lo, 0), min(hi, Hs)
                acc[k][a0:a1] += s1d[j] * l[a0 - lo : len(l) - (hi - a1)]
        for k in vp.CH:
            prof = acc[k].reshape(-1, SS).mean(1)[M:-M]
            out[k] = np.repeat(prof[:, None], W, 1)
        return out
    # axis == 'x': 各出力行 y の横の核 Kx_y(dx) = Σ_dy PSF(dx, dy; c(y − dy))
    Rmax = int(np.ceil((abs(c0) + abs(g) * (H / 2 + M) + 1) / 2 * SS)) + 2
    Kx = {k: np.zeros((Hs, 2 * Rmax + 1)) for k in vp.CH}
    cache = {}
    for j in range(Hs):
        c = round(float(c_of_y(ys1x[j], c0, g)), 4)
        if c not in cache:
            cache[c] = vp.psf(fn, c, h=1.0 / SS, sy=sy, nsub=3)
        ks = cache[c]
        for k in vp.CH:
            P = ks[k]
            R = (P.shape[0] - 1) // 2
            for dy in range(-R, R + 1):
                yy = j + dy
                if 0 <= yy < Hs:
                    Kx[k][yy, Rmax - R : Rmax + R + 1] += P[dy + R]
    for k in vp.CH:
        img = np.empty((Hs, Ws))
        for yy in range(Hs):
            img[yy] = np.convolve(s1d, Kx[k][yy], mode="same")
        out[k] = vp.block_mean(img, SS)[M:-M, M:-M]
    return out


def render_rowinterp(fn, s1d, axis, c0, g, sy, step=0.25):
    """physB 型の近似: 一定のぼけで描いた画像を、出力の行の c で線形補間"""
    Hs = (H + 2 * M) * SS
    Ws = (W + 2 * M) * SS
    subj = np.repeat(s1d[:, None], Ws, 1) if axis == "y" else np.repeat(s1d[None, :], Hs, 0)
    yc = c_of_y(np.arange(H) + 0.5, c0, g)
    lv = np.arange(np.floor(yc.min() / step) * step, np.ceil(yc.max() / step) * step + step / 2, step)
    imgs = [vp.render(subj, fn, float(l), SS, sy, M) for l in lv]
    out = {k: np.zeros((H, W)) for k in vp.CH}
    for y in range(H):
        t = (yc[y] - lv[0]) / step
        i0 = int(np.clip(np.floor(t), 0, len(lv) - 1))
        i1 = min(i0 + 1, len(lv) - 1)
        w = t - i0
        for k in vp.CH:
            out[k][y] = (1 - w) * imgs[i0][k][y] + w * imgs[i1][k][y]
    return out


def run_case(label, theta_deg, c0, use_g=True, use_sy=True, renderer="exact", seeds=(1, 2), sigma=1.5):
    th = np.radians(theta_deg)
    g = np.sin(th) / NF if use_g else 0.0
    sy = np.cos(th) if use_sy else 1.0
    res = {}
    for shape, axis in [("current", "x"), ("tb", "y")]:
        fn = vp.SHAPES[shape]
        q = vp.qvec(fn, sy)
        errs, errs_w, goods, ns = [], [], 0, 0
        for sd in seeds:
            rng = np.random.default_rng(1000 + sd)
            n_ss = ((H if axis == "y" else W) + 2 * M) * SS
            s1d = barcode1d(n_ss, rng)
            if renderer == "exact":
                lin = render_exact(fn, s1d, axis, c0, g, sy)
            else:
                lin = render_rowinterp(fn, s1d, axis, c0, g, sy)
            # 評価画素: G（どちらの形状でも全開口）の雑音なしの |∇|>20、中央 112×112
            g8 = vp.to8({k: lin["G"] for k in vp.CH}, 0.0)["G"]
            gm, gx, gy = vp.grad_mag(g8)
            E = gm > 20
            roi = np.zeros_like(E)
            roi[40:-40, 40:-40] = True
            E &= roi
            truth = np.repeat(c_of_y(np.arange(H) + 0.5, c0, g)[:, None], W, 1)
            # 窓内の基線方向の勾配エネルギーで重み付けした c（窓の推定が表す c の目安）
            wgt = (gx ** 2) if axis == "x" else (gy ** 2)
            cw = vp.uf(wgt * truth, K) / np.maximum(vp.uf(wgt, K), 1e-12)
            u8 = vp.to8(lin, sigma, np.random.default_rng(2000 + sd + int(c0)))
            est = vp.est_colorline(u8, q, CANDS, K=K, chans=("R", "B"))
            e = est[E] - truth[E]
            ew = est[E] - cw[E]
            fin = np.isfinite(e)
            errs.append(e[fin])
            errs_w.append(ew[fin])
            goods += int((np.abs(e[fin]) <= 1).sum())
            ns += int(E.sum())
        e = np.concatenate(errs)
        ew = np.concatenate(errs_w)
        sd_ = 1.4826 * np.median(np.abs(e - np.median(e)))
        sdw = 1.4826 * np.median(np.abs(ew - np.median(ew)))
        res[shape] = dict(good=goods / ns, sd=sd_, sdw=sdw, bias=np.median(e), n=ns)
    r = res["tb"]["sd"] / res["current"]["sd"]
    rw = res["tb"]["sdw"] / res["current"]["sdw"]
    log(f"  {label:28s} c0={c0:>4.0f}: 横の基線・縦エッジ 良好 {res['current']['good']:.2f} ばらつき {res['current']['sd']:.3f} "
        f"| 縦の基線・横エッジ 良好 {res['tb']['good']:.2f} ばらつき {res['tb']['sd']:.3f} | 比 {r:.2f}"
        f" || 窓の重み付き真値に対するばらつき 横 {res['current']['sdw']:.3f} 縦 {res['tb']['sdw']:.3f} 比 {rw:.2f}  ({time.time() - t0:.0f}s)")
    return res


if __name__ == "__main__":
    log("=== 厳密な描画（光源の行ごとのぼけ）。ばらつき = 1.4826·MAD [px of c]、良好 = |ĉ−c(行)|≤1 の割合（評価画素が分母）")
    for c0 in (6.0, 10.0):
        run_case("アオリなし", 0.0, c0)
        run_case("θ=3.97°", 3.97, c0)
        run_case("θ=30°", 30.0, c0)
        run_case("θ=30° 縦の縮み cosθ だけ", 30.0, c0, use_g=False)
        run_case("θ=30° 行の変化だけ", 30.0, c0, use_sy=False)
    log()
    log("=== physB 型の近似描画（一定のぼけの画像を出力の行で補間）")
    for c0 in (6.0,):
        run_case("近似・アオリなし", 0.0, c0, renderer="interp")
        run_case("近似・θ=30°", 30.0, c0, renderer="interp")
        run_case("近似・θ=3.97°", 3.97, c0, renderer="interp")

    with open(os.path.join(OUT, "v6_log.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(LOG) + "\n")
