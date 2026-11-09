"""verPhys の独立な合成・推定ライブラリ（physB の apsim.py は使わない・2026-11-09）

目的: physB の合成の主要な結論を、別に書いたコードで再現できるかを確かめる。
入力: なし（ライブラリ）
出力: なし
仮定（すべて [仮定]）:
  - 幾何光学。PSF は直径 |c| px の円をチャネルの透過領域で切り取ったもの（アオリは縦を sy 倍）。
    c<0 は開口の正立像、c>0 は倒立像（点対称）。どちらでもチャネル k の像のずれは (c/2)·q_k、q_k=(−u_k, sy·v_k)。
  - 理想フィルタ・無彩色の被写体。各チャネルの PSF の総和を 1（ホワイトバランス後）。
  - 4 倍の格子で畳み込み、箱型平均で縮小。ガンマ 1/2.2 → ガウス雑音（8 bit・既定 σ=1.5）→ 丸め。
    physB は sRGB の区分ガンマ・周期境界の FFT。ここでは単純なべき乗・余白つきの線形畳み込みにして、実装を意図的に変えた。
"""
import sys

sys.dont_write_bytecode = True
import cv2
import numpy as np
from scipy import ndimage, signal

CH = ("R", "G", "B")


# ---------------------------------------------------------------------------
# 開口（u 右・v 上・単位円）
# ---------------------------------------------------------------------------
def _ang(u, v):
    return np.degrees(np.arctan2(v, u)) % 360.0


def _sec(a, center, width):
    d = (a - center + 180.0) % 360.0 - 180.0
    return np.abs(d) <= width / 2.0


def ap_full(u, v):
    o = np.ones(np.shape(u))
    return {"R": o, "G": o, "B": o}


def ap_current(u, v):
    L = u < 0
    return {"R": L * 1.0, "G": np.ones(np.shape(u)), "B": (~L) * 1.0}


def ap_tb(u, v):
    """現行を 90° 回した上下分割（縦の基線）: R=上半分・B=下半分"""
    T = v > 0
    return {"R": T * 1.0, "G": np.ones(np.shape(u)), "B": (~T) * 1.0}


def ap_rgb3(u, v):
    a = _ang(u, v)
    return {"R": _sec(a, 210, 120) * 1.0, "G": _sec(a, 90, 120) * 1.0, "B": _sec(a, 330, 120) * 1.0}


def ap_cmy3(u, v):
    a = _ang(u, v)
    M, C, Y = _sec(a, 90, 120), _sec(a, 210, 120), _sec(a, 330, 120)
    return {"R": (M | Y) * 1.0, "G": (C | Y) * 1.0, "B": (M | C) * 1.0}


def make_famB(w):
    def f(u, v):
        a = _ang(u, v)
        s = _sec(a, 270, w)
        L = u < 0
        return {"R": (L & ~s) * 1.0, "G": np.ones(np.shape(u)), "B": (~L & ~s) * 1.0}

    return f


SHAPES = {
    "current": ap_current,
    "tb": ap_tb,
    "rgb3": ap_rgb3,
    "cmy3": ap_cmy3,
    "famB90": make_famB(90.0),
    "full": ap_full,
}


def moments(fn, n=1501):
    g = (np.arange(n) + 0.5) / n * 2 - 1
    U, V = np.meshgrid(g, g[::-1])
    ins = U ** 2 + V ** 2 <= 1
    T = fn(U, V)
    out = {}
    for k in CH:
        t = np.where(ins, T[k], 0.0)
        s = t.sum()
        out[k] = (s / ins.sum(), (t * U).sum() / s, (t * V).sum() / s)
    return out


def qvec(fn, sy=1.0):
    """像面（x 右・y 下）での c=2 あたりのずれ q_k = (−u_k, sy·v_k)"""
    m = moments(fn)
    return {k: np.array([-m[k][1], sy * m[k][2]]) for k in CH}


# ---------------------------------------------------------------------------
# PSF と描画
# ---------------------------------------------------------------------------
def psf(fn, c, h=1.0, sy=1.0, nsub=4):
    """符号付きぼけ c（1× の px）、格子の刻み h（1× の px）。各チャネルの核（奇数サイズ・中心が原点・総和 1）"""
    r = abs(c) / 2.0
    if r * min(1.0, sy) < 0.35 * h:
        d = np.ones((1, 1))
        return {k: d for k in CH}
    R = int(np.ceil(r / h)) + 1
    idx = np.arange(-R, R + 1, dtype=np.float64)
    off = (np.arange(nsub) + 0.5) / nsub - 0.5
    acc = {k: np.zeros((2 * R + 1, 2 * R + 1)) for k in CH}
    for oy in off:
        for ox in off:
            X = (idx[None, :] + ox) * h
            Y = (idx[:, None] + oy) * h
            if c > 0:
                u, v = -X / r, Y / (r * sy)
            else:
                u, v = X / r, -Y / (r * sy)
            ins = (u ** 2 + v ** 2) <= 1.0
            T = fn(u, v)
            for k in CH:
                acc[k] += ins * T[k]
    out = {}
    for k in CH:
        s = acc[k].sum()
        if s <= 0:
            raise ValueError("透過ゼロ")
        out[k] = acc[k] / s
    return out


def block_mean(a, ss):
    if ss == 1:
        return a
    H, W = a.shape
    return a[: H // ss * ss, : W // ss * ss].reshape(H // ss, ss, W // ss, ss).mean(axis=(1, 3))


def render(subj_ss, fn, c, ss=4, sy=1.0, margin=24):
    """subj_ss は余白 margin（1× の px）を含む 4 倍格子の被写体。線形の畳み込み→縮小→余白を落とす"""
    ks = psf(fn, c, h=1.0 / ss, sy=sy, nsub=3)
    out = {}
    cache = {}
    for k in CH:
        key = ks[k].tobytes()
        if key not in cache:
            img = signal.fftconvolve(subj_ss, ks[k], mode="same")
            cache[key] = block_mean(img, ss)[margin:-margin, margin:-margin]
        out[k] = cache[key]
    return out


def to8(lin, sigma=1.5, rng=None):
    out = {}
    for k in CH:
        v = 255.0 * np.clip(lin[k], 0, 1) ** (1 / 2.2)
        if sigma > 0:
            v = v + rng.normal(0, sigma, v.shape)
        out[k] = np.clip(np.rint(v), 0, 255).astype(np.uint8)
    return out


def lin8(a8):
    return (a8.astype(np.float64) / 255.0) ** 2.2


# ---------------------------------------------------------------------------
# 被写体（4 倍格子・線形の反射率）
# ---------------------------------------------------------------------------
def subj_1overf(H, W, ss, rng, aniso_deg=None, floor=0.03):
    """1/f 雑音（複素ガウスのスペクトル）。aniso_deg を与えると、周波数ベクトルが fy 軸に近い成分（横向きの構造）を強める:
    振幅 × (floor + exp(−ψ²/(2σ²)))。physB の『強い異方性』（σ=4°・床 0.03）の仕様を文書から写して自分で実装した"""
    Hs, Ws = H * ss, W * ss
    fy = np.fft.fftfreq(Hs)[:, None] * ss
    fx = np.fft.rfftfreq(Ws)[None, :] * ss
    f = np.hypot(fx, fy)
    amp = np.where((f > 0) & (f <= 0.5), 1.0 / np.maximum(f, 1e-9), 0.0)
    if aniso_deg is not None:
        psi = np.degrees(np.arctan2(np.abs(fx), np.abs(fy)))
        amp = amp * (floor + np.exp(-(psi ** 2) / (2 * aniso_deg ** 2)))
    spec = amp * (rng.normal(size=amp.shape) + 1j * rng.normal(size=amp.shape))
    img = np.fft.irfft2(spec, s=(Hs, Ws))
    img = (img - img.mean()) / img.std() * 0.12 + 0.30
    return np.clip(img, 0.03, 0.80)


def subj_bars(H, W, ss, rng, finite=True, bend=True, Lr=(40, 160), Ar=(0.5, 3.0), lamr=(80, 300), tr=(2, 8), density=2.5):
    """横の帯を重ね塗りした被写体。finite: 帯に端がある（長さ Lr px）。bend: 帯が y 方向に正弦波で曲がる（振幅 Ar px・波長 lamr px）。
    finite=False・bend=False は無限に長いまっすぐな横縞（＝バーコード φ=90°）"""
    Hs, Ws = H * ss, W * ss
    y = (np.arange(Hs) + 0.5) / ss
    x = (np.arange(Ws) + 0.5) / ss
    img = np.full((Hs, Ws), 0.30)
    meanL = np.mean(Lr) if finite else W
    nbar = int(density * H * W / (np.mean(tr) * meanL))
    for _ in range(nbar):
        y0 = rng.uniform(-5, H + 5)
        t = rng.uniform(*tr)
        val = rng.uniform(0.05, 0.70)
        if finite:
            Lb = rng.uniform(*Lr)
            x0 = rng.uniform(-Lb, W)
        else:
            Lb = W + 20
            x0 = -10
        cols = np.nonzero((x >= x0) & (x <= x0 + Lb))[0]
        if cols.size == 0:
            continue
        if bend:
            A = rng.uniform(*Ar)
            lam = rng.uniform(*lamr)
            ph = rng.uniform(0, 2 * np.pi)
            yc = y0 + A * np.sin(2 * np.pi * x[cols] / lam + ph)
        else:
            yc = np.full(cols.size, y0)
        m = np.abs(y[:, None] - yc[None, :]) < t / 2
        sub = img[:, cols]
        sub[m] = val
        img[:, cols] = sub
    return img


def subj_barcode(H, W, ss, rng, phi_deg, wr=(2, 10)):
    """1 次元のランダムなバーコード（法線の角度 φ・0=縦エッジ）"""
    Hs, Ws = H * ss, W * ss
    y = (np.arange(Hs) + 0.5) / ss - H / 2
    x = (np.arange(Ws) + 0.5) / ss - W / 2
    X, Y = np.meshgrid(x, y)
    n = np.array([np.cos(np.radians(phi_deg)), np.sin(np.radians(phi_deg))])
    t = X * n[0] + Y * n[1]
    e = [t.min() - 1]
    while e[-1] < t.max() + 1:
        e.append(e[-1] + rng.uniform(*wr))
    e = np.array(e)
    vals = rng.uniform(0.05, 0.70, len(e))
    return vals[np.clip(np.searchsorted(e, t) - 1, 0, len(vals) - 1)]


def subj_step(H, W, ss, phi_deg):
    Hs, Ws = H * ss, W * ss
    y = (np.arange(Hs) + 0.5) / ss - H / 2
    x = (np.arange(Ws) + 0.5) / ss - W / 2
    X, Y = np.meshgrid(x, y)
    n = np.array([np.cos(np.radians(phi_deg)), np.sin(np.radians(phi_deg))])
    return np.where(X * n[0] + Y * n[1] > 0, 0.55, 0.12)


# ---------------------------------------------------------------------------
# 推定法（すべて自作）
# ---------------------------------------------------------------------------
def uf(a, K=25):
    return ndimage.uniform_filter(a, size=K, mode="reflect")


def zncc_win(a, b, K=25, ma=None, va=None):
    if ma is None:
        ma = uf(a, K)
        va = uf(a * a, K) - ma * ma
    mb = uf(b, K)
    vb = uf(b * b, K) - mb * mb
    cov = uf(a * b, K) - ma * mb
    den = np.sqrt(np.maximum(va, 0) * np.maximum(vb, 0))
    return np.where(den > 1e-12, cov / np.maximum(den, 1e-30), 0.0)


def peak(S, cands):
    """候補軸 0。端を除く最大（最初の最大）→ 厳密な局所最大 → 放物線"""
    N = S.shape[0]
    j = np.argmax(S[1 : N - 1], axis=0) + 1
    tk = lambda jj: np.take_along_axis(S, jj[None], 0)[0]
    p0, pm, pp = tk(j), tk(j - 1), tk(j + 1)
    ok = (p0 > pm) & (p0 > pp)
    den = 2 * p0 - pm - pp
    dl = np.where(ok & (den > 0), 0.5 * (pp - pm) / np.where(den > 0, den, 1.0), 0.0)
    step = cands[1] - cands[0]
    return np.where(ok, cands[0] + (j + dl) * step, np.nan)


def sobel_dx_u8(a8):
    return cv2.convertScaleAbs(cv2.Sobel(a8, cv2.CV_32F, 1, 0, ksize=3)).astype(np.float64)


def grad_mag(a8):
    f = a8.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy).astype(np.float64), gx.astype(np.float64), gy.astype(np.float64)


def est_i(u8, q, K=25, S=41, T=20, feat="dx", sel="dx"):
    """(i) 現行の R–B 横 ZNCC を自分で書き直したもの（_disparity_reference.py の規則を読んで再実装）。
    feat: 'dx'＝|Sobel x| を uint8 に飽和（現行）、'lin'＝線形の明るさ。
    sel : 'dx'＝R の |∂x|>T（現行）、'grad'＝R の |∇|>T（向きを問わない選別）。
    戻り値は ĉ = 2d/(q_R−q_B)_x（d = x_R − x_B）"""
    R8, B8 = u8["R"], u8["B"]
    if feat == "dx":
        a, b = sobel_dx_u8(R8), sobel_dx_u8(B8)
    else:
        a, b = lin8(R8), lin8(B8)
    H, W = a.shape
    sr = (S - 1) // 2
    k = (K - 1) // 2
    ds = np.arange(-sr, sr + 1)
    ma = uf(a, K)
    va = uf(a * a, K) - ma * ma
    Sim = np.empty((len(ds), H, W))
    for n, d in enumerate(ds):
        Sim[n] = zncc_win(a, np.roll(b, d, axis=1), K, ma, va)  # roll: b(x−d)
    d = peak(Sim, ds.astype(np.float64))
    if sel == "dx":
        m = sobel_dx_u8(R8) > T
    else:
        m = grad_mag(R8)[0] > T
    valid = np.zeros_like(m)
    valid[k : H - k, sr + k + 1 : W - sr - k - 1] = True
    d[~(m & valid)] = np.nan
    return 2.0 * d / (q["R"][0] - q["B"][0])


def est_colorline(u8, q, cands, K=25, T=20, chans=CH):
    """(ii) Bando 2008 の color-line 指標: 候補 c ごとに各チャネルを (c/2)·q_k だけ戻し、窓内の線形の明るさの相関行列の det を最小化。
    2 チャネルだけ渡すと 1 − r²（＝ZNCC の 2 乗）を最大化することになる"""
    L = {k: lin8(u8[k]) for k in chans}
    co = {k: ndimage.spline_filter(L[k], order=3, mode="nearest") for k in chans}
    H, W = L[chans[0]].shape
    Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
    Sc = np.empty((len(cands), H, W))
    for n, c in enumerate(cands):
        A = {}
        for k in chans:
            dx, dy = 0.5 * c * q[k][0], 0.5 * c * q[k][1]
            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                A[k] = L[k]
            else:
                A[k] = ndimage.map_coordinates(co[k], [Y + dy, X + dx], order=3, mode="nearest", prefilter=False)
        m = {k: uf(A[k], K) for k in chans}
        C = {}
        for i, ki in enumerate(chans):
            for kj in chans[i:]:
                C[(ki, kj)] = uf(A[ki] * A[kj], K) - m[ki] * m[kj]
        if len(chans) == 3:
            r, g_, b = chans
            vr, vg, vb = [np.maximum(C[(x, x)], 1e-14) for x in chans]
            rrg = C[(r, g_)] / np.sqrt(vr * vg)
            rgb = C[(g_, b)] / np.sqrt(vg * vb)
            rrb = C[(r, b)] / np.sqrt(vr * vb)
            det = 1 + 2 * rrg * rgb * rrb - rrg ** 2 - rgb ** 2 - rrb ** 2
        else:
            a_, b_ = chans
            r_ = C[(a_, b_)] / np.sqrt(np.maximum(C[(a_, a_)], 1e-14) * np.maximum(C[(b_, b_)], 1e-14))
            det = 1 - np.sign(r_) * r_ ** 2  # 負の相関を誤って採らない
        Sc[n] = -det
    est = peak(Sc, np.asarray(cands, dtype=np.float64))
    sel = np.maximum.reduce([grad_mag(u8[k])[0] for k in CH]) > T
    est[~sel] = np.nan
    return est


def est_psfmatch(u8, fn, cands, K=25, T=20, sy=1.0, pairs=(("R", "G"), ("G", "B"), ("R", "B"))):
    """(iv) PSF 合わせの自作版: 候補 c ごとに A=I_i⊛K_j(c)、B=I_j⊛K_i(c)（線形の明るさ）を作り、窓内の明るさの ZNCC の対平均を最大化。
    physB は |∇A|・|∇B| の ZNCC。ここでは明るさそのものの ZNCC にして実装を変えた"""
    L = {k: lin8(u8[k]) for k in CH}
    H, W = L["R"].shape
    Sc = np.empty((len(cands), H, W))
    for n, c in enumerate(cands):
        ks = psf(fn, float(c), h=1.0, sy=sy, nsub=8)
        z = []
        for i, j in pairs:
            A = cv2.filter2D(L[i], cv2.CV_64F, np.ascontiguousarray(ks[j][::-1, ::-1]), borderType=cv2.BORDER_REFLECT)
            B = cv2.filter2D(L[j], cv2.CV_64F, np.ascontiguousarray(ks[i][::-1, ::-1]), borderType=cv2.BORDER_REFLECT)
            z.append(zncc_win(A, B, K))
        Sc[n] = sum(z) / len(z)
    est = peak(Sc, np.asarray(cands, dtype=np.float64))
    sel = np.maximum.reduce([grad_mag(u8[k])[0] for k in CH]) > T
    est[~sel] = np.nan
    return est


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------
def eval_mask(subj_ss, c, ss=4, sy=1.0, margin=24, T=20, roi=40):
    """形状によらない評価画素: 全開口・雑音なしで描いた G の |∇|>T（中央の ROI）"""
    lin = render(subj_ss, ap_full, c, ss, sy, margin)
    u8 = to8(lin, 0.0)
    gm, gx, gy = grad_mag(u8["G"])
    m = gm > T
    r = np.zeros_like(m)
    r[roi:-roi, roi:-roi] = True
    return m & r, gx, gy


def window_h(gx, gy, K=25):
    """窓の構造テンソルの Txx/(Txx+Tyy)（physA の指標 h を自分で計算）"""
    Txx, Tyy = uf(gx * gx, K), uf(gy * gy, K)
    return Txx / (Txx + Tyy + 1e-12)


def evaluate(est, truth, E, tol=1.0):
    n = int(E.sum())
    t = truth if np.isscalar(truth) else truth[E]
    e = est[E] - t
    fin = np.isfinite(e)
    ef = e[fin]
    if ef.size == 0:
        return dict(n=n, cov=0.0, good=0.0, good_own=np.nan, bias=np.nan, sd=np.nan)
    med = float(np.median(ef))
    return dict(n=n, cov=float(fin.mean()), good=float((np.abs(ef) <= tol).sum() / n),
                good_own=float((np.abs(ef) <= tol).mean()), bias=med, sd=float(1.4826 * np.median(np.abs(ef - med))))
