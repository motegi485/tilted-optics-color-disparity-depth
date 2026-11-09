# -*- coding: utf-8 -*-
"""色フィルタ開口の幾何評価ライブラリ（テーマ 2・設計空間の解析用）。

目的: 開口の各点にフィルタ（W/Y/C/M/R/G/B/O）を割り当てた設計について、RGB 各チャネルの
      透過領域の面積・重心・線広がり関数（LSF）を求め、方向ごとの基線・Fisher 情報・ぼけ形状の一致度を出す。
入力: 設計関数 f(x, y) -> T (N,3)（各点の R,G,B 透過率 0..1。開口の外は 0）
出力: 指標の辞書（evaluate()）
仮定（すべての計算に共通）:
  - 幾何光学・回折なし・ケラレなし・アオリなし（レンズ傾斜は別担当 physB が扱う）。
  - 開口は半径 1 の円。座標は x 右・y 上、角度は +x から反時計回り [deg]。
  - 「縦エッジ」= 法線が水平 n=(1,0)、「横エッジ」= 法線が垂直 n=(0,1)。
  - チャネル c の像は、ぼけの大きさ d に比例して重心 p_c だけずれる（d·p_c）。距離の誤差は基線に反比例。
  - 理想フィルタ: 各フィルタはそのチャネルを 0 か 1 で通す（実フィルタの近似は a4_spectral.py）。
"""
import numpy as np

FILTERS = {
    "W": (1.0, 1.0, 1.0),  # 素通し
    "Y": (1.0, 1.0, 0.0),  # イエロー（R+G）
    "C": (0.0, 1.0, 1.0),  # シアン（G+B）
    "M": (1.0, 0.0, 1.0),  # マゼンタ（R+B）
    "R": (1.0, 0.0, 0.0),
    "G": (0.0, 1.0, 0.0),
    "B": (0.0, 0.0, 1.0),
    "O": (0.0, 0.0, 0.0),  # 不透明
}
CH = ("R", "G", "B")
PAIRS = ((0, 1), (0, 2), (1, 2))


# ---------------------------------------------------------------------------
# 格子
# ---------------------------------------------------------------------------
class Grid:
    """半径 1 の円の内側の格子点（セル中心）。"""

    def __init__(self, n=1200):
        h = 2.0 / n
        g = -1.0 + (np.arange(n) + 0.5) * h
        X, Y = np.meshgrid(g, g)
        m = X * X + Y * Y <= 1.0
        self.n = n
        self.h = h
        self.dA = h * h
        self.x = X[m].astype(np.float64)
        self.y = Y[m].astype(np.float64)
        self.P = np.stack([self.x, self.y], 1)
        self.img_mask = m
        self.X, self.Y = X, Y


# ---------------------------------------------------------------------------
# 形状の部品
# ---------------------------------------------------------------------------
def angle_deg(x, y):
    return np.degrees(np.arctan2(y, x)) % 360.0


def in_disk(x, y, cx=0.0, cy=0.0, r=1.0):
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def in_sector(x, y, center_deg, width_deg):
    d = (angle_deg(x, y) - center_deg + 180.0) % 360.0 - 180.0
    return np.abs(d) <= width_deg / 2.0


def in_halfplane(x, y, dir_deg, offset=0.0):
    """方向 dir_deg の単位ベクトル u について u·x > offset（offset=0 で中心を通る半平面）"""
    u = np.deg2rad(dir_deg)
    return x * np.cos(u) + y * np.sin(u) > offset


def in_convex_polygon(x, y, verts):
    """反時計回りの頂点列 verts (k,2) の凸多角形の内側"""
    inside = np.ones_like(x, dtype=bool)
    k = len(verts)
    for i in range(k):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % k]
        inside &= (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0) >= 0
    return inside


def regular_polygon(cx, cy, circumradius, nside, rot_deg):
    t = np.deg2rad(rot_deg + 360.0 * np.arange(nside) / nside)
    return np.stack([cx + circumradius * np.cos(t), cy + circumradius * np.sin(t)], 1)


def paint(x, y, layers, background="O"):
    """layers: [(mask, filter_letter or (N,3) array), ...]。後の層が上書き。開口の外は 0。"""
    T = np.tile(np.array(FILTERS[background]), (x.size, 1))
    for mask, f in layers:
        if isinstance(f, str):
            T[mask] = FILTERS[f]
        else:
            T[mask] = f[mask]
    T[~in_disk(x, y)] = 0.0
    return T


def rotate_design(func, deg):
    """設計関数を deg だけ回した設計関数を返す"""
    c, s = np.cos(np.deg2rad(-deg)), np.sin(np.deg2rad(-deg))

    def f(x, y):
        return func(c * x - s * y, s * x + c * y)

    return f


# ---------------------------------------------------------------------------
# 円の切片（弦で切った領域）: 面積 a（開口比）に対する重心距離の上限 g(a)
# ---------------------------------------------------------------------------
def segment_centroid_dist(a):
    """面積が開口の a 倍の領域の重心が中心から離れうる最大距離（弦で切った切片で達成・浴槽原理）。

    切片の半中心角 α について 面積 = α − sinα cosα、重心距離 = (2/3) sin³α / 面積。
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    out = np.zeros_like(a)
    for i, ai in enumerate(a):
        if ai <= 0:
            out[i] = 1.0
            continue
        if ai >= 1:
            out[i] = 0.0
            continue
        target = ai * np.pi
        lo, hi = 0.0, np.pi
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if mid - np.sin(mid) * np.cos(mid) < target:
                lo = mid
            else:
                hi = mid
        al = 0.5 * (lo + hi)
        out[i] = (2.0 / 3.0) * np.sin(al) ** 3 / target
    return out if out.size > 1 else float(out[0])


# ---------------------------------------------------------------------------
# 重心だけで決まる指標
# ---------------------------------------------------------------------------
def tri_min_width(p, valid):
    """有効なチャネルの重心が作る三角形の最小幅（= 最も苦手な方向の『最大の重心差の法線成分』）"""
    q = p[valid]
    if len(q) < 3:
        return 0.0
    a2 = abs((q[1, 0] - q[0, 0]) * (q[2, 1] - q[0, 1]) - (q[1, 1] - q[0, 1]) * (q[2, 0] - q[0, 0]))
    sides = [np.hypot(*(q[1] - q[0])), np.hypot(*(q[2] - q[1])), np.hypot(*(q[0] - q[2]))]
    return a2 / max(sides) if max(sides) > 0 else 0.0


def fisher_matrix(a, p):
    """S = Σ a_i (p_i − p̄)(p_i − p̄)^T（p̄ は光量で重み付けした平均）。

    仮定: エッジ位置の推定雑音の分散がチャネルの光量 a_i に反比例（光子雑音律速）、
    共通のずれ（エッジの真の位置）は未知の nuisance。I(n) = n^T S n。
    """
    w = np.where(a > 1e-12, a, 0.0)
    if w.sum() <= 0:
        return np.zeros((2, 2))
    pb = (w[:, None] * np.nan_to_num(p)).sum(0) / w.sum()
    dp = np.nan_to_num(p) - pb
    return np.einsum("i,ij,ik->jk", w, dp, dp)


def eig2(S):
    tr = S[0, 0] + S[1, 1]
    det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
    disc = np.sqrt(max(tr * tr / 4 - det, 0.0))
    return tr / 2 - disc, tr / 2 + disc


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------
def channel_moments(T, grid):
    a = T.sum(0) * grid.dA / np.pi  # 開口比
    m = np.einsum("nc,nk->ck", T, grid.P) * grid.dA / np.pi
    with np.errstate(invalid="ignore", divide="ignore"):
        p = m / a[:, None]
    return a, m, p


def lsf_stack(T, grid, theta_deg, hs=0.004):
    """方向 theta の法線 n への投影（線広がり関数）。返り値 L (3, nb)・s 軸。"""
    th = np.deg2rad(theta_deg)
    s = grid.x * np.cos(th) + grid.y * np.sin(th)
    nb = int(round(2.0 / hs)) + 4
    # 線形補間で隣の 2 ビンに配る（格子とビンの周期が干渉して相互相関のピークが 0.02 刻みに固着するのを防ぐ）
    fpos = (s + 1.0) / hs + 0.5
    i0 = np.floor(fpos).astype(np.int64)
    w1 = fpos - i0
    i0 = np.clip(i0, 0, nb - 2)
    L = np.stack([np.bincount(i0, weights=T[:, c] * (1 - w1), minlength=nb)
                  + np.bincount(i0 + 1, weights=T[:, c] * w1, minlength=nb) for c in range(3)]) * grid.dA / hs
    sax = -1.0 + (np.arange(nb) - 0.5) * hs
    return L, sax


def xcorr_peak(Li, Lj, hs):
    """c(τ) = Σ_s Li(s) Lj(s+τ) のピーク位置 τ*（放物線補間）と、そこでの正規化相関"""
    c = np.correlate(Lj, Li, mode="full")
    k = int(np.argmax(c))
    off = 0.0
    if 0 < k < len(c) - 1:
        den = c[k - 1] - 2 * c[k] + c[k + 1]
        if den < 0:
            off = 0.5 * (c[k - 1] - c[k + 1]) / den
    tau = (k + off - (len(Li) - 1)) * hs
    ncc = c[k] / np.sqrt((Li * Li).sum() * (Lj * Lj).sum())
    return tau, ncc


def evaluate(func, grid, lsf_step_deg=2.0, hs=0.0025, eps_light=0.005, name=""):
    T = func(grid.x, grid.y)
    a, m, p = channel_moments(T, grid)
    valid = a > eps_light
    thetas = np.arange(0.0, 180.0, 1.0)
    N = np.stack([np.cos(np.deg2rad(thetas)), np.sin(np.deg2rad(thetas))], 1)
    q = np.where(valid[:, None], np.nan_to_num(p) @ N.T, np.nan)  # (3, nθ)
    prof = np.nanmax(q, 0) - np.nanmin(q, 0)  # 最大の重心差の法線成分
    S = fisher_matrix(np.where(valid, a, 0.0), p)
    lmin, lmax = eig2(S)
    Iprof = np.einsum("tj,jk,tk->t", N, S, N)
    out = dict(name=name, a=a, p=p, valid=valid,
               light_mean=a.sum() / 3.0, light_G=a[1], G_full=bool(a[1] > 0.995),
               vert=prof[0], horiz=prof[90], worst=prof.min(), worst_theta=thetas[np.argmin(prof)],
               worst_tri=tri_min_width(np.nan_to_num(p), valid),
               prof=prof, thetas=thetas, S=S, fisher_min=lmin, fisher_max=lmax, Iprof=Iprof,
               fisher_vert=Iprof[0], fisher_horiz=Iprof[90])
    # --- LSF に基づく指標（2° 刻み） ---
    lth = np.arange(0.0, 180.0, lsf_step_deg)
    tau = np.full((len(lth), 3), np.nan)
    ncc = np.full((len(lth), 3), np.nan)
    cdiff = np.full((len(lth), 3), np.nan)
    weq = np.full((len(lth), 3), np.nan)
    Iw = np.full(len(lth), np.nan)
    for t_i, th in enumerate(lth):
        L, sax = lsf_stack(T, grid, th, hs)
        tot = L.sum(1) * hs
        Lh = np.where(tot[:, None] > 0, L / np.maximum(tot[:, None], 1e-30), 0.0)
        for c in range(3):
            if valid[c]:
                weq[t_i, c] = 1.0 / ((Lh[c] ** 2).sum() * hs)
        n = np.array([np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))])
        for k, (i, j) in enumerate(PAIRS):
            if valid[i] and valid[j]:
                tau[t_i, k], ncc[t_i, k] = xcorr_peak(Lh[i], Lh[j], hs)
                cdiff[t_i, k] = (p[j] - p[i]) @ n
        # 幅で重みを補正した Fisher: u_i = a_i / w_i（光子雑音・ぼけた段差の位置推定の CRLB ∝ w/a）
        u = np.where(valid, a / np.where(np.isfinite(weq[t_i]), weq[t_i], np.inf), 0.0)
        qq = np.nan_to_num(p) @ n
        if u.sum() > 0:
            qb = (u * qq).sum() / u.sum()
            Iw[t_i] = (u * (qq - qb) ** 2).sum()
    # 相互相関ピークによる基線: 重心差が 0.02 を超えるペアについて、重心差の向きに測ったピーク位置（逆向きなら負）。
    # 重心差がほぼ 0 のペアでピークが ±τ0 に割れる場合（対称な二峰）は符号の情報が無いので数えない。
    signed = np.where(np.abs(cdiff) > 0.02, tau * np.sign(cdiff), np.nan)
    zprof = np.where(np.all(np.isnan(signed), 1), 0.0,
                     np.clip(np.nanmax(np.where(np.isnan(signed), -np.inf, signed), 1), 0, None))
    # 重心差に対する相互相関ピークの比（重心差が十分大きい方向・ペアだけ）
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(np.abs(cdiff) > 0.05, tau / cdiff, np.nan)
    # 各方向で「最大の重心差を与えるペア」の LSF 相関
    best_pair = np.nanargmax(np.where(np.isfinite(cdiff), np.abs(cdiff), -1), 1)
    ncc_best = ncc[np.arange(len(lth)), best_pair]
    out.update(lsf_thetas=lth, tau=tau, ncc=ncc, cdiff=cdiff, weq=weq, Iw=Iw,
               zprof=zprof, zworst=float(np.nanmin(zprof)), zvert=float(zprof[0]),
               zhoriz=float(zprof[int(round(90 / lsf_step_deg))]),
               ratio_med=float(np.nanmedian(ratio)) if np.any(np.isfinite(ratio)) else np.nan,
               ncc_best_min=float(np.nanmin(ncc_best)), ncc_best_med=float(np.nanmedian(ncc_best)),
               Iw_min=float(np.nanmin(Iw)))
    # --- 2D の形状一致（重心をそろえた透過マスクの正規化相関） ---
    sim2d = np.full(3, np.nan)
    for k, (i, j) in enumerate(PAIRS):
        if valid[i] and valid[j]:
            dlt = p[j] - p[i]
            Tj = func(grid.x + dlt[0], grid.y + dlt[1])[:, j]
            sim2d[k] = (T[:, i] * Tj).sum() / np.sqrt((T[:, i] ** 2).sum() * (T[:, j] ** 2).sum())
    out["sim2d"] = sim2d
    out["T"] = T
    return out


def render_rgb(func, n=241, bg=0.18):
    """設計を画像に（各点の透過率 (R,G,B) をそのまま色に。開口の外は灰色）"""
    g = -1.0 + (np.arange(n) + 0.5) * (2.0 / n)
    X, Y = np.meshgrid(g, g[::-1])
    T = func(X.ravel(), Y.ravel()).reshape(n, n, 3)
    m = (X * X + Y * Y <= 1.0)
    img = np.where(m[..., None], T, bg)
    return img
