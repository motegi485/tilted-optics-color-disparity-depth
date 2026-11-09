"""verPhys v1: 開口の幾何の原理と上限を、physA のコードを使わずに一から確かめる（2026-11-09）

目的:
  1. Y と C（＋不透明）だけで開口を埋めると R・G・B の重心は一直線に並ぶ（式と数値）。
     W か G 単独の領域を入れると崩れる。最小幅（＝最悪の向きの基線）と非共線の同値。
  2. 面積 a の領域の重心距離の上限 g(a)（弦で切った切片）。透過率 0〜1 の濃淡でも超えないか。
     最悪の基線の上限 1.5·g(L)、G 全開口 (√3/2)·g((3L−1)/2) を、光量を不均等にした場合も含めて最適化で確かめる。
  3. 族 A・族 B の表（光量・縦の基線・最悪の基線・Fisher）と、R の重心 x の解析式。
  4. Fisher（光子雑音）の値: 現行の縦 0.180、RGB 三扇形 0.152、上限 0.163。
     「全方向を取ると得意な向きの情報は必ず減る」の反例があるか。
入力: なし
出力: verPhys_out/v1_log.txt（UTF-8）
仮定: 開口は半径 1 の円。理想フィルタ（各チャネル 0/1 透過、濃淡の検査だけ 0〜1）。幾何光学。
"""
import os
import sys

sys.dont_write_bytecode = True
import numpy as np
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "verPhys_out")
os.makedirs(OUT, exist_ok=True)
LOG = []


def log(s=""):
    print(s)
    LOG.append(str(s))


rng = np.random.default_rng(20261109)

# ---------------------------------------------------------------------------
# 格子（開口座標 u 右・v 上）
# ---------------------------------------------------------------------------
N = 1001
g1 = (np.arange(N) + 0.5) / N * 2 - 1
U, V = np.meshgrid(g1, g1[::-1])
INS = U ** 2 + V ** 2 <= 1.0
NIN = INS.sum()
A_DEG = np.degrees(np.arctan2(V, U)) % 360.0


def sec(center, width):
    d = (A_DEG - center + 180.0) % 360.0 - 180.0
    return (np.abs(d) <= width / 2.0) & INS


def stats_from_masks(TR, TG, TB):
    """各チャネルの透過（0〜1 の配列）から 光量 a（開口比）と重心 p=(u,v)"""
    out = {}
    for k, T in zip("RGB", (TR, TG, TB)):
        t = np.where(INS, T, 0.0).astype(np.float64)
        s = t.sum()
        out[k] = (s / NIN, np.array([(t * U).sum() / s, (t * V).sum() / s]))
    return out


def min_width(P):
    """3 点の三角形の最小幅 = 2·面積/最長辺"""
    a, b, c = P
    area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    L = max(np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a))
    return area2 / L if L > 0 else 0.0


def width_scan(P, n=3600):
    """法線 n を走査して min_n max_ij |(p_i−p_j)·n| を直接求める"""
    th = np.linspace(0, np.pi, n, endpoint=False)
    nn = np.stack([np.cos(th), np.sin(th)], 1)
    pr = np.array(P) @ nn.T  # 3×n
    return (pr.max(0) - pr.min(0)).min()


def fisher(stats):
    a = np.array([stats[k][0] for k in "RGB"])
    p = np.array([stats[k][1] for k in "RGB"])
    pb = (a[:, None] * p).sum(0) / a.sum()
    S = sum(a[i] * np.outer(p[i] - pb, p[i] - pb) for i in range(3))
    return S


# ---------------------------------------------------------------------------
# 1. 共線性
# ---------------------------------------------------------------------------
log("=== 1. Y/C だけの開口は共線か（ランダムな Voronoi 分割）")
yy, xx = np.nonzero(INS)
pts_u, pts_v = U[INS], V[INS]


def voronoi_labels(nc):
    seeds = rng.uniform(-1, 1, (nc, 2))
    d = (pts_u[:, None] - seeds[None, :, 0]) ** 2 + (pts_v[:, None] - seeds[None, :, 1]) ** 2
    return np.argmin(d, 1)


# フィルタの種類: 透過 (R,G,B)
FT = {"W": (1, 1, 1), "Y": (1, 1, 0), "C": (0, 1, 1), "M": (1, 0, 1), "R": (1, 0, 0), "G": (0, 1, 0), "B": (0, 0, 1), "K": (0, 0, 0)}


def random_design(types, nc, rng_):
    lab = voronoi_labels(nc)
    assign = rng_.choice(types, nc)
    T = {k: np.zeros(INS.shape) for k in "RGB"}
    for ci in range(nc):
        sel = lab == ci
        tr = FT[assign[ci]]
        for kk, k in enumerate("RGB"):
            if tr[kk]:
                T[k][yy[sel], xx[sel]] = 1.0
    return T


for types, label in [(["Y", "C"], "Y/C"), (["Y", "C", "K"], "Y/C/不透明"), (["Y", "C", "W"], "Y/C/W"), (["Y", "C", "G"], "Y/C/G"),
                     (["Y", "C", "M"], "Y/C/M"), (["W", "Y", "C", "M", "R", "G", "B", "K"], "8 種")]:
    mws, idents = [], []
    for it in range(150):
        T = random_design(types, int(rng.integers(4, 25)), rng)
        if min(T[k][INS].sum() for k in "RGB") < 50:
            continue
        st = stats_from_masks(T["R"], T["G"], T["B"])
        P = [st[k][1] for k in "RGB"]
        mws.append(min_width(P))
        # 恒等式 a_R p_R + a_B p_B − a_G p_G（Y/C/不透明なら 0）
        idents.append(np.linalg.norm(st["R"][0] * st["R"][1] + st["B"][0] * st["B"][1] - st["G"][0] * st["G"][1]))
    log(f"  {label:10s}: n={len(mws)}  最小幅の最大 {max(mws):.3e}  中央値 {np.median(mws):.3e}  |a_R p_R + a_B p_B − a_G p_G| の最大 {max(idents):.2e}")

log("  走査と公式の一致（ランダムな三角形 1000 個）:")
dif = []
for it in range(1000):
    P = list(rng.uniform(-1, 1, (3, 2)))
    dif.append(abs(width_scan(P, 7200) - min_width(P)))
log(f"    |走査 − 2·面積/最長辺| の最大 {max(dif):.2e}（走査の刻み 0.025° 由来の誤差を含む）")

# ---------------------------------------------------------------------------
# 2. g(a) と上限
# ---------------------------------------------------------------------------
log()
log("=== 2. 切片の重心距離 g(a)")


def seg_alpha(a):
    """面積比 a の切片（弦で切った小さい側…a>0.5 では大きい側）の半角 α: (α − sinα cosα)/π = a"""
    return optimize.brentq(lambda al: (al - np.sin(al) * np.cos(al)) / np.pi - a, 1e-12, np.pi - 1e-12)


def g(a):
    if a <= 0:
        return 1.0
    if a >= 1:
        return 0.0
    al = seg_alpha(a)
    return (2.0 / 3.0) * np.sin(al) ** 3 / (al - np.sin(al) * np.cos(al))


for a in [0.25, 1 / 3, 0.5, 2 / 3, 0.75]:
    log(f"  g({a:.4f}) = {g(a):.4f}")

# 最適化用の表（半角 α で媒介変数表示して補間。brentq を毎回呼ばない）
_al = np.linspace(1e-6, np.pi - 1e-6, 200001)
_a_tab = (_al - np.sin(_al) * np.cos(_al)) / np.pi
_g_tab = (2.0 / 3.0) * np.sin(_al) ** 3 / (_al - np.sin(_al) * np.cos(_al))
g_exact = g


def g(a):  # noqa: F811  （以降は表の補間を使う）
    return float(np.interp(a, _a_tab, _g_tab))


log("  表の補間の誤差: " + ", ".join(f"{abs(g(a) - g_exact(a)):.1e}" for a in [0.01, 0.25, 1 / 3, 0.5, 2 / 3, 0.9]))
log(f"  検算: 半円 4/(3π) = {4 / (3 * np.pi):.4f}")

# 浴槽原理の検査: ランダムな濃淡（0〜1）と二値の領域が g を超えないか
log("  浴槽原理の検査（ランダムな領域の重心距離 / g(a)。1 を超えたら上限が破れる）:")
worst = 0.0
for it in range(400):
    kind = it % 3
    if kind == 0:  # 二値の Voronoi の和集合
        lab = voronoi_labels(int(rng.integers(3, 30)))
        keep = rng.random(lab.max() + 1) < rng.uniform(0.1, 0.9)
        t = keep[lab].astype(float)
    elif kind == 1:  # 濃淡（滑らかなランダム場をクリップ）
        f = np.cos(rng.uniform(0, 6) * pts_u + rng.uniform(0, 6) * pts_v + rng.uniform(0, 6)) + rng.normal(0, 0.3, pts_u.size)
        t = np.clip(f * rng.uniform(0.5, 3) + rng.uniform(-1, 1), 0, 1)
    else:  # 切片をわずかに崩したもの（上限の近く）
        th = rng.uniform(0, 2 * np.pi)
        s = rng.uniform(-0.9, 0.9)
        t = ((pts_u * np.cos(th) + pts_v * np.sin(th)) > s).astype(float)
        t = np.clip(t + rng.normal(0, 0.05, t.size), 0, 1)
    a = t.sum() / NIN
    if a < 0.02 or a > 0.98:
        continue
    pc = np.array([(t * pts_u).sum(), (t * pts_v).sum()]) / t.sum()
    worst = max(worst, np.linalg.norm(pc) / g(a))
log(f"    最大の比 {worst:.4f}（格子の離散化で 1 をわずかに超えうる）")

# ---------------------------------------------------------------------------
# 最悪の基線の上限（光量の配分も最適化）
# ---------------------------------------------------------------------------
log()
log("=== 2b. 最悪の基線の上限: 対称解 1.5·g(L) と、光量・角度・半径を自由にした最適化")


def tri_from(params, L, gfull=False):
    if gfull:
        a1, th2, s1, s2 = params
        a2 = 3 * L - 1 - a1
        if not (0 < a2 <= 1 and 0 < a1 <= 1):
            return None
        P = [np.array([s1 * g(a1), 0.0]), np.array([0.0, 0.0]), s2 * g(a2) * np.array([np.cos(th2), np.sin(th2)])]
        return P
    a1, a2, th2, th3, s1, s2, s3 = params
    a3 = 3 * L - a1 - a2
    if not (0 < a3 <= 1):
        return None
    P = [s1 * g(a1) * np.array([1.0, 0.0]), s2 * g(a2) * np.array([np.cos(th2), np.sin(th2)]),
         s3 * g(a3) * np.array([np.cos(th3), np.sin(th3)])]
    return P


def neg_mw(params, L, gfull):
    P = tri_from(params, L, gfull)
    if P is None:
        return 1.0
    return -min_width(P)


rows = []
for L in [0.2, 0.25, 0.3, 1 / 3, 0.4, 0.5, 0.6, 2 / 3, 0.75, 5 / 6]:
    sym = 1.5 * g(L)
    b = [(1e-4, 1), (1e-4, 1), (0, 2 * np.pi), (0, 2 * np.pi), (0, 1), (0, 1), (0, 1)]
    best = 0.0
    bestx = None
    for sd in range(2):
        r = optimize.differential_evolution(neg_mw, b, args=(L, False), seed=sd, popsize=40, maxiter=400, tol=1e-10, polish=True)
        if -r.fun > best:
            best, bestx = -r.fun, r.x
    a3 = 3 * L - bestx[0] - bestx[1]
    line = f"  L={L:.3f}: 対称 1.5·g(L)={sym:.4f}  最適化 {best:.4f}（光量 {bestx[0]:.3f}/{bestx[1]:.3f}/{a3:.3f}）"
    if 3 * L - 1 > 0:
        symG = (np.sqrt(3) / 2) * g((3 * L - 1) / 2)
        bG = [(1e-4, 1), (0, 2 * np.pi), (0, 1), (0, 1)]
        bestG, bxG = 0.0, None
        for sd in range(2):
            r = optimize.differential_evolution(neg_mw, bG, args=(L, True), seed=sd, popsize=40, maxiter=400, tol=1e-10, polish=True)
            if -r.fun > bestG:
                bestG, bxG = -r.fun, r.x
        line += f" | G 全開口: 対称 (√3/2)·g={symG:.4f}  最適化 {bestG:.4f}（R/B 光量 {bxG[0]:.3f}/{3 * L - 1 - bxG[0]:.3f}）"
    log(line)

# ---------------------------------------------------------------------------
# 3. 族 A・族 B
# ---------------------------------------------------------------------------
log()
log("=== 3. 族 A（下に M 扇形）・族 B（下に G 扇形）。格子 1001² と解析式")


def fam_masks(kind, w):
    L_ = (U < 0) & INS
    Rr = (U >= 0) & INS
    s = sec(270.0, w) if w > 0 else np.zeros_like(INS)
    if kind == "A":
        return (L_ | s).astype(float), (INS & ~s).astype(float), (Rr | s).astype(float)
    else:
        return (L_ & ~s).astype(float), INS.astype(float), (Rr & ~s).astype(float)


def summarize(st):
    P = [st[k][1] for k in "RGB"]
    a = [st[k][0] for k in "RGB"]
    xs = [p[0] for p in P]
    ys = [p[1] for p in P]
    S = fisher(st)
    ev = np.linalg.eigvalsh(S)
    return dict(L=np.mean(a), a=a, vert=max(xs) - min(xs), horiz=max(ys) - min(ys), worst=min_width(P), fmin=ev[0], fx=S[0, 0])


log("  族・w | 平均光量 | R/G/B 光量 | 縦の基線 | 横の基線 | 最悪の基線 | Fisher 最小 | 縦の Fisher | 解析 R の x")
for kind in "AB":
    for w in [0, 30, 45, 60, 90, 120, 180]:
        if kind == "A" and w in (45, 180):
            continue
        st = stats_from_masks(*fam_masks(kind, w))
        r = summarize(st)
        wr = np.radians(w)
        if kind == "B":
            ax = -(1 + np.cos(wr / 2)) / 3 / (np.pi / 2 - wr / 4)
        else:
            # 族 A: R = 左半分 ∪ 下の扇形の右側部分 [270, 270+w/2]
            # x モーメント: 左半分 −2/3 ＋ 右側部分 (1/3)[sin θ]_{270}^{270+w/2} = (1/3)(−cos(w/2)·(−1)...) を数値で
            mx = -2 / 3 + (1 / 3) * (np.sin(np.radians(270 + w / 2)) - np.sin(np.radians(270)))
            ar = np.pi / 2 + wr / 4
            ax = mx / ar
        log(f"  {kind}・{w:3d}° | {r['L']:.3f} | {r['a'][0]:.3f}/{r['a'][1]:.3f}/{r['a'][2]:.3f} | {r['vert']:.3f} | {r['horiz']:.3f} | "
            f"{r['worst']:.3f} | {r['fmin']:.4f} | {r['fx']:.4f} | {ax:.4f}（格子 {st['R'][1][0]:.4f}）")

# 閉じた式での族 B（扇形の重心）
log("  族 B の閉じた式（R=扇形 [90°, 270°−w/2]、重心距離 2 sinβ/(3β)、β=半角）:")
for w in [45, 90, 120, 180]:
    beta = np.radians((180 - w / 2) / 2)
    cen = np.radians(90 + (180 - w / 2) / 2)
    d = 2 * np.sin(beta) / (3 * beta)
    pr = d * np.array([np.cos(cen), np.sin(cen)])
    aR = (180 - w / 2) / 360
    P = [pr, np.zeros(2), np.array([-pr[0], pr[1]])]
    pb = (aR * pr + aR * P[2]) / (2 * aR + 1)
    S = aR * np.outer(pr - pb, pr - pb) + 1.0 * np.outer(-pb, -pb) + aR * np.outer(P[2] - pb, P[2] - pb)
    log(f"    w={w}: R 重心 ({pr[0]:.4f}, {pr[1]:.4f})  光量 {(2 * aR + 1) / 3:.4f}  縦 {2 * abs(pr[0]):.4f}  最悪 {min_width(P):.4f}  "
        f"縦の Fisher {S[0, 0]:.4f}  Fisher 最小 {np.linalg.eigvalsh(S)[0]:.4f}")

# ---------------------------------------------------------------------------
# 4. Fisher
# ---------------------------------------------------------------------------
log()
log("=== 4. Fisher（I(n)=Σ a_i((p_i−p̄_a)·n)²）")
cur = stats_from_masks((U < 0) & INS, INS, (U >= 0) & INS)
log(f"  現行 C/Y の縦（n=x）: {fisher(cur)[0, 0]:.4f}  横: {fisher(cur)[1, 1]:.2e}")
rgb = stats_from_masks(sec(210, 120), sec(90, 120), sec(330, 120))
log(f"  RGB 三扇形の最小固有値: {np.linalg.eigvalsh(fisher(rgb))[0]:.4f}")
log(f"  対称解 1.5·a·g(a)² の最大: " + ", ".join(f"a={a:.3f}:{1.5 * a * g(a) ** 2:.4f}" for a in [0.25, 0.28, 0.3, 0.32, 1 / 3, 0.36, 0.4, 0.5, 2 / 3]))


def fisher_from(params, L=None, gfull=False):
    if gfull:
        a1, a2, th2, s1, s2 = params
        A = [a1, 1.0, a2]
        P = [s1 * g(a1) * np.array([1.0, 0]), np.zeros(2), s2 * g(a2) * np.array([np.cos(th2), np.sin(th2)])]
    else:
        a1, a2, a3, th2, th3, s1, s2, s3 = params
        A = [a1, a2, a3]
        P = [s1 * g(a1) * np.array([1.0, 0]), s2 * g(a2) * np.array([np.cos(th2), np.sin(th2)]),
             s3 * g(a3) * np.array([np.cos(th3), np.sin(th3)])]
    A = np.array(A)
    P = np.array(P)
    pb = (A[:, None] * P).sum(0) / A.sum()
    S = sum(A[i] * np.outer(P[i] - pb, P[i] - pb) for i in range(3))
    return S, A


def neg_fmin(params, L, gfull):
    S, A = fisher_from(params, gfull=gfull)
    pen = 0.0
    if L is not None:
        pen = 50 * abs(A.mean() - L)
    return -np.linalg.eigvalsh(S)[0] + pen


b8 = [(1e-3, 1), (1e-3, 1), (1e-3, 1), (0, 2 * np.pi), (0, 2 * np.pi), (0, 1), (0, 1), (0, 1)]
best = (0, None)
for sd in range(2):
    r = optimize.differential_evolution(neg_fmin, b8, args=(None, False), seed=sd, popsize=40, maxiter=400, tol=1e-12)
    if -r.fun > best[0]:
        best = (-r.fun, r.x)
log(f"  Fisher 最小固有値の大域最大（光量自由）: {best[0]:.4f}  光量 {best[1][0]:.3f}/{best[1][1]:.3f}/{best[1][2]:.3f}（平均 {np.mean(best[1][:3]):.3f}）")
for L in [1 / 3, 0.5, 2 / 3]:
    bb = (0, None)
    for sd in range(3):
        r = optimize.differential_evolution(neg_fmin, b8, args=(L, False), seed=sd, popsize=40, maxiter=400, tol=1e-12)
        S, A = fisher_from(r.x)
        v = np.linalg.eigvalsh(S)[0]
        if abs(A.mean() - L) < 2e-3 and v > bb[0]:
            bb = (v, A)
    line = f"  L={L:.3f}: 制約なし {bb[0]:.4f}（光量 {np.round(bb[1], 3) if bb[1] is not None else '-'}・対称 {1.5 * L * g(L) ** 2:.4f}）"
    if 3 * L - 1 > 0:
        b5 = [(1e-3, 1), (1e-3, 1), (0, 2 * np.pi), (0, 1), (0, 1)]
        bg = (0, None)
        for sd in range(3):
            r = optimize.differential_evolution(neg_fmin, b5, args=(L, True), seed=sd, popsize=40, maxiter=400, tol=1e-12)
            S, A = fisher_from(r.x, gfull=True)
            v = np.linalg.eigvalsh(S)[0]
            if abs(A.mean() - L) < 2e-3 and v > bg[0]:
                bg = (v, A)
        line += f" | G 全開口 {bg[0]:.4f}（光量 {np.round(bg[1], 3) if bg[1] is not None else '-'}）"
    log(line)

# 「全方向を取ると得意な向きの情報は必ず減る」の検査: λ_min>0 で縦（最大固有値）が 0.180 を超える設計はあるか
log()
log("=== 4b. 全方向（λ_min>0）を持ちつつ、得意な向きの Fisher（λ_max）が現行の 0.180 を超える設計")
fB45 = stats_from_masks(*fam_masks("B", 45))
S = fisher(fB45)
log(f"  族 B 45°: λ_max={np.linalg.eigvalsh(S)[1]:.4f}  λ_min={np.linalg.eigvalsh(S)[0]:.4f}  最悪の基線 {min_width([fB45[k][1] for k in 'RGB']):.3f}  平均光量 {np.mean([fB45[k][0] for k in 'RGB']):.3f}")


def neg_fmax_with_min(params, eps):
    S, A = fisher_from(params)
    ev = np.linalg.eigvalsh(S)
    return -ev[1] + 100 * max(0.0, eps - ev[0])


for eps in [0.0, 0.01, 0.03, 0.05, 0.08, 0.12]:
    bb = (0, None)
    for sd in range(3):
        r = optimize.differential_evolution(neg_fmax_with_min, b8, args=(eps,), seed=sd, popsize=40, maxiter=400, tol=1e-12)
        S, A = fisher_from(r.x)
        ev = np.linalg.eigvalsh(S)
        if ev[0] >= eps - 1e-4 and ev[1] > bb[0]:
            bb = (ev[1], A, ev[0])
    log(f"  λ_min ≥ {eps:.2f}: λ_max の最大 {bb[0]:.4f}（λ_min {bb[2]:.4f}・光量 {np.round(bb[1], 3)}）")

with open(os.path.join(OUT, "v1_log.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(LOG) + "\n")
