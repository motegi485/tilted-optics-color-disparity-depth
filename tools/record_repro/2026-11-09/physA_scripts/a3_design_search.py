# -*- coding: utf-8 -*-
"""設計空間の探索: 開口を扇形（＋中心の円・2 重のリング）に分け、各領域にフィルタを割り当てて全探索し、
平均光量に対する『最も苦手な方向の基線』『Fisher 最小固有値』のパレート前線を、G 全開口の制約あり・なしで出す。
あわせて、領域の形を問わない理論上限（各チャネルの領域を独立に選べる理想フィルタでの上限）を最適化で出す。

入力: なし（argv[1] に "quick" を渡すと大きい構成を省く）
出力: physA_out/a3_front_*.csv（前線の設計）、a3_bound.csv（理論上限）、a3_fronts.png、a3_front_designs.png、
      a3_reps.csv（前線上の代表形状を aperture_lib.evaluate で再評価した指標）
方法:
  - 環状扇形 (r0,r1,φ0,φ1) の面積 (φ1−φ0)(r1²−r0²)/2 と一次モーメント (r1³−r0³)/3·(sinφ1−sinφ0, cosφ0−cosφ1) は解析値。
    理想フィルタでは各チャネルの面積・モーメントは『そのチャネルを通す領域』の和なので、割り当てをベクトル化して全探索する。
  - 平均光量を 0.0025 幅のビンに分け、ビンごとの最大値と設計を保持し、そこから前線（より明るく、より良い設計が無いもの）を取る。
  - 理論上限: 面積 a の領域の重心は中心から g(a)（弦で切った切片の重心距離）より遠くにできない（浴槽原理）。
    理想フィルタでは R・G・B の領域を独立に選べる（各点のフィルタ = その点を含むチャネルの組）ので、
    |p_i| ≤ g(a_i) の下で指標を最大化すれば上限になり、切片の組で達成される。差分進化法で数値的に解く。
仮定: 理想フィルタ・幾何光学・回折なし・ケラレなし・アオリなし。Fisher は光子雑音律速（分散 ∝ 1/a_i）・共通のずれは nuisance。
"""
import csv
import itertools
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aperture_lib as al

OUT = os.path.join(os.path.dirname(HERE), "physA_out")
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
BIN = 0.0025
NB = int(1.0 / BIN) + 2


# ---------------------------------------------------------------------------
# 領域
# ---------------------------------------------------------------------------
def sectors(n, r0=0.0, r1=1.0, offset_deg=0.0):
    w = 360.0 / n
    return [(r0, r1, offset_deg + k * w - w / 2, offset_deg + k * w + w / 2) for k in range(n)]


def region_moments(regions):
    A, M = [], []
    for r0, r1, p0, p1 in regions:
        t0, t1 = np.deg2rad(p0), np.deg2rad(p1)
        A.append(0.5 * (t1 - t0) * (r1 ** 2 - r0 ** 2) / np.pi)
        k = (r1 ** 3 - r0 ** 3) / 3.0 / np.pi
        M.append([k * (np.sin(t1) - np.sin(t0)), k * (np.cos(t0) - np.cos(t1))])
    return np.array(A), np.array(M)


CONFIGS = {
    # 名前: (領域の列, アルファベット)
    "S6_all8": (sectors(6), "WYCMRGBO"),
    "S8_all8": (sectors(8), "WYCMRGBO"),
    "C030_S6_all8": ([(0, 0.3, -180, 180)] + sectors(6, 0.3, 1.0), "WYCMRGBO"),
    "C050_S6_all8": ([(0, 0.5, -180, 180)] + sectors(6, 0.5, 1.0), "WYCMRGBO"),
    "C070_S6_all8": ([(0, 0.7, -180, 180)] + sectors(6, 0.7, 1.0), "WYCMRGBO"),
    "S12_WYCM": (sectors(12), "WYCM"),
    "S12_WYCG": (sectors(12), "WYCG"),
    "R2S6_WYCG": (sectors(6, 0.0, 0.6) + sectors(6, 0.6, 1.0), "WYCG"),
    "C050_S8_all8": ([(0, 0.5, -180, 180)] + sectors(8, 0.5, 1.0), "WYCMRGBO"),
}


# ---------------------------------------------------------------------------
# 指標（ベクトル化）
# ---------------------------------------------------------------------------
def metrics_vec(A, M):
    """A (n,3) 開口比、M (n,3,2) 一次モーメント（開口比×長さ）。最小幅と Fisher 最小固有値を返す。"""
    valid = A > 1e-9
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(valid[..., None], M / np.where(valid, A, 1.0)[..., None], 0.0)
    v1 = p[:, 1] - p[:, 0]
    v2 = p[:, 2] - p[:, 0]
    cr = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    s01 = np.hypot(*v1.T)
    s02 = np.hypot(*v2.T)
    s12 = np.hypot(*(p[:, 2] - p[:, 1]).T)
    longest = np.maximum(np.maximum(s01, s02), s12)
    mw = np.where(valid.all(1) & (longest > 1e-12), cr / np.where(longest > 0, longest, 1), 0.0)
    # Fisher: S = Σ m m^T / a − (Σm)(Σm)^T / Σa
    inv = np.where(valid, 1.0 / np.where(valid, A, 1.0), 0.0)
    Sxx = (M[..., 0] ** 2 * inv).sum(1)
    Syy = (M[..., 1] ** 2 * inv).sum(1)
    Sxy = (M[..., 0] * M[..., 1] * inv).sum(1)
    tot = A.sum(1)
    Mx = M[..., 0].sum(1)
    My = M[..., 1].sum(1)
    tt = np.where(tot > 0, tot, 1.0)
    Sxx -= Mx * Mx / tt
    Syy -= My * My / tt
    Sxy -= Mx * My / tt
    tr2 = 0.5 * (Sxx + Syy)
    lmin = tr2 - np.sqrt(np.maximum(0.25 * (Sxx - Syy) ** 2 + Sxy ** 2, 0.0))
    return mw, np.maximum(lmin, 0.0)


class Best:
    """ビンごとの最大値と設計コードを保持"""

    def __init__(self):
        self.val = np.full(NB, -1.0)
        self.light = np.full(NB, np.nan)
        self.code = np.full((NB, 2), -1, dtype=np.int64)

    def update(self, light, val, pre, suf_idx, mask=None):
        if mask is not None:
            light, val, suf_idx = light[mask], val[mask], suf_idx[mask]
        if light.size == 0:
            return
        b = np.clip((light / BIN).astype(np.int64), 0, NB - 1)
        order = np.lexsort((-val, b))
        bu, first = np.unique(b[order], return_index=True)
        idx = order[first]
        better = val[idx] > self.val[bu]
        bu, idx = bu[better], idx[better]
        self.val[bu] = val[idx]
        self.light[bu] = light[idx]
        self.code[bu, 0] = pre
        self.code[bu, 1] = suf_idx[idx]

    def front(self):
        ok = self.val >= 0
        L, V, C = self.light[ok], self.val[ok], self.code[ok]
        o = np.argsort(-L)
        L, V, C = L[o], V[o], C[o]
        keep, best = [], -1.0
        for i in range(len(L)):
            if V[i] > best + 1e-9:
                keep.append(i)
                best = V[i]
        return L[keep], V[keep], C[keep]


def search(name, regions, alpha, msuf=6, tv=None, gfull_th=1 - 1e-9):
    """tv: フィルタごとの (R,G,B) 実効透過率 (F,3)。None なら理想フィルタ（a4_spectral.py が実フィルタの近似を渡す）"""
    A_k, M_k = region_moments(regions)
    K = len(regions)
    F = len(alpha)
    if tv is None:
        tv = np.array([al.FILTERS[f] for f in alpha])  # (F,3)
    tv = np.asarray(tv, dtype=float)
    cA = A_k[:, None, None] * tv[None]  # (K,F,3)
    cM = M_k[:, None, None, :] * tv[None, :, :, None]  # (K,F,3,2)
    m = min(K, msuf)
    pre_k = list(range(K - m))
    suf_k = list(range(K - m, K))
    SA = np.zeros((1, 3))
    SM = np.zeros((1, 3, 2))
    for k in suf_k:
        SA = (SA[:, None] + cA[k][None]).reshape(-1, 3)
        SM = (SM[:, None] + cM[k][None]).reshape(-1, 3, 2)
    suf_idx = np.arange(SA.shape[0])
    best = {(met, con): Best() for met in ("mw", "fi") for con in ("all", "gfull")}
    t0 = time.time()
    n_eval = 0
    for pre, digits in enumerate(itertools.product(range(F), repeat=len(pre_k))):
        pA = sum((cA[k, d] for k, d in zip(pre_k, digits)), np.zeros(3))
        pM = sum((cM[k, d] for k, d in zip(pre_k, digits)), np.zeros((3, 2)))
        A = SA + pA
        M = SM + pM
        light = A.mean(1)
        mw, fi = metrics_vec(A, M)
        gfull = A[:, 1] > gfull_th
        best[("mw", "all")].update(light, mw, pre, suf_idx)
        best[("fi", "all")].update(light, fi, pre, suf_idx)
        best[("mw", "gfull")].update(light, mw, pre, suf_idx, gfull)
        best[("fi", "gfull")].update(light, fi, pre, suf_idx, gfull)
        n_eval += A.shape[0]
    print(f"[{name}] 領域 {K}・フィルタ {alpha}・評価 {n_eval:,} 通り・{time.time() - t0:.1f} s")
    return best, (pre_k, suf_k, F)


def decode(code, pre_k, suf_k, F, alpha):
    pre, s = int(code[0]), int(code[1])
    K = len(pre_k) + len(suf_k)
    digits = [0] * K
    pd = np.unravel_index(pre, [F] * len(pre_k)) if pre_k else []
    for k, d in zip(pre_k, pd):
        digits[k] = int(d)
    sd = np.unravel_index(s, [F] * len(suf_k))
    for k, d in zip(suf_k, sd):
        digits[k] = int(d)
    return "".join(alpha[d] for d in digits)


def design_from_assignment(regions, assign):
    def f(x, y):
        r = np.sqrt(x * x + y * y)
        lay = []
        for (r0, r1, p0, p1), ch in zip(regions, assign):
            m = (r >= r0) & (r < r1) & al.in_sector(x, y, 0.5 * (p0 + p1), p1 - p0)
            lay.append((m, ch))
        return al.paint(x, y, lay)
    return f


# ---------------------------------------------------------------------------
# 理論上限
# ---------------------------------------------------------------------------
_AG = np.linspace(0, 1, 4001)
_GG = al.segment_centroid_dist(_AG)


def gfun(a):
    return np.interp(a, _AG, _GG)


def bound_obj(x, L, metric, gfull):
    if gfull:
        aR, phB, tR, tB = x
        aG = 1.0
        aB = 3 * L - 1 - aR
        a = np.array([aR, aG, aB])
        if aB < 0 or aB > 1:
            return 10 + abs(aB - np.clip(aB, 0, 1))
        p = np.array([[tR * gfun(aR), 0.0], [0.0, 0.0],
                      [tB * gfun(aB) * np.cos(phB), tB * gfun(aB) * np.sin(phB)]])
    else:
        aR, aG, phG, phB, tR, tG, tB = x
        aB = 3 * L - aR - aG
        if aB < 0 or aB > 1:
            return 10 + abs(aB - np.clip(aB, 0, 1))
        a = np.array([aR, aG, aB])
        p = np.array([[tR * gfun(aR), 0.0],
                      [tG * gfun(aG) * np.cos(phG), tG * gfun(aG) * np.sin(phG)],
                      [tB * gfun(aB) * np.cos(phB), tB * gfun(aB) * np.sin(phB)]])
    if metric == "mw":
        return -al.tri_min_width(p, a > 1e-9)
    S = al.fisher_matrix(a, p)
    return -al.eig2(S)[0]


def theoretical_bound(metric, gfull, Ls, seed=0):
    out = []
    for L in Ls:
        if gfull:
            bounds = [(0, 1), (0, 2 * np.pi), (0, 1), (0, 1)]
        else:
            bounds = [(0, 1), (0, 1), (0, 2 * np.pi), (0, 2 * np.pi), (0, 1), (0, 1), (0, 1)]
        r = differential_evolution(bound_obj, bounds, args=(L, metric, gfull), seed=seed, popsize=20,
                                   maxiter=300, tol=1e-9, polish=True)
        out.append((L, -r.fun, r.x))
    return out


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    results = {}
    for name, (regions, alpha) in CONFIGS.items():
        if quick and name in ("C050_S8_all8",):
            continue
        msuf = 6 if len(alpha) == 8 else 8
        best, meta = search(name, regions, alpha, msuf)
        results[name] = (regions, alpha, best, meta)
    # 理論上限
    Ls_all = np.round(np.arange(0.05, 0.951, 0.025), 4)
    Ls_g = np.round(np.arange(0.35, 0.951, 0.025), 4)
    bound = {}
    t0 = time.time()
    for met in ("mw", "fi"):
        bound[(met, "all")] = theoretical_bound(met, False, Ls_all)
        bound[(met, "gfull")] = theoretical_bound(met, True, Ls_g)
    print(f"理論上限の最適化 {time.time() - t0:.1f} s")
    with open(os.path.join(OUT, "a3_bound.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "constraint", "light_mean", "value", "x"])
        for (met, con), lst in bound.items():
            for L, v, x in lst:
                w.writerow([met, con, L, v, " ".join(f"{e:.4f}" for e in x)])
    # 等光量の切片 3 つ（120° おき）の解析値: 最小幅 1.5 g(L)、Fisher 1.5 L g(L)²
    print("等光量の切片 3 つ（120°）: L, 最小幅 1.5g, Fisher 1.5Lg² | 数値の上限（最小幅, Fisher）")
    for i, L in enumerate(Ls_all):
        if abs((L * 40) % 4) < 1e-6 or L in (0.5, 0.65):
            g = gfun(L)
            print(f"  {L:.3f}  {1.5 * g:.4f}  {1.5 * L * g * g:.4f} | {bound[('mw', 'all')][i][1]:.4f}  {bound[('fi', 'all')][i][1]:.4f}")
    print("G 全開口の上限: L, 最小幅, Fisher, 最小幅の最適解 (aR, φB[deg], tR, tB)")
    for i, L in enumerate(Ls_g):
        x = bound[("mw", "gfull")][i][2]
        print(f"  {L:.3f}  {bound[('mw', 'gfull')][i][1]:.4f}  {bound[('fi', 'gfull')][i][1]:.4f}  "
              f"aR={x[0]:.3f} aB={3 * L - 1 - x[0]:.3f} φB={np.degrees(x[1]) % 360:.1f} tR={x[2]:.3f} tB={x[3]:.3f}")
    # 前線の CSV
    for name, (regions, alpha, best, (pre_k, suf_k, F)) in results.items():
        with open(os.path.join(OUT, f"a3_front_{name}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["metric", "constraint", "light_mean", "value", "assignment"])
            for (met, con), b in best.items():
                L, V, C = b.front()
                for l, v, c in zip(L, V, C):
                    w.writerow([met, con, f"{l:.5f}", f"{v:.5f}", decode(c, pre_k, suf_k, F, alpha)])
    # 図: 前線
    import csv as _csv
    cat = []
    cpath = os.path.join(OUT, "a2_catalog.csv")
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as fh:
            cat = list(_csv.DictReader(fh))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {"S6_all8": "#2a78d6", "S8_all8": "#eb6834", "C050_S6_all8": "#1baf7a", "S12_WYCM": "#eda100",
              "S12_WYCG": "#e87ba4", "R2S6_WYCG": "#6250d6", "C050_S8_all8": "#008300"}
    for ax, met, ttl in zip(axs, ("mw", "fi"), ("最も苦手な方向の基線（三角形の最小幅）", "Fisher 情報の最小固有値（光量で重み付け）")):
        for con, ls in (("all", "-"), ("gfull", "--")):
            Lb = [e[0] for e in bound[(met, con)]]
            Vb = [e[1] for e in bound[(met, con)]]
            ax.plot(Lb, Vb, color="#0b0b0b", ls=ls, lw=2.2,
                    label=f"理論上限（{'制約なし' if con == 'all' else 'G 全開口'}）")
        for name, (regions, alpha, best, meta) in results.items():
            if name not in colors:
                continue
            for con, ls in (("all", "-"), ("gfull", "--")):
                L, V, C = best[(met, con)].front()
                if len(L) == 0:
                    continue
                o = np.argsort(L)
                # 前線の値 V_i は『光量 L_i 以下ならこの値が取れる』ので、区間 (L_{i-1}, L_i] に描く（where="pre"）
                ax.step(L[o], V[o], where="pre", color=colors[name], ls=ls, lw=1.4,
                        label=f"{name}（{'制約なし' if con == 'all' else 'G 全開口'}）")
        if cat:
            for row in cat:
                x = float(row["light_mean"])
                y = float(row["worst"] if met == "mw" else row["fisher_min"])
                gf = row["G_full"] == "True"
                ax.plot(x, y, "s" if gf else "o", ms=6, mfc="white", mec="#52514e", mew=1.3)
                if row["key"] in ("cy_lr", "rgb3", "cmy3", "ycg3", "quad", "wheel6", "ywcg4_v", "par_hex", "bando", "mca",
                                  "cy_m90", "ycw3", "w_top_cy"):
                    ax.annotate(row["label"], (x, y), fontsize=6.5, xytext=(3, 3), textcoords="offset points", color="#52514e")
        ax.set_xlabel("平均光量（開口全体に対する比・3 チャネルの平均）")
        ax.set_ylabel(ttl)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=6.3, loc="upper right", ncol=1)
    fig.suptitle("平均光量と全方向の基線のパレート前線（理想フィルタ・幾何光学・アオリなし。□=G 全開口のカタログ形状、○=それ以外）", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "a3_fronts.png"), dpi=110)
    plt.close(fig)
    # 代表形状（前線上）: 指定光量に最も近い前線点
    grid = al.Grid(800)
    reps = []
    targets = {"all": [0.333, 0.5, 0.6, 0.667, 0.75, 0.833], "gfull": [0.5, 0.6, 0.667, 0.75, 0.833]}
    for name in ("S6_all8", "S8_all8", "C050_S6_all8", "S12_WYCG", "R2S6_WYCG", "S12_WYCM"):
        if name not in results:
            continue
        regions, alpha, best, (pre_k, suf_k, F) = results[name]
        for met in ("mw", "fi"):
            for con, tl in targets.items():
                L, V, C = best[(met, con)].front()
                if len(L) == 0:
                    continue
                for t in tl:
                    cand = np.where(L <= t + 1e-6)[0]
                    if len(cand) == 0:
                        continue
                    i = cand[np.argmax(L[cand])]  # 目標以下で最も明るい前線点
                    asg = decode(C[i], pre_k, suf_k, F, alpha)
                    reps.append(dict(config=name, metric=met, constraint=con, target=t, light=L[i], value=V[i],
                                     assignment=asg, regions=regions))
    # 重複を除いて再評価
    seen = {}
    for r in reps:
        k = (r["config"], r["assignment"])
        if k in seen:
            continue
        f = design_from_assignment(r["regions"], r["assignment"])
        e = al.evaluate(f, grid, lsf_step_deg=3.0)
        seen[k] = (f, e)
    with open(os.path.join(OUT, "a3_reps.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "metric", "constraint", "target", "assignment", "light_mean", "a_R", "a_G", "a_B",
                    "vert", "horiz", "worst", "fisher_min", "Iw_min", "zworst", "zvert", "ratio_med", "ncc_best_min",
                    "sim2d_RG", "sim2d_RB", "sim2d_GB"])
        for r in reps:
            f, e = seen[(r["config"], r["assignment"])]
            w.writerow([r["config"], r["metric"], r["constraint"], r["target"], r["assignment"], f"{e['light_mean']:.4f}",
                        *[f"{v:.4f}" for v in e["a"]], f"{e['vert']:.4f}", f"{e['horiz']:.4f}", f"{e['worst']:.4f}",
                        f"{e['fisher_min']:.5f}", f"{e['Iw_min']:.5f}", f"{e['zworst']:.4f}", f"{e['zvert']:.4f}",
                        f"{e['ratio_med']:.3f}", f"{e['ncc_best_min']:.3f}", *[f"{v:.3f}" for v in e["sim2d"]]])
            print(f"{r['config']:13s} {r['metric']} {r['constraint']:5s} 目標{r['target']:.3f} {r['assignment']:14s} 光量 {e['light_mean']:.3f} "
                  f"G {e['a'][1]:.3f} 縦 {e['vert']:.3f} 横 {e['horiz']:.3f} 最悪 {e['worst']:.3f} Fisher {e['fisher_min']:.4f} "
                  f"xcorr最悪 {e['zworst']:.3f} 2D形状 {np.nanmin(e['sim2d']):.3f}")
    # 図: 代表形状
    sel = [r for r in reps if r["config"] in ("S6_all8", "C050_S6_all8", "S12_WYCG") and r["metric"] == "mw"]
    uniq = []
    for r in sel:
        if (r["config"], r["assignment"]) not in [(u["config"], u["assignment"]) for u in uniq]:
            uniq.append(r)
    uniq = uniq[:24]
    cols = 6
    rows_ = int(np.ceil(len(uniq) / cols))
    fig, axs = plt.subplots(rows_, cols, figsize=(cols * 2.5, rows_ * 2.9))
    for ax, r in zip(np.atleast_1d(axs).ravel(), uniq):
        f, e = seen[(r["config"], r["assignment"])]
        ax.imshow(al.render_rgb(f), extent=[-1, 1, -1, 1])
        p = e["p"]
        for c, col in zip(range(3), ["#ff3030", "#30c030", "#3060ff"]):
            if e["valid"][c]:
                ax.plot(p[c, 0], p[c, 1], "o", ms=6, mfc=col, mec="black")
        ax.set_title(f"{r['config']}・{'G全' if r['constraint'] == 'gfull' else '制約なし'}\n光量 {e['light_mean']:.2f}・最悪 {e['worst']:.2f}・縦 {e['vert']:.2f}",
                     fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in np.atleast_1d(axs).ravel()[len(uniq):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "a3_front_designs.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
