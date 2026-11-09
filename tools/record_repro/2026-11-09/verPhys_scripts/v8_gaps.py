"""verPhys v8: v1〜v7 で残った確認（2026-11-09・再開後に追加）

目的:
  A. 「全方向を取ると得意な向きの情報は必ず減る」の反例が、平均光量 2/3（現行と同じ）や G 全開口を保っても成り立つか。
     λ_min ≥ ε を課して λ_max（得意な向きの Fisher）の最大を差分進化で求める。
  B. アオリ θ=30° の縦の縮み（cosθ）で、最悪の向きの基線と Fisher 最小固有値がどれだけ下がるか（開口の向きを回して最良にした値も）。
  C. 現行 C/Y＋color-line／PSF 合わせで、バーコードの法線 φ=75〜90° のどこで測れなくなるか（critic が未測定と指摘した境目）。
  D. 現行 C/Y の color-line が横向きの筋の強い被写体で示す偏り（c=3 で −14 %、c=6 で +16 %）は c でどう変わるか。
     対照として RGB 三扇形・族 B 90° でも同じ被写体の偏りを見る。
入力: なし（vp_lib.py を import）
出力: verPhys_out/v8_log.txt（UTF-8）
仮定: vp_lib.py・v1_principles.py と同じ（理想フィルタ・幾何光学・192×192・σ=1.5）。
"""
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy import optimize

import vp_lib as vp

OUT = os.path.join(os.path.dirname(HERE), "verPhys_out")
LOG = []
t0 = time.time()


def log(s=""):
    print(s, flush=True)
    LOG.append(str(s))


def flush():
    with open(os.path.join(OUT, "v8_log.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(LOG) + "\n")


# ---------------------------------------------------------------------------
# g(a) の表（v1 と同じ媒介変数表示）
# ---------------------------------------------------------------------------
_al = np.linspace(1e-6, np.pi - 1e-6, 200001)
_a_tab = (_al - np.sin(_al) * np.cos(_al)) / np.pi
_g_tab = (2.0 / 3.0) * np.sin(_al) ** 3 / (_al - np.sin(_al) * np.cos(_al))


def g(a):
    return float(np.interp(a, _a_tab, _g_tab))


def fisher_S(A, P):
    A = np.asarray(A, float)
    P = np.asarray(P, float)
    pb = (A[:, None] * P).sum(0) / A.sum()
    return sum(A[i] * np.outer(P[i] - pb, P[i] - pb) for i in range(len(A)))


def min_width(P):
    a, b, c = [np.asarray(x, float) for x in P]
    area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    L = max(np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a))
    return area2 / L if L > 0 else 0.0


def max_width(P):
    a, b, c = [np.asarray(x, float) for x in P]
    return max(np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a))


# ---------------------------------------------------------------------------
# A. 光量の制約つきの λ_max と λ_min の両立
# ---------------------------------------------------------------------------
log("=== A. 平均光量の下限 L0 と G 全開口を課したときの、λ_min ≥ ε での λ_max の最大（現行の縦は 0.1801）")


def params_to(params, gfull):
    if gfull:
        a1, a3, th3, s1, s3 = params
        A = [a1, 1.0, a3]
        P = [s1 * g(a1) * np.array([1.0, 0.0]), np.zeros(2), s3 * g(a3) * np.array([np.cos(th3), np.sin(th3)])]
    else:
        a1, a2, a3, th2, th3, s1, s2, s3 = params
        A = [a1, a2, a3]
        P = [s1 * g(a1) * np.array([1.0, 0.0]), s2 * g(a2) * np.array([np.cos(th2), np.sin(th2)]),
             s3 * g(a3) * np.array([np.cos(th3), np.sin(th3)])]
    return np.array(A), np.array(P)


def obj(params, eps, L0, gfull):
    A, P = params_to(params, gfull)
    ev = np.linalg.eigvalsh(fisher_S(A, P))
    pen = 100 * max(0.0, eps - ev[0]) + 100 * max(0.0, L0 - A.mean())
    return -ev[1] + pen


b8 = [(1e-3, 1), (1e-3, 1), (1e-3, 1), (0, 2 * np.pi), (0, 2 * np.pi), (0, 1), (0, 1), (0, 1)]
b5 = [(1e-3, 1), (1e-3, 1), (0, 2 * np.pi), (0, 1), (0, 1)]
for L0, gfull, lab in [(2 / 3, False, "光量≥2/3"), (2 / 3, True, "光量≥2/3・G 全開口"), (0.5, True, "光量≥1/2・G 全開口")]:
    for eps in [0.0, 0.004, 0.01, 0.02, 0.04]:
        best = None
        for sd in range(3):
            r = optimize.differential_evolution(obj, b5 if gfull else b8, args=(eps, L0, gfull), seed=sd, popsize=40,
                                                maxiter=500, tol=1e-12)
            A, P = params_to(r.x, gfull)
            ev = np.linalg.eigvalsh(fisher_S(A, P))
            if ev[0] >= eps - 1e-4 and A.mean() >= L0 - 1e-3 and (best is None or ev[1] > best[0]):
                best = (ev[1], ev[0], A, min_width(P), max_width(P))
        if best is None:
            log(f"  {lab:18s} ε={eps:.3f}: 解なし")
        else:
            log(f"  {lab:18s} ε={eps:.3f}: λ_max {best[0]:.4f}（λ_min {best[1]:.4f}）光量 {np.round(best[2], 3)} 平均 {best[2].mean():.3f}"
                f"  最長辺 {best[4]:.3f}・最小幅 {best[3]:.3f}")
    flush()

# 参考: 族 B の各 w の (λ_max, λ_min, 平均光量)（閉じた式）
log("  参考（族 B の閉じた式）:")
for w in [30, 45, 60, 90]:
    beta = np.radians((180 - w / 2) / 2)
    cen = np.radians(90 + (180 - w / 2) / 2)
    d = 2 * np.sin(beta) / (3 * beta)
    pr = d * np.array([np.cos(cen), np.sin(cen)])
    aR = (180 - w / 2) / 360
    S = fisher_S([aR, 1.0, aR], [pr, np.zeros(2), np.array([-pr[0], pr[1]])])
    ev = np.linalg.eigvalsh(S)
    log(f"    w={w:3d}°: λ_max {ev[1]:.4f} λ_min {ev[0]:.4f} 平均光量 {(2 * aR + 1) / 3:.3f}")
flush()

# ---------------------------------------------------------------------------
# B. アオリ θ=30° の縦の縮みでの最悪の基線・Fisher（開口の向きを回して最良にした値）
# ---------------------------------------------------------------------------
log()
log("=== B. 縦の縮み cos30°=0.866 での最悪の基線（三角形の最小幅）と Fisher 最小固有値。β は開口の回転")


def hue6(u, v):
    a = vp._ang(u, v)
    # 色相環 6 分割: R・G・B がそれぞれ 120° ずつずれた半円（R は 0°〜180° を中心 90° に…の代わりに中心 90/210/330 の半円）
    return {"R": vp._sec(a, 90, 180) * 1.0, "G": vp._sec(a, 210, 180) * 1.0, "B": vp._sec(a, 330, 180) * 1.0}


shapes = {"current": vp.ap_current, "rgb3": vp.ap_rgb3, "cmy3": vp.ap_cmy3, "famB90": vp.make_famB(90.0),
          "famB180": vp.make_famB(180.0), "hue6": hue6}
sy = np.cos(np.radians(30))
for name, fn in shapes.items():
    m = vp.moments(fn, n=1201)
    A = np.array([m[k][0] for k in vp.CH])
    P0 = np.array([[m[k][1], m[k][2]] for k in vp.CH])
    res = []
    for beta in np.radians(np.arange(0, 180, 0.5)):
        Rm = np.array([[np.cos(beta), -np.sin(beta)], [np.sin(beta), np.cos(beta)]])
        P = (Rm @ P0.T).T * np.array([1.0, sy])
        res.append((min_width(P), np.linalg.eigvalsh(fisher_S(A, P))[0], beta))
    mw0 = min_width(P0)
    f0 = np.linalg.eigvalsh(fisher_S(A, P0))[0]
    r_def = res[0]
    r_best = max(res, key=lambda x: x[0])
    log(f"  {name:8s}: 縮みなし 最小幅 {mw0:.3f}・Fisher 最小 {f0:.4f} | 縮み・既定の向き {r_def[0]:.3f}（{r_def[0] / mw0 if mw0 > 0 else 0:.3f} 倍）"
        f"・{r_def[1]:.4f} | 縮み・最良の向き β={np.degrees(r_best[2]):.1f}° {r_best[0]:.3f}（{r_best[0] / mw0 if mw0 > 0 else 0:.3f} 倍）・{r_best[1]:.4f}")
flush()

# ---------------------------------------------------------------------------
# C. 現行 C/Y の境目（φ=75〜90°）
# ---------------------------------------------------------------------------
log()
log("=== C. 現行 C/Y・バーコードの法線 φ（0°=縦エッジ）ごとの良好率（raw／校正）・偏り・ばらつき。種 2 つ")
H = W = 192
SS = 4
M = 24
CANDS = np.arange(-24, 25, 1.0)
CANDS4 = np.arange(-20, 21, 1.0)
fn = vp.ap_current
q = vp.qvec(fn)


def calib(fn_, q_, c, methods):
    acc = {mm: [] for mm in methods}
    for sd in (901, 902):
        subj = vp.subj_1overf(H + 2 * M, W + 2 * M, SS, np.random.default_rng(sd))
        E, _, _ = vp.eval_mask(subj, c, SS, margin=M)
        u8 = vp.to8(vp.render(subj, fn_, c, SS, margin=M), 1.5, np.random.default_rng(sd + 7))
        for mm in methods:
            e = vp.est_colorline(u8, q_, CANDS) if mm == "cl3" else vp.est_psfmatch(u8, fn_, CANDS4)
            v = e[E]
            acc[mm].append(v[np.isfinite(v)] / c)
    return {mm: float(np.median(np.concatenate(v))) for mm, v in acc.items()}


for c in (3.0, 6.0, 12.0):
    k = calib(fn, q, c, ("cl3", "iv"))
    for phi in (75.0, 80.0, 84.0, 87.0, 89.0, 90.0):
        out = {}
        for mm in ("cl3", "iv"):
            raw, cal = [], []
            for sd in (1, 2):
                subj = vp.subj_barcode(H + 2 * M, W + 2 * M, SS, np.random.default_rng(700 + sd + int(phi)), phi)
                E, _, _ = vp.eval_mask(subj, c, SS, margin=M)
                u8 = vp.to8(vp.render(subj, fn, c, SS, margin=M), 1.5, np.random.default_rng(800 + sd + int(c)))
                e = vp.est_colorline(u8, q, CANDS) if mm == "cl3" else vp.est_psfmatch(u8, fn, CANDS4)
                raw.append(vp.evaluate(e, c, E))
                cal.append(vp.evaluate(e / k[mm], c, E))
            n = sum(x["n"] for x in raw)
            out[mm] = (sum(x["good"] * x["n"] for x in raw) / n, sum(x["good"] * x["n"] for x in cal) / n,
                       np.nanmean([x["bias"] for x in raw]), np.nanmean([x["sd"] for x in raw]))
        h = np.cos(np.radians(phi)) ** 2
        log(f"  c={c:4.1f} φ={phi:4.1f}°（h=cos²φ={h:.4f}）: color-line 良好 {out['cl3'][0]:.2f}／{out['cl3'][1]:.2f} 偏り {out['cl3'][2]:+.2f} ばらつき {out['cl3'][3]:.2f}"
            f" | PSF 合わせ 良好 {out['iv'][0]:.2f}／{out['iv'][1]:.2f} 偏り {out['iv'][2]:+.2f} ばらつき {out['iv'][3]:.2f}  ({time.time() - t0:.0f}s)")
    flush()

# ---------------------------------------------------------------------------
# D. 横向きの筋の強い被写体での color-line の偏りの c 依存
# ---------------------------------------------------------------------------
log()
log("=== D. 強い異方性の 1/f（σ_ψ=4°・床 0.03）での color-line の偏り（raw・中央値 / c）と良好率。種 2 つ")
for name in ("current", "rgb3", "famB90"):
    fn_ = vp.SHAPES[name]
    q_ = vp.qvec(fn_)
    parts = []
    for c in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        bs, gs, n = [], 0.0, 0
        for sd in (1, 2):
            subj = vp.subj_1overf(H + 2 * M, W + 2 * M, SS, np.random.default_rng(210 + sd), aniso_deg=4.0, floor=0.03)
            E, _, _ = vp.eval_mask(subj, c, SS, margin=M)
            u8 = vp.to8(vp.render(subj, fn_, c, SS, margin=M), 1.5, np.random.default_rng(sd + 50 + int(c)))
            ev = vp.evaluate(vp.est_colorline(u8, q_, CANDS), c, E)
            bs.append(ev["bias"])
            gs += ev["good"] * ev["n"]
            n += ev["n"]
        parts.append(f"c={c:g}: {100 * np.mean(bs) / c:+.0f}%（良好 {gs / n:.2f}）")
    log(f"  {name:8s} " + " ; ".join(parts) + f"  ({time.time() - t0:.0f}s)")
    flush()

log(f"done {time.time() - t0:.0f}s")
flush()
