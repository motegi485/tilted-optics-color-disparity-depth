"""verPhys v1b: 最悪の基線の上限 1.5·g(L) が、低い光量でいつ破れるか（2026-11-09）

目的: v1 で L=0.2 では光量を不均等（2 チャネルがほぼ 0）にした方が上限を超えた。どの L から破れるか、
      各チャネルに最低光量 a_min を課すとどうなるかを確かめる。
入力: なし
出力: verPhys_out/v1b_log.txt
"""
import os
import sys

sys.dont_write_bytecode = True
import numpy as np
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "verPhys_out")
LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(str(s))


_al = np.linspace(1e-6, np.pi - 1e-6, 200001)
_a = (_al - np.sin(_al) * np.cos(_al)) / np.pi
_g = (2.0 / 3.0) * np.sin(_al) ** 3 / (_al - np.sin(_al) * np.cos(_al))


def g(a):
    return float(np.interp(a, _a, _g))


def mw(P):
    a, b, c = P
    area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    L = max(np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a))
    return area2 / L if L > 0 else 0.0


def neg(params, L, amin):
    a1, a2, th2, th3, s1, s2, s3 = params
    a3 = 3 * L - a1 - a2
    if not (amin <= a3 <= 1) or a1 < amin or a2 < amin:
        return 1.0
    P = [s1 * g(a1) * np.array([1.0, 0]), s2 * g(a2) * np.array([np.cos(th2), np.sin(th2)]), s3 * g(a3) * np.array([np.cos(th3), np.sin(th3)])]
    return -mw(P)


for amin in (1e-4, 0.05, 0.1):
    log(f"--- 各チャネルの最低光量 {amin}")
    for L in (0.15, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3):
        b = [(amin, 1), (amin, 1), (0, 2 * np.pi), (0, 2 * np.pi), (0, 1), (0, 1), (0, 1)]
        best, bx = 0, None
        for sd in range(3):
            r = optimize.differential_evolution(neg, b, args=(L, amin), seed=sd, popsize=40, maxiter=400, tol=1e-10)
            if -r.fun > best:
                best, bx = -r.fun, r.x
        log(f"  L={L:.2f}: 対称 {1.5 * g(L):.4f}  最適化 {best:.4f}（比 {best / (1.5 * g(L)):.3f}・光量 {bx[0]:.3f}/{bx[1]:.3f}/{3 * L - bx[0] - bx[1]:.3f}）")

with open(os.path.join(OUT, "v1b_log.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(LOG) + "\n")
