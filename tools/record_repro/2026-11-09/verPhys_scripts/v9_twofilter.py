"""verPhys v9: フィルタ 2 種類（＋不透明）だけの開口は、分光の混信があっても 3 つの重心が一直線か（2026-11-09）

目的: v1 の「Y/C だけなら一直線」を、理想フィルタでなく任意の実効透過率（3×2 の行列・各成分 0〜1）に広げて確かめる。
      式: T_k(x) = α_k·1_P(x) + β_k·1_Q(x) なら p_k = (α_k m_P + β_k m_Q)/(α_k A_P + β_k A_Q) は
      2 点 m_P/A_P と m_Q/A_Q の重み付き平均なので、どの k でもその 2 点を結ぶ直線上にある。
      3 種類なら一般に崩れることも見る。
入力: なし
出力: verPhys_out/v9_log.txt（UTF-8）
仮定: 開口は半径 1 の円。被写体は灰色（チャネルごとの実効透過率がフィルタごとに一定）。
"""
import os
import sys

sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "verPhys_out")
LOG = []


def log(s=""):
    print(s)
    LOG.append(str(s))


rng = np.random.default_rng(1109)
N = 601
g1 = (np.arange(N) + 0.5) / N * 2 - 1
U, V = np.meshgrid(g1, g1[::-1])
INS = U ** 2 + V ** 2 <= 1
pu, pv = U[INS], V[INS]


def min_width(P):
    a, b, c = P
    area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    L = max(np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a))
    return area2 / L if L > 0 else 0.0


for nf in (2, 3):
    mws = []
    for it in range(300):
        seeds = rng.uniform(-1, 1, (int(rng.integers(3, 20)), 2))
        lab = np.argmin((pu[:, None] - seeds[None, :, 0]) ** 2 + (pv[:, None] - seeds[None, :, 1]) ** 2, 1)
        ftype = rng.integers(-1, nf, lab.max() + 1)  # −1 は不透明
        Tm = rng.uniform(0, 1, (3, nf))  # 実効透過率（チャネル × フィルタ）
        P = []
        ok = True
        for k in range(3):
            t = np.where(ftype[lab] >= 0, Tm[k, np.maximum(ftype[lab], 0)], 0.0)
            s = t.sum()
            if s < 50:
                ok = False
                break
            P.append(np.array([(t * pu).sum() / s, (t * pv).sum() / s]))
        if ok:
            mws.append(min_width(P))
    log(f"  フィルタ {nf} 種類＋不透明・透過率は乱数: n={len(mws)}  三角形の最小幅 最大 {max(mws):.2e}  中央値 {np.median(mws):.2e}")

with open(os.path.join(OUT, "v9_log.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(LOG) + "\n")
