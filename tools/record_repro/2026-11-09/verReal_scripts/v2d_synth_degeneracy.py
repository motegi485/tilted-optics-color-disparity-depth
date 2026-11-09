# -*- coding: utf-8 -*-
"""physB の合成器（apsim.py を import するだけ・変更しない）で、現行 C/Y の color-line の det が c=0（ずらさない）で
どれだけ小さくなるかを見る。理想フィルタでは G = (R+B)/2 なので、ずらさない候補で 3 チャネルが線形従属になる。
実データの「det の最小が d=0 に潰れる」現象が合成でも出るか（雑音・G の重みを変えて）を確かめる。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PB = os.path.join(os.path.dirname(HERE), "physB_scripts")
sys.path.insert(0, PB)
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

import apsim as A

OUT = os.path.join(os.path.dirname(HERE), "verReal_out")
log = open(os.path.join(OUT, "v2d_synth_degeneracy.txt"), "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")


H = W = 192
ss = 4
rng = np.random.default_rng(9001)
sc = A.Scene(A.scene_1overf(H, W, ss, rng), ss, "iso")
st = A.shape_stats(A.shape_cy_lr)
pc = A.centroid_img(st)
roi = (slice(40, 152), slice(40, 152))
cands = np.arange(-14, 15, 1.0)
for c_true in (6.0, -6.0):
    lin = A.render(sc, A.shape_cy_lr, c_true)
    for gw, label in ((None, "理想 G=(R+B)/2"), (0.7, "G=0.7·Y半+0.3·C半（synth.py の重み）")):
        L2 = dict(lin)
        if gw is not None:
            # G だけ Y:C=gw:(1−gw) で描き直す（R=Y 半・B=C 半の PSF の重み付き和）
            L2["G"] = gw * lin["R"] + (1 - gw) * lin["B"]
        for sigma in (0.0, 1.5, 4.0):
            u8 = A.to_u8(L2, sigma=sigma, rng=np.random.default_rng(5))
            res, sel = A.est_scalar2d(u8, pc, cands, metrics=("bando", "zsum"))
            for m in ("bando", "zsum"):
                v = res[m][0][roi][sel[roi]]
                v = v[np.isfinite(v)]
                p(f"c={c_true:+.0f} {label:28s} σ={sigma:.1f} {m:5s}: 推定 ĉ の中央値 {np.median(v):+6.2f}・|ĉ|<1 の割合 {np.mean(np.abs(v) < 1):.2f}・"
                  f"|ĉ−c|≤1 の割合 {np.mean(np.abs(v - c_true) <= 1):.2f}（n={v.size}）")
log.close()
