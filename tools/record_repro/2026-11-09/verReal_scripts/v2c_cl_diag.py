# -*- coding: utf-8 -*-
"""color-line が neg 枝で d≈0 に潰れる理由の切り分け（ノイズ板 2050 mm・1 フレーム・読み取りのみ）。
(i) の d が −3.5〜−2.5 の画素の帯で、候補 d ごとの det の平均と、各対の相関 ρ の平均を、G の位置 g と補間法を変えて出す。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import cv2
import numpy as np

import vr_lib as L

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verReal_out")
log = open(os.path.join(OUT, "v2c_cl_diag.txt"), "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")


def moments(proc, g, interp, off=0.25):
    R = L.srgb_lin(proc[:, :, 2]).astype(np.float32)
    Gc = L.srgb_lin(proc[:, :, 1]).astype(np.float32)
    B = L.srgb_lin(proc[:, :, 0]).astype(np.float32)

    def rm(a, s):
        return cv2.remap(a, L._X + np.float32(s), L._MAPY, interp, borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
    Gs = rm(Gc, off)
    mG = L.box(Gs)
    vG = L.box(Gs * Gs) - mG * mG
    res = []
    for d in L.DS:
        Rs = rm(R, g * d + off)
        Bs = rm(B, -(1 - g) * d + off)
        mR, mB = L.box(Rs), L.box(Bs)
        vR = L.box(Rs * Rs) - mR * mR
        vB = L.box(Bs * Bs) - mB * mB
        rg = (L.box(Rs * Gs) - mR * mG) / np.sqrt(vR * vG)
        rb = (L.box(Rs * Bs) - mR * mB) / np.sqrt(vR * vB)
        gb = (L.box(Bs * Gs) - mB * mG) / np.sqrt(vB * vG)
        det = 1 - rg ** 2 - rb ** 2 - gb ** 2 + 2 * rg * rb * gb
        res.append((rg, rb, gb, det))
    return res


for gt, lo, hi in ((2050, -3.5, -2.5), (2050, 2.5, 3.5), (1450, -2.0, -1.0)):
    proc = L.load_proc(f"b5_dist_{gt}", 10)
    di = L.banded(L.rust_disp(proc, L.TH))
    m = np.isfinite(di) & (di > lo) & (di < hi)
    p(f"\n== b5_dist_{gt} #10: (i) の d が ({lo}, {hi}) の画素 {int(m.sum())}（中央値 {np.median(di[m]):.2f}）")
    for g, interp, name in ((0.5, cv2.INTER_CUBIC, "g=0.5 cubic"), (0.1, cv2.INTER_CUBIC, "g=0.1 cubic"),
                            (0.1, cv2.INTER_LANCZOS4, "g=0.1 lanczos"), (0.0, cv2.INTER_CUBIC, "g=0.0 cubic"),
                            (0.27, cv2.INTER_CUBIC, "g=0.27 cubic"), (0.5, cv2.INTER_LINEAR, "g=0.5 linear")):
        res = moments(proc, g, interp)
        det = np.array([np.nanmean(r[3][m]) for r in res])
        rg = np.array([np.nanmean(r[0][m]) for r in res])
        rb = np.array([np.nanmean(r[1][m]) for r in res])
        gb = np.array([np.nanmean(r[2][m]) for r in res])
        # 画素ごとの最小の位置（端 2 ビン除外）
        D = np.stack([r[3] for r in res])
        mins = L.DS[np.argmin(np.where(np.isnan(D[1:-1]), np.inf, D[1:-1]), axis=0) + 1][m]
        k = int(np.argmin(det[1:-1])) + 1
        sel = [i for i in range(len(L.DS)) if abs(L.DS[i]) <= 6]
        p(f"  {name:14s}: 平均 det の最小 d={L.DS[k]:+d}; 画素ごとの最小の中央値 {np.median(mins):+.1f}")
        p("     d   : " + " ".join(f"{L.DS[i]:+6d}" for i in sel))
        p("     det : " + " ".join(f"{det[i]:6.4f}" for i in sel))
        p("     ρRG : " + " ".join(f"{rg[i]:6.3f}" for i in sel))
        p("     ρRB : " + " ".join(f"{rb[i]:6.3f}" for i in sel))
        p("     ρGB : " + " ".join(f"{gb[i]:6.3f}" for i in sel))
log.close()
