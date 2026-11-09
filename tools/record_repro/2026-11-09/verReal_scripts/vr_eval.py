# -*- coding: utf-8 -*-
"""verReal: 推定法 × 画素の選び方の組を 1 フレームで計算し、誤差を集計する部品。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import vr_lib as L

# (推定法, 画素の選び方, 表示名)
COMBOS = [
    ("i", "S_i", "(i) 現行"),
    ("i_all", "S_mag", "(i) の照合・S_mag"),
    ("i_all", "S_h0.05", "(iii) (i) の照合・h≥0.05"),
    ("i_all", "S_h0.1", "(iii) (i) の照合・h≥0.1"),
    ("i_all", "S_J0", "(iii) (i) の照合・Txx≥J0"),
    ("cl50", "S_i", "(ii) color-line・S_i"),
    ("cl50", "S_mag", "(ii) color-line・S_mag"),
    ("cl50", "S_h0.05", "(ii) color-line・h≥0.05"),
    ("cl27", "S_mag", "(ii') color-line G=0.27・S_mag"),
    ("rb", "S_mag", "明るさ ZNCC（R,B）・S_mag"),
    ("gm", "S_mag", "|∇| ZNCC・S_mag"),
]
METHODS = ["i", "i_all", "cl50", "cl27", "rb", "gm"]
HB = np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0001])
HB_LABEL = ["h<0.02", "0.02-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.4", "h≥0.4"]
PB = np.array([0.0, 30.0, 60.0, 75.0, 90.001])
PB_LABEL = ["φ<30", "30-60", "60-75", "φ≥75"]


def compute(proc, methods=METHODS):
    f = L.feats(proc)
    d = {}
    if "i" in methods:
        d["i"] = L.rust_disp(proc, L.TH)
    if "i_all" in methods:
        d["i_all"] = L.rust_disp(proc, 0)
    if "cl50" in methods:
        d["cl50"] = L.disp_cl(proc, 0.5)
    if "cl27" in methods:
        d["cl27"] = L.disp_cl(proc, 0.27)
    if "rb" in methods:
        d["rb"] = L.disp_rb(proc)
    if "gm" in methods:
        d["gm"] = L.disp_gm(proc)
    return f, d


def combo_disp(f, d, method, sel):
    return np.where(f[sel], d[method], np.nan).astype(np.float32)


def bins_of(f):
    hb = np.clip(np.digitize(np.nan_to_num(f["h_R"], nan=0.0), HB) - 1, 0, len(HB_LABEL) - 1)
    pb = np.clip(np.digitize(f["phi_R"], PB) - 1, 0, len(PB_LABEL) - 1)
    return hb, pb


def err_stats(err, gt):
    n = int(err.size)
    if n == 0:
        return dict(n=0, bias=np.nan, median=np.nan, mae=np.nan, rmse=np.nan, good10=np.nan)
    return dict(n=n, bias=float(err.mean()), median=float(np.median(err)), mae=float(np.abs(err).mean()),
                rmse=float(np.sqrt((err ** 2).mean())), good10=float((np.abs(err) <= 0.1 * gt).mean()))
