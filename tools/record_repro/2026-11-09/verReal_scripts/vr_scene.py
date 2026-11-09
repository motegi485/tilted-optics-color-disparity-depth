# -*- coding: utf-8 -*-
"""verReal: 通常シーン・屋内の場面での集計部品（正解のない指標: 時間方向のばらつき・空間の滑らかさ・(i) との一致）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import vr_lib as L

SCOMBOS = [
    ("i", "S_i", "(i) 現行"),
    ("i_all", "S_mag", "(i) の照合・S_mag"),
    ("i_all", "S_h0.1", "(iii) (i) の照合・h≥0.1"),
    ("i_all", "S_J0", "(iii) (i) の照合・Txx≥J0"),
    ("cl50", "S_mag", "(ii) color-line・S_mag"),
    ("rb", "S_i", "明るさ ZNCC・S_i"),
    ("rb", "S_mag", "明るさ ZNCC・S_mag"),
    ("rb", "S_J0", "明るさ ZNCC・Txx≥J0"),
    ("gm", "S_mag", "|∇| ZNCC・S_mag"),
]
SMETHODS = ["i", "i_all", "cl50", "rb", "gm"]


def compute(proc):
    f = L.feats(proc)
    d = {"i": L.rust_disp(proc, L.TH), "i_all": L.rust_disp(proc, 0), "cl50": L.disp_cl(proc, 0.5),
         "rb": L.disp_rb(proc), "gm": L.disp_gm(proc)}
    return f, d


def local_dev(db, r=3, min_n=5):
    """有効画素ごとに |d − 近傍（(2r+1)² の有効画素）の中央値|。近傍の有効画素が min_n 未満なら NaN。"""
    H, W = db.shape
    pad = np.pad(db, r, constant_values=np.nan)
    win = sliding_window_view(pad, (2 * r + 1, 2 * r + 1))
    ys, xs = np.nonzero(np.isfinite(db))
    out = np.full(db.shape, np.nan)
    if ys.size == 0:
        return out
    w = win[ys, xs].reshape(ys.size, -1)
    n = np.isfinite(w).sum(1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(w, axis=1)
    v = np.abs(db[ys, xs] - med)
    v[n < min_n] = np.nan
    out[ys, xs] = v
    return out
