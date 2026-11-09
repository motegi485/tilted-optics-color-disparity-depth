# -*- coding: utf-8 -*-
"""自己照合: (1) Rust (i) の再計算が本番の保存 npy とビット一致 (2) T=0 で呼んだ Rust を S_i で切ると T=20 と一致
(3) 自作の box filter 版 (i) が Rust と一致 (4) 計算時間。読み取りのみ。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import vr_lib as L
from tools.analysis_common import apply_gates, depth_of

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verReal_out")
log = open(os.path.join(OUT, "v1_selfcheck.txt"), "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")


def same(a, b):
    return a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])


for ses, idx in (("b5_dist_2050", 10), ("b5_dist_1000", 60), ("b5_scene_01", 10), ("b5_scene_01", 200)):
    proc = L.load_proc(ses, idx)
    t0 = time.time()
    d20 = L.rust_disp(proc, 20)
    t1 = time.time()
    d0 = L.rust_disp(proc, 0)
    t2 = time.time()
    z = depth_of(apply_gates(d20, disp_max=L.config.DISP_MAX, band=L.BAND), L.PARAMS)
    saved = np.load(os.path.join(L.OUTP, ses, f"{idx:06d}.npy"))
    p(f"{ses} #{idx}: Rust T=20 {t1 - t0:.2f}s / T=0 {t2 - t1:.2f}s; 保存 npy とビット一致: {same(z.astype(np.float32), saved)}")
    f = L.feats(proc)
    Si = f["S_i"]
    fin20 = np.isfinite(d20)
    p(f"   T=20 の有限画素 ⊂ S_i: {bool(np.all(Si[fin20]))}; S_i {int(Si.sum())} 画素・T=20 有限 {int(fin20.sum())}")
    d0m = np.where(Si, d0, np.nan)
    p(f"   T=0 を S_i で切ったものと T=20 が一致: {same(d0m.astype(np.float32), d20)}")
    t3 = time.time()
    dbx = L.disp_gx_box(proc)
    t4 = time.time()
    dbxm = np.where(Si, dbx, np.nan)
    both = np.isfinite(dbxm) & fin20
    only_r = fin20 & ~np.isfinite(dbxm)
    only_b = np.isfinite(dbxm) & ~fin20
    diff = np.abs(dbxm[both] - d20[both])
    p(f"   box 版 (i) {t4 - t3:.2f}s: 両方有効 {int(both.sum())}・Rust のみ {int(only_r.sum())}・box のみ {int(only_b.sum())}; "
      f"差の最大 {diff.max():.2e} px・差>1e-3 の画素 {int((diff > 1e-3).sum())}")
    t5 = time.time()
    dcl = L.disp_cl(proc, 0.5)
    t6 = time.time()
    p(f"   color-line g=0.5 {t6 - t5:.2f}s; S_mag 内の有効 {int(np.isfinite(dcl[f['S_mag']]).sum())} / {int(f['S_mag'].sum())}")
log.close()
