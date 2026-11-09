"""verPhys v0: 自作ライブラリの自己検査（2026-11-09）

目的: (1) PSF の重心が (c/2)·q_k に一致するか、(2) 自作 (i) が開発リポの Rust 実装（core.disparity）と一致するか、
      (3) 等方 1/f で各推定法の比 median(ĉ/c) が 1 に近いか（符号の約束の確認を兼ねる）。
入力: なし（開発リポの core.disparity を読み取りで import）
出力: verPhys_out/v0_log.txt
"""
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import vp_lib as vp

OUT = os.path.join(os.path.dirname(HERE), "verPhys_out")
LOG = []


def log(s=""):
    print(s)
    LOG.append(str(s))


ROOT = r"C:\Users\t5424001\dev\tilted_optics_color_disparity_depth"

log("=== (1) PSF の重心（4 倍格子）と (c/2)·q_k")
for name in ["current", "rgb3", "cmy3", "famB90", "tb"]:
    fn = vp.SHAPES[name]
    for sy in (1.0, np.cos(np.radians(30))):
        q = vp.qvec(fn, sy)
        for c in (-10.0, 10.0):
            ks = vp.psf(fn, c, h=0.25, sy=sy, nsub=3)
            errs = []
            for k in vp.CH:
                K = ks[k]
                R = (K.shape[0] - 1) // 2
                idx = np.arange(-R, R + 1) * 0.25
                cx = (K.sum(0) * idx).sum()
                cy = (K.sum(1) * idx).sum()
                errs.append(np.hypot(cx - 0.5 * c * q[k][0], cy - 0.5 * c * q[k][1]))
            log(f"  {name:8s} sy={sy:.3f} c={c:+.0f}: 最大の差 {max(errs):.4f} px  q={ {k: np.round(q[k], 4).tolist() for k in vp.CH} }")

log()
log("=== (2) 自作 (i) と Rust 実装の一致")
sys.path.insert(0, ROOT)
from core.disparity import calcDisparityByZNCC

H = W = 192
ss = 4
M = 24
fn = vp.ap_current
q = vp.qvec(fn)
for sd, c in [(1, 4.0), (2, 10.0), (3, -6.0)]:
    rng = np.random.default_rng(sd)
    subj = vp.subj_1overf(H + 2 * M, W + 2 * M, ss, rng)
    u8 = vp.to8(vp.render(subj, fn, c, ss, margin=M), 1.5, np.random.default_rng(sd + 100))
    mine = vp.est_i(u8, q)
    d_rust = calcDisparityByZNCC(np.ascontiguousarray(u8["R"]), np.ascontiguousarray(u8["B"]), 25, 41, 20)
    rust = 2.0 * d_rust / (q["R"][0] - q["B"][0])
    fm, fr = np.isfinite(mine), np.isfinite(rust)
    both = fm & fr
    log(f"  c={c:+.0f}: 有効画素 自作 {fm.sum()} / Rust {fr.sum()} / 両方 {both.sum()}  値の差の最大 {np.nanmax(np.abs(mine[both] - rust[both])):.2e} px（ĉ）")

log()
log("=== (3) 等方 1/f での比 median(ĉ/c)・良好率（校正なし）")
cands = np.arange(-24, 25, 1.0)
for name in ["current", "rgb3", "cmy3", "famB90"]:
    fn = vp.SHAPES[name]
    q = vp.qvec(fn)
    for c in (3.0, 6.0, 12.0):
        t0 = time.time()
        rng = np.random.default_rng(7)
        subj = vp.subj_1overf(H + 2 * M, W + 2 * M, ss, rng)
        E, _, _ = vp.eval_mask(subj, c, ss, margin=M)
        u8 = vp.to8(vp.render(subj, fn, c, ss, margin=M), 1.5, np.random.default_rng(8))
        res = {}
        if name == "current":
            res["i"] = vp.est_i(u8, q)
        res["cl"] = vp.est_colorline(u8, q, cands)
        res["iv"] = vp.est_psfmatch(u8, fn, np.arange(-20, 21, 1.0))
        parts = []
        for mt, e in res.items():
            v = e[E]
            v = v[np.isfinite(v)]
            ev = vp.evaluate(e, c, E)
            parts.append(f"{mt}: 比 {np.median(v / c):.3f} 良好 {ev['good']:.2f} 被覆 {ev['cov']:.2f} 偏り {ev['bias']:+.2f} ばらつき {ev['sd']:.2f}")
        log(f"  {name:8s} c={c:4.1f}: " + " | ".join(parts) + f"  ({time.time() - t0:.0f}s)")

with open(os.path.join(OUT, "v0_log.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(LOG) + "\n")
