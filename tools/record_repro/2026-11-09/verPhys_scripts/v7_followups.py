"""verPhys v7: v5・v6 で見えた点の追加検査（2026-11-09）

目的:
  A. 現行 C/Y・横向きの筋の強い被写体で color-line に残る ±13 % の偏りは、候補 c の刻み（1 px）と放物線補間の産物か、
     G（全円）と R/B（半円）の形の違いによるものか。刻み 0.25 px と、R–B の 2 チャネルだけ（鏡像の半円どうし）で比べる。
  B. アオリのばらつきの比（縦/横）は雑音の水準でどう変わるか（σ=1.5 と 4）。比は雑音の床に依存するので、
     絶対値の上乗せ √(sd_tilt² − sd_0²) も出す。
入力: なし（v6_tilt.py の関数を import）
出力: verPhys_out/v7_log.txt
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
    print(s, flush=True)
    LOG.append(str(s))


H = W = 192
SS = 4
M = 24
t0 = time.time()

log("=== A. color-line の偏り（現行 C/Y・校正なしの中央値の偏り / 良好率）")
fn = vp.ap_current
q = vp.qvec(fn)
for kind, maker in [("iso", lambda r: vp.subj_1overf(H + 2 * M, W + 2 * M, SS, r)),
                    ("aniso_strong", lambda r: vp.subj_1overf(H + 2 * M, W + 2 * M, SS, r, aniso_deg=4.0, floor=0.03)),
                    ("bars_bend", lambda r: vp.subj_bars(H + 2 * M, W + 2 * M, SS, r, finite=False, bend=True))]:
    for c in (3.0, 6.0):
        out = []
        for label, cands, chans in [("3ch 刻み1", np.arange(-24, 25, 1.0), vp.CH), ("3ch 刻み0.25", np.arange(-16, 16.01, 0.25), vp.CH),
                                    ("R–B 刻み1", np.arange(-24, 25, 1.0), ("R", "B")), ("R–B 刻み0.25", np.arange(-16, 16.01, 0.25), ("R", "B"))]:
            biases, goods, n = [], 0, 0
            for sd in (1, 2):
                subj = maker(np.random.default_rng(500 + sd))
                E, _, _ = vp.eval_mask(subj, c, SS, margin=M)
                u8 = vp.to8(vp.render(subj, fn, c, SS, margin=M), 1.5, np.random.default_rng(600 + sd))
                e = vp.est_colorline(u8, q, cands, chans=chans)
                ev = vp.evaluate(e, c, E)
                biases.append(ev["bias"])
                goods += ev["good"] * ev["n"]
                n += ev["n"]
            out.append(f"{label}: 偏り {np.mean(biases):+.2f}（{100 * np.mean(biases) / c:+.0f}%）良好 {goods / n:.2f}")
        log(f"  {kind:12s} c={c:g}: " + " | ".join(out) + f"  ({time.time() - t0:.0f}s)")

log()
log("=== B. アオリのばらつきの比と雑音の水準（厳密な描画・R–B の color-line・c0=6）")
import v6_tilt as v6

v6.LOG = LOG  # 同じログに書く
for sig in (1.5, 4.0):
    base = v6.run_case(f"σ={sig} アオリなし", 0.0, 6.0, sigma=sig)
    for th in (3.97, 30.0):
        r = v6.run_case(f"σ={sig} θ={th}°", th, 6.0, sigma=sig)
        add_h = np.sqrt(max(r["current"]["sd"] ** 2 - base["current"]["sd"] ** 2, 0))
        add_v = np.sqrt(max(r["tb"]["sd"] ** 2 - base["tb"]["sd"] ** 2, 0))
        log(f"      上乗せ √(sd²−sd0²): 横の基線 {add_h:.3f} px・縦の基線 {add_v:.3f} px（c の px）")

with open(os.path.join(OUT, "v7_log.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(LOG) + "\n")
