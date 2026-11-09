"""verPhys v11: physB の apsim.py そのもので、現行 C/Y＋color-line の c=4/5/8/10 を評価する（2026-11-09）

目的: 自作（v8 D・v10）で、現行 C/Y の color-line は強い異方性の 1/f に対し c=4/5/10 で良好率 0.00〜0.19（偏り ±30〜39 %）だった。
      physB は c=3/6/12 だけを評価している。自作の実装の問題か、physB の実装でも出るかを切り分ける。
      physB の run_aniso_strong.py と同じ被写体（種 211・212）・雑音・校正（iso_gain）・候補（−47〜47）で、c だけ増やす。
入力: physB_scripts/apsim.py（読み取りで import。physB のファイルは変更しない）
出力: verPhys_out/v11_log.txt（UTF-8）
"""
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PHYSB = os.path.join(os.path.dirname(HERE), "physB_scripts")
sys.path.insert(0, PHYSB)
import numpy as np

import apsim as ap

OUT = os.path.join(os.path.dirname(HERE), "verPhys_out")
LOG = []
t0 = time.time()
H = W = 192
SS = 4
ROI = (slice(40, 152), slice(40, 152))
CANDS = np.arange(-47, 48, 1.0)


def log(s=""):
    print(s, flush=True)
    LOG.append(str(s))


scenes = [ap.Scene(ap.scene_1overf(H, W, SS, np.random.default_rng(s), aniso_sigma_deg=4.0, floor=0.03), SS) for s in (211, 212)]
log("=== physB の apsim・現行 C/Y（cy_lr）・(ii) color-line（ii_bando）・強い異方性（種 211・212）")
for shape in ("cy_lr", "rgb3"):
    fn = ap.SHAPES[shape][0]
    p = ap.centroid_img(ap.shape_stats(fn))
    for c in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0):
        k = ap.iso_gain(fn, p, c, CANDS, False)["ii_bando"]
        raw, cal, med = [], [], []
        for j, sc in enumerate(scenes):
            m, _, _ = ap.eval_mask_from_full(sc, c, roi=ROI)
            u8 = ap.to_u8(ap.render(sc, fn, c), 1.5, np.random.default_rng(j + int(c)))
            e = ap.run_methods(u8, p, CANDS, False)["ii_bando"]
            raw.append(ap.stats(e, c, m))
            cal.append(ap.stats(e / k, c, m))
            med.append(100 * raw[-1]["bias"] / c)
        n = sum(s["n"] for s in raw)
        gr = sum(s["good"] * s["n"] for s in raw) / n
        gc = sum(s["good"] * s["n"] for s in cal) / n
        log(f"  {shape:6s} c={c:4.1f} k={k:.3f}: 良好 raw {gr:.2f}／校正 {gc:.2f}  偏り（種 211, 212）{med[0]:+.0f}%, {med[1]:+.0f}%  ({time.time() - t0:.0f}s)")
        with open(os.path.join(OUT, "v11_log.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(LOG) + "\n")
