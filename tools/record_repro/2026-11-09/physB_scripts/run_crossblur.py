"""ぼけ形状の不一致への対処: PSF 合わせ（(iv)）は重心モデルの照合（(ii)）の偏りを消すか（physB・2026-11-09）

目的: run_orient.py で、CMY 三扇形や「C/Y＋下 M 扇形」のように各チャネルの開口が大きく形が違う組では、
      重心で戻して照合する (ii) の ĉ に c の 15〜25 % の偏りが出た。[Mishima2018] の「チャネル別 PSF を畳み込んで
      ぼけを揃える」考え方（apsim.est_crossblur・2 次元・任意形状に拡張した自作版）で偏りが消えるかを見る。
入力: なし
出力: physB_out/crossblur.txt, physB_out/crossblur.csv（UTF-8）
条件: 形状 6 種 × 被写体（等方 1/f と 向き 0/45/90/135° のバーコード）× c=3,6,12 × σ=1.5。
      候補 c=−30〜30（1 px 刻み・両法で同じ）。校正の比では割らない（raw）。
仮定: apsim.py 冒頭のとおり。推定側は真の PSF の形を知っている（実機では PSF の校正が要る）[仮定]。
"""
import csv
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import apsim as ap

OUT = os.path.join(os.path.dirname(HERE), "physB_out")
H = W = 192
SS = 4
ROI = (slice(40, 152), slice(40, 152))
CS = [3.0, 6.0, 12.0]
SHAPES = ["cy_lr", "rgb3", "cmy3", "ycg3", "cy_msec", "kaneko"]
CANDS = np.arange(-30, 31, 1.0)
SCN = [("iso", None), ("bar0", 0.0), ("bar45", 45.0), ("bar90", 90.0), ("bar135", 135.0)]

t0 = time.time()
scenes = {}
for name, phi in SCN:
    rng = np.random.default_rng(4242 + (0 if phi is None else int(phi)))
    img = ap.scene_1overf(H, W, SS, rng) if phi is None else ap.scene_barcode(H, W, SS, phi, rng)
    scenes[name] = ap.Scene(img, SS)
rows = []
for shape in SHAPES:
    fn = ap.SHAPES[shape][0]
    p = ap.centroid_img(ap.shape_stats(fn))
    for c in CS:
        for name, phi in SCN:
            sc = scenes[name]
            lin = ap.render(sc, fn, c)
            u8 = ap.to_u8(lin, 1.5, np.random.default_rng(int(c) + 17))
            m, _, _ = ap.eval_mask_from_full(sc, c, roi=ROI)
            res, _ = ap.est_scalar2d(u8, p, CANDS)
            e4, _, _ = ap.est_crossblur(u8, fn, CANDS)
            for mt, e in [("ii_zsum", res["zsum"][0]), ("ii_bando", res["bando"][0]), ("iv_xblur", e4)]:
                rows.append(dict(shape=shape, c=c, scene=name, method=mt, **ap.stats(e, c, m)))
    print(f"{shape} 済 ({time.time() - t0:.0f} s)")

with open(os.path.join(OUT, "crossblur.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


log("=== PSF 合わせ (iv) と 重心で戻す照合 (ii) の比較（raw・σ=1.5）===")
log("各セル: 偏り/c [%] , ばらつき [px] , 良好率   の順に 被写体 " + ", ".join(n for n, _ in SCN))
for c in CS:
    log(f"--- c={c:g}")
    for shape in SHAPES:
        for mt in ["ii_zsum", "ii_bando", "iv_xblur"]:
            cells = []
            for name, _ in SCN:
                r = [x for x in rows if x["shape"] == shape and x["c"] == c and x["scene"] == name and x["method"] == mt][0]
                cells.append(f"{100 * r['bias'] / c:+5.1f}%,{r['sd']:.2f},{r['good']:.2f}")
            log(f"  {ap.SHAPES[shape][1]:<12s} {mt:<9s}: " + " | ".join(cells))
log(f"所要 {time.time() - t0:.0f} s")
with open(os.path.join(OUT, "crossblur.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(LOG) + "\n")
