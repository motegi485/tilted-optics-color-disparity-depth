# -*- coding: utf-8 -*-
"""c2_rb_ratio_columns.py (critic, 2026-11-09)

目的: 現行フィルタが絞り位置にあるか（光束の足跡が像高とともにずれないか）を間接的に見る。
      フィルタが絞りから離れていると、左右の像高で R（Y 半円）と B（C 半円）を通る光量の比が
      変わる（c1_decenter.py で s=0.1 なら R の光量 0.44〜0.56）。左右方向に R/G・B/G の比が
      系統的に傾くかを、校正板（白色ノイズの平板）の画像で列ごとに数える。
限界: 周辺減光・照明むら・板の色むら・WB でも比は傾きうるので、傾きが無ければ「絞り位置か
      その近く」と読めるが、傾きがあっても原因は特定できない。読み取りのみ。
入力: 開発リポ data/calib_raw/*.png（存在する分） 出力: critic_out/c2_rb_ratio_columns.txt
"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(r"C:\Users\t5424001\dev\tilted_optics_color_disparity_depth")
OUT = Path(__file__).resolve().parent.parent / "critic_out"
files = sorted((ROOT / "data" / "calib_raw").glob("*.png"))
lines = ["# 校正板画像の列ごとの平均 R/G・B/G（画像全体の比で正規化）。左端・中央・右端の 1/8 幅の帯"]
for f in files:
    img = cv2.imread(str(f), cv2.IMREAD_COLOR)
    if img is None:
        continue
    h, w = img.shape[:2]
    # 黒帯があれば除く（列平均が暗い列）
    colmean = img.mean(axis=(0, 2))
    valid = np.where(colmean > 0.25 * colmean.max())[0]
    x0, x1 = valid.min(), valid.max() + 1
    sub = img[:, x0:x1].astype(np.float64)
    B, G, R = sub[..., 0], sub[..., 1], sub[..., 2]
    W = x1 - x0
    band = W // 8
    def ratio(sl):
        return R[:, sl].mean() / G[:, sl].mean(), B[:, sl].mean() / G[:, sl].mean()
    rg_all, bg_all = ratio(slice(0, W))
    out = []
    for name, sl in (("左", slice(0, band)), ("中", slice(W // 2 - band // 2, W // 2 + band // 2)), ("右", slice(W - band, W))):
        rg, bg = ratio(sl)
        out.append(f"{name} R/G {rg / rg_all:.3f} B/G {bg / bg_all:.3f}")
    # 上下も（参考）
    def ratio_r(sl):
        return R[sl, :].mean() / G[sl, :].mean(), B[sl, :].mean() / G[sl, :].mean()
    hb = h // 8
    out2 = []
    for name, sl in (("上", slice(0, hb)), ("下", slice(h - hb, h))):
        rg, bg = ratio_r(sl)
        out2.append(f"{name} R/G {rg / rg_all:.3f} B/G {bg / bg_all:.3f}")
    lines.append(f"{f.name} ({w}x{h}, 有効列 {x0}-{x1 - 1}): " + " | ".join(out) + " || " + " | ".join(out2))
txt = "\n".join(lines)
(OUT / "c2_rb_ratio_columns.txt").write_text(txt, encoding="utf-8", newline="\n")
print(txt)
