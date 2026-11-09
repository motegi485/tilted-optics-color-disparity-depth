# -*- coding: utf-8 -*-
"""design_space_analysis.md §8.2 の「S_i は S_mag の 31〜43 %」「h<0.05 の窓が R で 21〜38 %・G で 28〜48 %」「h<0.02 が R 10〜20 %・G 21〜34 %」を、
verReal の 6 場面＋ノイズ板 2 距離で独立に数える（S_mag の画素を中心とする 25×25 窓。読み取りのみ）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

import vr_lib as L

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verReal_out")
log = open(os.path.join(OUT, "v5c_hfrac.txt"), "w", encoding="utf-8")


def p(s):
    print(s)
    log.write(s + "\n")


def frames_of(ses):
    if ses == "b5_scene_01":
        return list(range(10, 470, 50))
    if ses.startswith("b5_dist"):
        return [10, 70, 130, 190]
    idxs = L.list_indices(ses)
    mid = len(idxs) // 2
    return [idxs[min(max(mid - 4 + 3 * k, 0), len(idxs) - 1)] for k in range(4)]


for ses in ["b5_scene_01", "20260715_184146", "20260715_155442", "20260715_180809", "20260715_180845", "20260710_143749",
            "b5_dist_1450", "b5_dist_3050"]:
    acc = dict(si=0, sm=0, r05=0, r02=0, g05=0, g02=0)
    for idx in frames_of(ses):
        f = L.feats(L.load_proc(ses, idx))
        sm = f["S_mag"]
        acc["si"] += int(f["S_i"].sum())
        acc["sm"] += int(sm.sum())
        acc["r05"] += int((f["h_R"][sm] < 0.05).sum())
        acc["r02"] += int((f["h_R"][sm] < 0.02).sum())
        acc["g05"] += int((f["h_G"][sm] < 0.05).sum())
        acc["g02"] += int((f["h_G"][sm] < 0.02).sum())
    sm = max(acc["sm"], 1)
    p(f"{ses:18s} S_i/S_mag {acc['si'] / sm:.2f} | S_mag の窓 h_R<0.05 {acc['r05'] / sm:.2f}・h_R<0.02 {acc['r02'] / sm:.2f}"
      f" | h_G<0.05 {acc['g05'] / sm:.2f}・h_G<0.02 {acc['g02'] / sm:.2f}")
log.close()
