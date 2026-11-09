# -*- coding: utf-8 -*-
"""v2_noise_rows.csv（ノイズ板・既に計算済み）から、画素の勾配方向 φ と窓の h の区分別に有効画素と MAE をまとめる（再計算しない）。"""
import csv
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verReal_out")
rows = list(csv.DictReader(open(os.path.join(OUT, "v2_noise_rows.csv"), encoding="utf-8")))
IN = [1450, 1650, 1850, 2050, 2250, 2450, 2650, 2850, 3050]
COMBOS = [("i", "S_i"), ("i_all", "S_mag"), ("i_all", "S_J0"), ("cl50", "S_mag"), ("rb", "S_mag"), ("gm", "S_mag")]
log = open(os.path.join(OUT, "v5b_noise_bins.txt"), "w", encoding="utf-8")


def p(s):
    print(s)
    log.write(s + "\n")


p("校正範囲内 1450〜3050 mm の 9 距離（各 4 フレーム）。値は距離ごとの n/フレーム・MAE・良好10% の単純平均")
for prefix in ("pbin:", "hbin:"):
    subs = sorted({r["subset"] for r in rows if r["subset"].startswith(prefix)}, key=lambda s: [r["subset"] for r in rows].index(s))
    p(f"\n--- {prefix[:-1]}（pbin = R の画素の勾配方向 φ〈0°=縦エッジ・90°=横エッジ〉、hbin = R の 25×25 窓の h=Txx/(Txx+Tyy)）---")
    for (m, sel) in COMBOS:
        parts = []
        for sub in subs:
            rr = [r for r in rows if int(r["gt"]) in IN and r["method"] == m and r["sel"] == sel and r["subset"] == sub]
            n = np.mean([float(r["n_per_frame"]) for r in rr])
            mae = np.nanmean([float(r["mae"]) if r["mae"] not in ("", "nan") else np.nan for r in rr])
            g = np.nanmean([float(r["good10"]) if r["good10"] not in ("", "nan") else np.nan for r in rr])
            parts.append(f"{sub[len(prefix):]}: n {n:6.0f} MAE {mae:6.1f} 良10 {g:.3f}")
        p(f"  {m}/{sel:6s} " + " | ".join(parts))
log.close()
