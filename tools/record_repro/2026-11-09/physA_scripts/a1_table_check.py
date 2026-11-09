# -*- coding: utf-8 -*-
"""研究方針.md §3 の表（6 形状の平均光量・縦/横エッジの基線・最も苦手な方向の基線）を独立に再計算して照合する。

入力: なし（形状は研究方針.md §7 の定義を catalog.py に実装）
出力: 標準出力の表、physA_out/a1_table_check.csv
方法: (1) 解析値: 扇形（半中心角 α）の重心距離 2 sinα/(3α)、半円 4/(3π) から重心を出す
      (2) 数値積分: 半径 1 の円を 2400×2400 のセル中心格子で積分
      基線 = 3 チャネルの重心の差の法線成分の最大値。最も苦手な方向は 0.01° 刻みの最小と三角形の最小幅の両方。
仮定: 理想フィルタ・幾何光学・回折なし・アオリなし（aperture_lib の仮定）
"""
import csv
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

import aperture_lib as al
import catalog as ct

OUT = os.path.join(os.path.dirname(HERE), "physA_out")
os.makedirs(OUT, exist_ok=True)


def pol(r, deg):
    t = np.deg2rad(deg)
    return np.array([r * np.cos(t), r * np.sin(t)])


hd = 4 / (3 * np.pi)  # 半円の重心距離
sc = 2 * np.sin(np.pi / 3) / (3 * np.pi / 3)  # 120° 扇形の重心距離

# 解析の重心と光量（開口比）
ANALYTIC = {
    "cy_lr": (np.array([pol(hd, 180), [0, 0], pol(hd, 0)]), np.array([0.5, 1, 0.5])),
    "cy_lr_45": (np.array([pol(hd, 225), [0, 0], pol(hd, 45)]), np.array([0.5, 1, 0.5])),
    "rgb3": (np.array([pol(sc, 210), pol(sc, 90), pol(sc, 330)]), np.array([1 / 3, 1 / 3, 1 / 3])),
    # CMY: R=M∪Y → 全円から C 扇形を除いた領域の重心 = −(1/3)·p_C/(2/3) = −p_C/2
    "cmy3": (np.array([-pol(sc, 210) / 2, -pol(sc, 90) / 2, -pol(sc, 330) / 2]), np.array([2 / 3, 2 / 3, 2 / 3])),
    "ycg3": (np.array([pol(sc, 150), [0, 0], pol(sc, 30)]), np.array([1 / 3, 1, 1 / 3])),
    "quad": (np.array([pol(hd, 180), pol(hd, 90), pol(hd, 0)]), np.array([0.5, 0.5, 0.5])),
}
TABLE = {  # 研究方針.md §3 の表の値
    "cy_lr": (0.67, 0.85, 0.0, 0.0),
    "cy_lr_45": (0.67, 0.60, 0.60, 0.0),
    "rgb3": (0.33, 0.95, 0.83, 0.83),
    "cmy3": (0.67, 0.48, 0.41, 0.41),
    "ycg3": (0.56, 0.95, 0.28, 0.28),
    "quad": (0.50, 0.85, 0.42, 0.42),
}


def metrics_from_centroids(p, a):
    th = np.deg2rad(np.arange(0, 180, 0.01))
    N = np.stack([np.cos(th), np.sin(th)], 1)
    q = p @ N.T
    prof = q.max(0) - q.min(0)
    return dict(light=a.mean(), vert=prof[0], horiz=prof[9000], worst=prof.min(),
                worst_tri=al.tri_min_width(p, np.ones(3, bool)))


grid = al.Grid(2400)
rows = []
print("形状 | 光量 表/解析/数値 | 縦 表/解析/数値 | 横 表/解析/数値 | 最も苦手 表/解析/数値/三角形の最小幅(数値) | 判定")
fns = {k: f for k, f, _, _ in ct.DESIGNS}
for key in ANALYTIC:
    pA, aA = ANALYTIC[key]
    mA = metrics_from_centroids(pA, aA)
    T = fns[key](grid.x, grid.y)
    a, m, p = al.channel_moments(T, grid)
    mN = metrics_from_centroids(p, a)
    tb = TABLE[key]
    ok = all(abs(x - y) <= 0.006 for x, y in zip(tb, (mA["light"], mA["vert"], mA["horiz"], mA["worst"])))
    ok2 = all(abs(x - y) <= 2e-3 for x, y in zip((mA["light"], mA["vert"], mA["horiz"], mA["worst"]),
                                                 (mN["light"], mN["vert"], mN["horiz"], mN["worst"])))
    print(f"{key} | {tb[0]:.2f}/{mA['light']:.4f}/{mN['light']:.4f} | {tb[1]:.2f}/{mA['vert']:.4f}/{mN['vert']:.4f} | "
          f"{tb[2]:.2f}/{mA['horiz']:.4f}/{mN['horiz']:.4f} | {tb[3]:.2f}/{mA['worst']:.4f}/{mN['worst']:.4f}/{mN['worst_tri']:.4f} | "
          f"表≒解析 {'一致' if ok else '不一致'}・解析≒数値 {'一致' if ok2 else '不一致'}")
    rows.append(dict(key=key, table_light=tb[0], an_light=mA["light"], num_light=mN["light"],
                     table_vert=tb[1], an_vert=mA["vert"], num_vert=mN["vert"],
                     table_horiz=tb[2], an_horiz=mA["horiz"], num_horiz=mN["horiz"],
                     table_worst=tb[3], an_worst=mA["worst"], num_worst=mN["worst"], num_worst_tri=mN["worst_tri"],
                     match_table=ok, match_numeric=ok2))
    print("    重心（数値）R,G,B =", np.round(p, 4).tolist(), " 光量 =", np.round(a, 4).tolist())

with open(os.path.join(OUT, "a1_table_check.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("格子の円の面積/π =", grid.x.size * grid.dA / np.pi)
