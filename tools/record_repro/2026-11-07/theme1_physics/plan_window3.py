"""身長 1.7 m の人の高さのうち、視差が帯に入る割合を下限 0.45 px と 0.75 px で比べる（528 幅・薄レンズの光線追跡・実測なし）。"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from optics import Rig, rig_parallel, rig_untilted, raytrace_world, BETA

S = 528.0 / 1080.0
HIGH = 5.0
rigs = [
    rig_untilted("傾けない・無限遠"),
    rig_untilted("傾けない・5 m 合焦", z_focus=5000.0),
    rig_parallel("路面合焦", 0.0),
    rig_parallel("1.0 m 面", 1000.0),
    Rig("θ=30°・v=103.92", 30.0, 90.0 / np.cos(np.deg2rad(30.0)), Hc=1330.0),
    Rig("現行 θ=30°・v=109.3", 30.0, 109.3, Hc=1330.0),
]
heights = np.arange(0.0, 1700.1, 20.0)
print("配置 | v | F528 | K528 | 10 m | 20 m | 30 m | 50 m（各セル: 下限0.45の割合/下限0.75の割合/|d|範囲/最大ぼけ÷人の縦）")
for r in rigs:
    F = r.F * S
    K = BETA * r.A * F / 1000.0
    cells = []
    for z in [10000.0, 20000.0, 30000.0, 50000.0]:
        ds, cs = [], []
        for H in heights:
            rt = raytrace_world(r, float(H), z)
            ds.append(abs(rt["d"]) * S)
            cs.append(rt["c_x"] * S)
        ds, cs = np.array(ds), np.array(cs)
        a = np.mean((ds > 0.45) & (ds < HIGH)) * 100
        b = np.mean((ds > 0.75) & (ds < HIGH)) * 100
        box = F * 1700.0 / z
        cells.append(f"{a:.0f}/{b:.0f}/{ds.min():.2f}-{ds.max():.2f}/{cs.max() / box * 100:.1f}%")
    print(f"{r.name} | {r.v:.2f} | {F:.0f} | {K:.2f} | " + " | ".join(cells))

K = 14.83
print()
print(f"傾けない無限遠の全行が帯に入る距離 K={K}: 下限 0.45 → {K / 0.45:.1f} m、下限 0.75 → {K / 0.75:.1f} m、上限 5 → {K / 5:.2f} m")
print()
print("同じぼけの許容量で許される F 値（c_t=5 px@1080, z_min=10 m, Hc=1.3 m, F=4445 px@1080）")
beta = BETA
F1080 = 4445.0
c_t, zmin = 5.0, 10.0
K_flat = beta * c_t * zmin
print(f"傾けない: K={K_flat:.1f} px・m@1080 → A={K_flat * 1000 / (beta * F1080):.2f} mm → F{90 / (K_flat * 1000 / (beta * F1080)):.1f}")
for Ht in [0.3, 1.0, 1.7]:
    K_r = K_flat * 1.3 / Ht
    A = K_r * 1000 / (beta * F1080)
    print(f"路面合焦 H_t={Ht} m: K={K_r:.1f} → A={A:.2f} mm → F{90 / A:.1f}（{K_r / K_flat:.2f} 倍）")
