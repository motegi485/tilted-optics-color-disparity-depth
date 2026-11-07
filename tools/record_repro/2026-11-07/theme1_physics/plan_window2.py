"""身長 1.7 m の人の高さのうち、|d| が 0.75〜5 px に入る割合（528 幅・薄レンズの光線追跡・実測なし）。"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from optics import Rig, rig_parallel, rig_untilted, raytrace_world

S = 528.0 / 1080.0
LOW, HIGH = 0.75, 5.0
rigs = [
    rig_untilted("傾けない・無限遠"),
    rig_untilted("傾けない・5 m 合焦", z_focus=5000.0),
    rig_parallel("路面合焦", 0.0),
    rig_parallel("1.0 m 面", 1000.0),
    Rig("θ=30°・v=103.92", 30.0, 90.0 / np.cos(np.deg2rad(30.0)), Hc=1330.0),
    Rig("現行 θ=30°・v=109.3", 30.0, 109.3, Hc=1330.0),
]
heights = np.arange(0.0, 1700.1, 20.0)
for r in rigs:
    print(f"\n{r.name}: v={r.v:.2f}")
    for z in [10000.0, 15000.0, 20000.0, 30000.0, 40000.0]:
        ds, cs = [], []
        for H in heights:
            rt = raytrace_world(r, float(H), z)
            ds.append(abs(rt["d"]) * S)
            cs.append(rt["c_x"] * S)
        ds, cs = np.array(ds), np.array(cs)
        inside = np.mean((ds > LOW) & (ds < HIGH)) * 100
        box = r.F * S * 1700.0 / z
        print(
            f" z={z/1000:.0f} m 人 {box:.0f}px 窓内 {inside:5.1f} % | |d| {ds.min():.2f}〜{ds.max():.2f} | c {cs.min():.1f}〜{cs.max():.1f} px | c 最大/人 {cs.max()/box*100:.1f} %"
        )
