"""人が視差の使いやすい帯に入るかを配置ごとに光線追跡で確かめる（528 幅換算・薄レンズ・実測なし）。"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from optics import Rig, rig_parallel, rig_untilted, raytrace_world, BETA

S = 528.0 / 1080.0
v_flat = 90.0 / np.cos(np.deg2rad(30.0))
rigs = [
    rig_untilted("傾けない・無限遠"),
    rig_parallel("路面合焦", 0.0),
    rig_parallel("1.0 m 面", 1000.0),
    Rig(f"θ=30°・v={v_flat:.2f}(水平面)", 30.0, v_flat, Hc=1330.0),
    Rig("現行 θ=30°・v=109.3", 30.0, 109.3, Hc=1330.0),
]
for r in rigs:
    t = r.tan_psi
    p = r.pof_height(0.0) if np.isfinite(t) else float("nan")
    print(f"\n{r.name}: θ={r.theta_deg:.2f} v={r.v:.2f} F528={r.F*S:.0f} K528={BETA*r.A*r.F*S:.0f} tanψ={t:.4f} 面の高さ(z=0)={p:.0f}")
    for z in [10000.0, 15000.0, 20000.0, 30000.0, 40000.0]:
        out = []
        for name, H in [("頭", 1700.0), ("腰", 900.0), ("足", 0.0)]:
            rt = raytrace_world(r, H, z)
            out.append(f"{name} c={rt['c_x']*S:.1f} |d|={abs(rt['d'])*S:.2f}")
        h = r.F * S * 1700.0 / z
        print(f" z={z/1000:.0f} m 人 {h:.0f}px K/z={BETA*r.A*r.F*S/z:.2f} | " + " ; ".join(out))
