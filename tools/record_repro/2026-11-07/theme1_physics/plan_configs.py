"""合焦面の配置ごとのぼけと視差（528 幅換算）。光線追跡で式を照合する。薄レンズ・実測なし。"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from optics import Rig, rig_parallel, rig_untilted, raytrace_world, BETA, P_1080, C0

S = 528.0 / 1080.0  # 1080 幅 → 528 幅
rigs = [
    rig_untilted("傾けない・無限遠"),
    rig_parallel("路面合焦", 0.0),
    rig_parallel("0.5 m 面", 500.0),
    rig_parallel("1.0 m 面", 1000.0),
    Rig("現行 θ=30°・v=109.3", 30.0, 109.3, Hc=1330.0),
    Rig("θ=15°・v=95(仮)", 15.0, 95.0, Hc=1300.0),
]
for r in rigs:
    print(
        f"\n=== {r.name}: θ={r.theta_deg:.2f}° v={r.v:.1f} F528={r.F * S:.0f} K528={BETA * r.A * r.F * S:.0f} "
        f"row係数 A/J={r.row_coef:.4f} tanψ={r.tan_psi if np.isfinite(r.tan_psi) else float('nan'):.4f} "
        f"J={r.J if np.isfinite(r.J) else float('inf'):.0f}"
    )
    # (名前, 上端H, 下端H, 縦の大きさ)
    cases = [
        ("人の頭 1.7", 1700.0, 1700.0),
        ("人の足元", 0.0, 1700.0),
        ("車の上端 1.5", 1500.0, 1500.0),
        ("車の足元", 0.0, 1500.0),
        ("コーン上端 0.7", 700.0, 700.0),
        ("路側標識中心 2.1", 2100.0, 600.0),
        ("片持標識中心 5.45", 5450.0, 900.0),
    ]
    for z in [10000.0, 20000.0, 30000.0, 40000.0]:
        parts = []
        for name, H, h in cases:
            rt = raytrace_world(r, H, z)
            c = rt["c_x"] * S
            d = rt["d"] * S
            box = r.F * S * h / z
            parts.append(f"{name}: c={c:.1f}px d={d:+.2f} ({c / box * 100:.1f}%)")
        print(f" z={z / 1000:.0f} m | K/z={BETA * r.A * r.F * S / z:.2f}px | " + " ; ".join(parts))
