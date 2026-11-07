"""細かい確認: 光線追跡の格子の影響、実物大の灯器での天井高の逆算、物体側のぼけ径、灯器の縦エッジの長さ。"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

import optics as op

Ra = op.rig_parallel("(a)", 0.0)
print("=== 光線追跡の d/(βc)（格子を細かくすると 1 に近づくか） ===")
for n in [161, 401, 801, 1601, 3201]:
    Yc, Zc = Ra.to_camera(5000.0, 5e4)
    rt = op.raytrace(Ra, 0.0, float(Yc), float(Zc), n_grid=n)
    c = float(Ra.blur_signed(5000.0, 5e4))
    print(n, abs(rt["d"]) / (op.BETA * c))
Rc = op.Rig("c", 30.0, 109.3, Hc=1330.0)
Yc, Zc = Rc.to_camera(5000.0, 2e4)
rt = op.raytrace(Rc, 0.0, float(Yc), float(Zc), n_grid=1601)
print("θ=30°, 5 m, 20 m:", abs(rt["d"]) / (op.BETA * abs(float(Rc.blur_signed(5000.0, 2e4)))), " c_y/c_x=", rt["c_y"] / rt["c_x"], " cosθ=", np.cos(Rc.theta))

print()
print("=== 物体側に換算したぼけ径 c_obj = c·z/F = (A/J)|H−p|（合焦面が路面と平行なら距離によらない）[cm] ===")
rigs = [op.rig_parallel("(a) 路面", 0.0), op.rig_parallel("(b) 0.5m", 500.0), op.rig_parallel("(b) 1.0m", 1000.0), op.rig_parallel("(d1) 5m", 5000.0), op.rig_parallel("(e) (a)F2.8", 0.0, N=2.8)]
print("配置 | 信号灯器 5.0 | 片持 5.45 | 路側 2.1 | 頭 1.7 | 車上端 1.5 | 尾灯 0.9 | 路面 0")
for R in rigs:
    cells = []
    for H in [5000.0, 5450.0, 2100.0, 1700.0, 1500.0, 900.0, 0.0]:
        c = float(R.blur_signed(H, 5e4))
        cells.append(f"{abs(c)*5e4/R.F/10:.1f}")
    print(R.name, " | ".join(cells))
print("灯火径 30 cm・灯火の間隔 40 cm・灯火に隣が混ざり始める c_obj=2×(間隔−半径)=50 cm（R,B は半円なので片側 c/2）")

print()
print("=== 現行 θ=30° の物体側のぼけ径（距離で変わる）[cm] ===")
print("z | 信号灯器 | 片持 | 路側 | 頭")
for z in [20e3, 30e3, 50e3, 70e3, 100e3]:
    print(int(z / 1000), " | ".join(f"{abs(float(Rc.blur_signed(H, z)))*z/Rc.F/10:.0f}" for H in [5000.0, 5450.0, 2100.0, 1700.0]))

print()
print("=== 実物大の灯器を使うときの天井高（灯器の高さとカメラ高だけを k 分の 1。像の大きさ・ぼけ・視差は実物と同じ） ===")
print("天井 | k | カメラ高 | 灯器中心 | 灯器上端 | θ | v | c(50m) | c 比 | 枠に入る最小 z（水平・シフトなし）| 同（レンズ上げ 3 mm）")
c50 = float(Ra.blur_signed(5000.0, 5e4))
for ceil in [2400.0, 2500.0, 2700.0, 3000.0, 3500.0]:
    k = 5000.0 / (ceil - 100.0 - 225.0)
    Hc = 1300.0 / k
    H = 5000.0 / k
    R = op.rig_parallel("x", 0.0, Hc=Hc)
    c = op.raytrace_world(R, H, 5e4)["c_x"]
    zfit = R.F * (H + 225.0 - Hc) / 360.0 / 1000.0
    zfit2 = R.F * (H + 225.0 - Hc) / (360.0 + 3.0 / R.p) / 1000.0
    print(f"{ceil/1000:.1f} m | {k:.2f} | {Hc:.0f} mm | {H:.0f} | {H+225:.0f} | {R.theta_deg:.2f}° | {R.v:.2f} | {c:.2f} | {c/c50:.3f} | {zfit:.1f} m | {zfit2:.1f} m")
print("片持標識 0.9 m（上端 5.9 m・中心 5.45 m）: 天井 2.7 m（余裕 0.1 m）→ k=" + f"{5450/(2600-450):.2f}, カメラ高 {1300/(5450/(2150)):.0f} mm")
print("路側標識 0.6 m（中心 2.1 m）と人・車はカメラ高 1.3 m のまま等身大で置ける（高さ 2.4 m 以下）")

print()
print("=== 灯器の使える縦エッジ ===")
print("z | 1080: 像の縦(px) | 縦/K=25 | 縦/K=51（528 の K=25 と同じ物理窓） | 528: 像の縦 | 縦/K=25 | 灯器幅@528 | 幅/K=25 | 縦型の灯器（長辺が縦）の縦エッジ@1080, @528")
for z in [20e3, 30e3, 50e3, 70e3, 100e3]:
    h = Ra.F * 450 / z
    w = Ra.F * 1200 / z
    s = op.R_1080_TO_528
    print(f"{z/1000:.0f} | {h:.0f} | {h/25:.2f} | {h/51:.2f} | {h/s:.1f} | {h/s/25:.2f} | {w/s:.0f} | {w/s/25:.2f} | {w:.0f}, {w/s:.0f}")
print("窓の中に灯器の両端が同時に入る距離（528・K=25）: z≤" + f"{Ra.F*1200/s_ if False else 0}" if False else "")
s = op.R_1080_TO_528
print(f"528・K=25 で窓の幅が灯器の幅を超える距離: z > {Ra.F*1200/s/25/1000:.0f} m、灯火径の 2 倍を超える距離: z > {Ra.F*600/s/25/1000:.0f} m")
print(f"灯火の縁が端の縁の窓に入らない条件: 灯火と短辺の隙間 (1200-3*300-2*100)/2=50 mm → 灯火の縁と端の距離は常に {Ra.F*50/5e4:.1f} px@1080（50 m）で窓より遥かに小さい")

print()
print("=== 人の頭（参考）: 色視差で並ぶ δd の単位の訂正 ===")
for W, name in [(1200.0, "信号灯器"), (600.0, "路側標識")]:
    a1080 = op.BETA * Ra.A * 2.0 / W
    print(f"{name}: δd*(±2px@1080)={a1080:.3f} px@1080。B5 の 0.15 px@528 = {0.15*s:.2f} px@1080 → {0.15*s/a1080:.0f} 倍の差（前回の書き方 0.15 と {a1080:.3f} の比は {0.15/a1080:.0f} 倍）")
