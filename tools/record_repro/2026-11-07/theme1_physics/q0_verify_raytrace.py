"""式 c=(A/J)(u-r)+AF/z を光線追跡で検算する。境界（θ→0、z→∞）も確かめる。"""
import csv
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from scipy.optimize import least_squares

import optics as op

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

rigs = op.default_rigs() + [
    op.rig_road_pitched("路面合焦・上3°", 3.0),
    op.rig_road_pitched("路面合焦・上5°", 5.0),
    op.rig_parallel("Hc0.65m路面", 0.0, Hc=650.0),
    op.rig_parallel("Hc0.26m路面", 0.0, Hc=260.0),
]
Hs = [0.0, 900.0, 1700.0, 2100.0, 5000.0, 5450.0]
zs = [5e3, 1e4, 2e4, 3e4, 5e4, 7e4, 1e5, 1e6, 1e9]

rows = []
print("=== 1. 式 vs 光線追跡（横幅 c_x、視差 d、行） ===")
print("名前 | θ[deg] | v[mm] | F[px] | A/J | u | ratio c_x(rt/式) min..max | ratio d/(βc) min..max | |Δrow|max | c_y/c_x min..max")
for R in rigs:
    rat_c, rat_d, drow, cycx = [], [], [], []
    for H in Hs:
        for z in zs:
            for xlat in [0.0, 0.1 * z]:
                Yc, Zc = R.to_camera(H, z)
                rt = op.raytrace(R, xlat, float(Yc), float(Zc))
                c_formula = float(R.blur_signed_rowz(op.C0 - R.F * Yc / Zc, Zc))
                drow.append(abs(rt["row0"] - (op.C0 - R.F * Yc / Zc)))
                if abs(c_formula) > 0.02:
                    rat_c.append(rt["c_x"] / abs(c_formula))
                    rat_d.append(abs(rt["d"]) / (op.BETA * abs(c_formula)))
                    cycx.append(rt["c_y"] / rt["c_x"])
                rows.append(dict(rig=R.name, H=H, z=z, x=xlat, c_formula=c_formula, c_x_rt=rt["c_x"], c_y_rt=rt["c_y"], d_rt=rt["d"]))
    print(f"{R.name} | {R.theta_deg:.3f} | {R.v:.3f} | {R.F:.1f} | {R.row_coef:.5f} | {R.u:.1f} | "
          f"{min(rat_c):.4f}..{max(rat_c):.4f} | {min(rat_d):.4f}..{max(rat_d):.4f} | {max(drow):.2e} | {min(cycx):.4f}..{max(cycx):.4f}")

with open(os.path.join(OUT, "q0_raytrace_vs_formula.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# 2. 路面合焦の θ,v を、ヒンジ則を使わず光線追跡で探す
print()
print("=== 2. 床（路面）に合焦する θ, v を光線追跡で探索（式は使わない） ===")
for Hc in [1300.0, 1330.0, 650.0, 260.0]:
    def resid(x):
        th_deg, v = x
        R = op.Rig("t", th_deg, v, Hc=Hc)
        out = []
        for z in np.linspace(max(5 * Hc, 3000), 100000, 30):
            Yc, Zc = R.to_camera(0.0, z)
            ax = np.array([R.A / 2, -R.A / 2])
            at = np.array([0.0, 0.0])
            col, row = op._trace(R, 0.0, float(Yc), float(Zc), ax, at)
            out.append(col[0] - col[1])
        return np.array(out)
    sol = least_squares(resid, [2.0, 91.0])
    print(f"Hc={Hc:.0f}: θ={sol.x[0]:.4f}° (asin(f/Hc)={np.degrees(np.arcsin(90/Hc)):.4f}°)  v={sol.x[1]:.4f} (f/cosθ={90/np.cos(np.arcsin(90/Hc)):.4f})  残差最大 {np.abs(sol.fun).max():.2e} px")

# 3. 境界条件
print()
print("=== 3. 境界条件 ===")
R0 = op.rig_untilted("θ=0・無限遠")
print("θ=0, v=f, z=1e12 mm のぼけ[px]（0 になるはず）:", float(R0.blur_signed(5000.0, 1e12)))
print("θ=0, 無限遠合焦, z=50m: 式", float(R0.blur_signed(5000.0, 5e4)), " AF/z=", R0.A * R0.F / 5e4,
      " 光線追跡", op.raytrace_world(R0, 5000.0, 5e4)["c_x"])
R1 = op.Rig("極小θ", 1e-9, 90.0)
print("θ=1e-9°: J=", R1.J, " A/J=", R1.row_coef, " 定数項=", R1.const_term, " c(5m,50m)=", float(R1.blur_signed(5000.0, 5e4)))
Ra = op.rig_parallel("a", 0.0)
print("路面合焦 z→∞ の行は地平線 u に漸近:", float(Ra.row(5000.0, 1e12)), " u=", Ra.u, " ぼけ→", float(Ra.blur_signed(5000.0, 1e12)))
print("路面合焦 路面の点のぼけ（0 になるはず）: z=20m", float(Ra.blur_signed(0.0, 2e4)), " z=100m", float(Ra.blur_signed(0.0, 1e5)))
try:
    op.rig_parallel("x", 1250.0)
except ValueError as e:
    print("|J|<=f の案の拒否:", e)
for bad in [(5000.0, 80.0)]:
    try:
        op.raytrace(op.rig_untilted("x"), 0.0, bad[0], bad[1])
    except ValueError as e:
        print("物点が f より手前の拒否:", e)

# 4. 現行リグの係数との照合
print()
print("=== 4. live 係数（2026-09-18）との照合 ===")
v_pos, u_pos, a_pos = 432599.32942224736, -63.927380677321885, -26.568072891235353
v_neg, u_neg, a_neg = 423627.23803714506, -21.368758126996866, -22.562290573120116
Rc = op.Rig("c", 30.0, 109.3, Hc=1330.0)
for name, v_, u_, a_ in [("pos", v_pos, u_pos, a_pos), ("neg", v_neg, u_neg, a_neg)]:
    J_emp = op.BETA * Rc.A * abs(a_)
    F528 = v_ / J_emp
    print(f"{name}: J=βA|ā|={J_emp:.1f} mm（f/sin30°=180）、F528=v/J={F528:.0f}（センサ 22.3mm/528 px で v/p={109.3/(22.3/528):.0f}）、"
          f"K=v/|ā|={v_/abs(a_):.0f}、u@1080={u_*op.R_1080_TO_528:.1f} 行")
print(f"モデル: tanψ={Rc.tan_psi:.4f}, u@1080(F5398)={Rc.u:.1f}, A/J={Rc.row_coef:.4f}, 2100mm の合焦行={float(Rc.row(1330.0 - Rc.J + 2100*Rc.tan_psi, 2100)):.1f}")
print(f"F の照合: v=90.22 で F(校正由来 p)={90.22/op.P_1080:.0f}、F(センサ幅 p)={90.22/op.P_1080_SENSOR:.0f}（差 {100*(op.P_1080/op.P_1080_SENSOR-1):.1f} %）")
