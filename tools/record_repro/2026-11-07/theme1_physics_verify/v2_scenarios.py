"""主張5の検算: 路面合焦・無限遠合焦・中間合焦・現行 θ=30° を、路面に立つ物体で比べる。
式は v1 で光線追跡と一致を確かめた c=(A/J)(c0-r)+定数項+AF/Z。要所は raytrace でも照合する。
単位は 1080 幅の px（F=v/p）。gate は [仮定] で 3 通り。
"""
import os
import sys

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "theme1_physics"))
import numpy as np

import optics as op

BETA = op.BETA
C0 = op.C0
GATES = [(0.45, 10.2, "下限0.45@1080"), (0.92, 10.2, "下限0.45@528相当"), (1.6, 10.2, "平坦域±0.78@528相当")]
OBJ = [("人1.7m", 0.0, 1.7), ("車1.5m", 0.0, 1.5), ("低い物0.3m", 0.0, 0.3), ("路側標識(1.8-2.4m)", 1.8, 2.4)]
ZS = [5, 8, 10, 15, 20, 30, 50, 80]


def bias_v(R0, c0_px):
    """R0 の v を c0_px だけ c が増える向きへ動かす（光軸方向の移動）"""
    dv = -c0_px * R0.f * R0.p / (R0.A * np.cos(R0.theta))
    return R0.v + dv, dv


Rroad = op.rig_parallel("路面合焦 θ=3.97", 0.0)
Rinf = op.rig_untilted("θ=0 無限遠")
R23 = op.rig_untilted("θ=0 z_f=23m", z_focus=23000.0)
Rcur = op.Rig("現行 θ=30 v=109.3 Hc=1.33", 30.0, 109.3, Hc=1330.0)
vb, dv1 = bias_v(Rinf, 3.3)
Rpast = op.Rig("θ=0 無限遠より先(d+1.4)", 0.0, vb)
vb2, dv2 = bias_v(Rroad, 3.3)
Rroadb = op.Rig("路面合焦＋偏り(d+1.4)", Rroad.theta_deg, vb2)
Rs = op.rig_hinge_slope("路面ヒンジ tanψ=0.05", 1300.0, 0.05)
CFG = [Rroad, Rinf, R23, Rcur, Rpast, Rroadb, Rs]
print("構成 | θ | v | F | K[px·m] | A/J | u | 動かした v[mm]")
for R, dv in zip(CFG, [0, 0, 0, 0, dv1, dv2, 0]):
    print(f"{R.name} | {R.theta_deg:.3f} | {R.v:.3f} | {R.F:.0f} | {BETA*R.A*R.F/1000:.2f} | {R.row_coef:.5f} | {R.u if np.isfinite(R.u) else float('nan'):.1f} | {dv:+.3f}")

# 光線追跡との照合
for R in CFG:
    for H, z in [(0.0, 20000.0), (1700.0, 20000.0), (900.0, 8000.0), (0.0, 50000.0)]:
        rt = op.raytrace_world(R, H, z)
        c = float(R.blur_signed(H, z))
        if abs(c) > 0.5:
            assert abs(rt["c_x"] - abs(c)) / abs(c) < 0.01, (R.name, H, z, rt["c_x"], c)
            assert abs(abs(rt["d"]) / (BETA * abs(c)) - 1) < 0.01
print("光線追跡と式の照合 OK（|c|>0.5 px で 1 % 以内）")


def obj_stats(R, hlo, hhi, z_m, glo, ghi, n=400):
    z = z_m * 1000.0
    Hs = np.linspace(hlo * 1000, hhi * 1000, n)
    rows = np.array([float(R.row(H, z)) for H in Hs])
    vis = (rows >= 0) & (rows <= 720)
    if vis.sum() == 0:
        return None
    cs = np.array([float(R.blur_signed(H, z)) for H in Hs])
    d = BETA * cs
    frac_vis = vis.mean()
    in_band = (np.abs(d) > glo) & (np.abs(d) < ghi) & vis
    return dict(frac_vis=frac_vis, frac_band=in_band.sum() / vis.sum(), mean_c=np.abs(cs[vis]).mean(), max_c=np.abs(cs[vis]).max(), dmax=d[vis][np.argmax(np.abs(d[vis]))], dmin=d[vis][np.argmin(np.abs(d[vis]))], px=vis.sum() / n * abs(rows[-1] - rows[0]), signs=(np.sum(d[vis] > 0), np.sum(d[vis] < 0)))


print()
print("=== 物体ごとの比較（行で一様に数える。画面外は除く）。帯内割合は下限の 3 仮定 ===")
for (nm, hlo, hhi) in OBJ:
    print(f"--- {nm} ---")
    print("z[m] | 構成 | 見える割合 | 像の高さ[px] | |c| 平均/最大[px] | d 最小/最大[px] | 帯内割合 下限0.45 / 0.92 / 1.6 | 符号(+,-)")
    for z in ZS:
        for R in CFG:
            s = [obj_stats(R, hlo, hhi, z, g[0], g[1]) for g in GATES]
            if s[0] is None:
                print(f"{z} | {R.name} | 画面外")
                continue
            st = s[0]
            print(f"{z} | {R.name} | {st['frac_vis']:.2f} | {st['px']:.0f} | {st['mean_c']:.2f}/{st['max_c']:.2f} | {st['dmin']:+.2f}/{st['dmax']:+.2f} | {s[0]['frac_band']:.2f}/{s[1]['frac_band']:.2f}/{s[2]['frac_band']:.2f} | {st['signs'][0]},{st['signs'][1]}")

print()
print("=== 無傾斜・無限遠の帯（全行で同じ）: K/5 から K/0.45 まで ===")
for label, K in [("1080 F4445", BETA * Rinf.A * Rinf.F / 1000), ("528 F2173", BETA * Rinf.A * Rinf.F / 1000 / 2.045)]:
    print(f"{label}: K={K:.2f} px·m, |d|<5 の近端 {K/5:.1f} m（1080 では d<10.2 で {K/10.2:.1f} m）, |d|>0.45 の遠端 {K/0.45:.1f} m, Δ(1/z)={(5-0.45)/K:.3f} /m")

print()
print("=== 剛体ピッチ: カメラごと φ 傾けても、行と d から出す Z_c は変わらないか（路面合焦・光線追跡）。接地点法と比べる ===")
print("φ[°] | z[m] | H[m] | 行 | d[px] | Z_c 推定[m] | 水平距離に換算する真値 z | 誤差[%] | 接地点法（ピッチ 0 を仮定）の推定[m] | 誤差[%]")
for phi in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    R = op.Rig("t", Rroad.theta_deg, Rroad.v, phi_deg=phi)
    R0 = Rroad
    for z_m, H in [(20, 1.7), (20, 0.9), (50, 1.7), (50, 0.9)]:
        z = z_m * 1000.0
        rt = op.raytrace_world(R, H * 1000.0, z)
        r = rt["row0"]
        # raytrace の d の符号: c>0（手前）で負。式の d は c と同符号
        c_sign = -np.sign(rt["d"])
        d_sgn = c_sign * abs(rt["d"])
        Zest = R.A * R.F * BETA / (d_sgn - BETA * (R.row_coef * (C0 - r) + R.const_term))
        Ztrue = float(R.to_camera(H * 1000.0, z)[1])
        # 接地点法: 足元の行
        rg = float(R.row(0.0, z))
        zg = R.F * R.Hc / (rg - C0) if rg > C0 else np.inf
        print(f"{phi} | {z_m} | {H} | {r:.1f} | {d_sgn:+.3f} | {Zest/1000:.3f} | {z_m} | {100*(Zest-Ztrue)/Ztrue:+.3f}（Z_c 基準）/{100*(Zest-z)/z:+.3f}（z 基準） | {zg/1000:.2f} | {100*(zg-z)/z:+.1f}")

print()
print("=== ピッチで符号が反転する高さ（H/z<tanφ）。路面合焦で φ>0（上向き）。下限の範囲も示す ===")
for phi in [0.25, 0.5, 1.0, 2.0]:
    t = np.tan(np.deg2rad(phi))
    print(f"φ={phi}°: 路面全体の c={Rroad.row_coef*Rroad.F*t:.2f} px, d={BETA*Rroad.row_coef*Rroad.F*t:.2f} px@1080。符号反転の高さ: " + ", ".join(f"{z} m で H<{z*t:.2f} m" for z in [10, 20, 30, 50]))

print()
print("=== 機械的な変位の感度（光線追跡の差分。行 360、z=20 m の d と距離の変化。薄レンズ） ===")
print("θ | v | δv=10 µm の δd[px@1080], δ(1/z)[/m], 20 m の δz[m] | δθ=0.01° の δd, δ(1/z), 20 m の δz | 1°あたりの δc（定数項）")
for th, v in [(0.0, 90.0), (3.97, 90.22), (19.11, 98.43), (30.0, 109.3)]:
    Ra = op.Rig("a", th, v)
    Yc = 0.0
    Z = 20000.0
    base = op.raytrace(Ra, 0.0, Yc, Z, n_grid=321)
    Rb = op.Rig("b", th, v + 0.01)
    dv_ = op.raytrace(Rb, 0.0, Yc, Z, n_grid=321)
    Rc = op.Rig("c", th + 0.01, v)
    dth = op.raytrace(Rc, 0.0, Yc, Z, n_grid=321)
    K = BETA * Ra.A * Ra.F
    # d の符号付き差
    dd_v = abs(dv_["d"] - base["d"])
    dd_t = abs(dth["d"] - base["d"])
    print(f"{th} | {v} | {dd_v:.4f}, {dd_v/K*1000:.5f}, {dd_v/K*Z*Z/1000/1000:.3f} | {dd_t:.4f}, {dd_t/K*1000:.5f}, {dd_t/K*Z*Z/1000/1000:.3f} | {dd_t/0.01/BETA:.2f} px/°")
print("比較: ステレオのヨー δψ=0.01° の視差変化 = F·δψ =", f"{4456*np.deg2rad(0.01):.3f} px@F4456。基線 b の δ(1/z)=δψ/b: b=350 mm で {np.deg2rad(0.01)/350*1000:.5f} /m")
print("熱膨張の例 [仮定]: 鏡筒 90 mm・23 ppm/K で 10 K なら δv=20.7 µm → δ(1/z)=δv·cosθ/(f·v)=", f"{20.7e-3/(90*90)*1000:.5f} /m → 20 m で {20.7e-3/(90*90)*1000*400:.2f} m")

print()
print("=== ぼけ予算と開口: 高さ H_t 以下の物体のぼけを z_min 以遠で c_t 以下に抑えるとき許される K ===")
print("無傾斜・無限遠: c=AF/z ≤ c_t → K_max=β·c_t·z_min。路面合焦: c=AF·H/(Hc·z) → K_max=β·c_t·z_min·Hc/H_t")
for ct, zmin in [(5.0, 10.0), (5.0, 20.0), (10.0, 10.0)]:
    K0 = BETA * ct * zmin
    A0 = K0 * 1000 / (BETA * 4456)
    print(f"c_t={ct} px@1080, z_min={zmin} m: 無傾斜 K_max={K0:.1f} px·m (A={A0:.1f} mm, F{90/A0:.1f})")
    for Ht in [0.3, 0.5, 1.0, 1.3, 1.7]:
        K1 = K0 * 1.3 / Ht
        A1 = K1 * 1000 / (BETA * 4456)
        print(f"   H_t={Ht} m: 路面合焦 K_max={K1:.1f} px·m (A={A1:.1f} mm, F{90/A1:.1f}), 倍率 {1.3/Ht:.2f}")

print()
print("=== 合焦面を上げる: 平行に上げる場合とヒンジを保って回す場合の、人(0〜1.7 m)の最大ぼけ（路面合焦の頭のぼけとの比） ===")
Rpar = op.rig_parallel("平行 H_f=0.85", 850.0)
print("z[m] | 路面合焦の頭 c[px] | 平行 H_f=0.85 c, 比 | ヒンジ tanψ=0.05 c, 比, 合焦面の高さ[m] | ヒンジ tanψ=0.02 c, 比, 高さ")
Rs2 = op.rig_hinge_slope("tanψ=0.02", 1300.0, 0.02)
for z in [10, 15, 20, 30, 50]:
    zmm = z * 1000.0
    c0_ = abs(float(Rroad.blur_signed(1700.0, zmm)))
    def maxc(R):
        return max(abs(float(R.blur_signed(0.0, zmm))), abs(float(R.blur_signed(1700.0, zmm))))
    print(f"{z} | {c0_:.2f} | {maxc(Rpar):.2f}, {maxc(Rpar)/c0_:.2f} | {maxc(Rs):.2f}, {maxc(Rs)/c0_:.2f}, {float(Rs.pof_height(zmm))/1000:.2f} | {maxc(Rs2):.2f}, {maxc(Rs2)/c0_:.2f}, {float(Rs2.pof_height(zmm))/1000:.2f}")

print()
print("=== 接地点法と色視差の 1 距離あたりの誤差（路面合焦 F=4456、Hc=1.3） ===")
K = BETA * Rroad.A * Rroad.F / 1000
print(f"K={K:.2f} px·m, 1 行の誤差と等価な視差 = β·A/Hc = {BETA*Rroad.A/1300:.5f} px/行, δd=0.1 px ≡ {0.1/(BETA*Rroad.A/1300):.1f} 行")
print("δd[px] | 等価な接地行の誤差[行] | 枠の下端の誤差が 3,5,10 行のとき どちらが小さいか（比 δz_d/δz_g）")
for dd in [0.021, 0.05, 0.1, 0.15]:
    eq = dd / (BETA * Rroad.A / 1300)
    print(f"{dd} | {eq:.1f} | " + ", ".join(f"δr={dr}: {eq/dr:.2f}" for dr in [3, 5, 10]))
print("交点ピッチ φ*（色視差の誤差＝接地点法のピッチ誤差）= δd·J/K: " + ", ".join(f"δd={dd}: {np.degrees(dd*1.3/K):.3f}°" for dd in [0.021, 0.05, 0.1, 0.15]))
print(f"足元が写る距離（水平・中央の地平線）: z≥Hc·F/360 = {1.3*Rroad.F/360:.1f} m、θ=30° リグ F=5398: {1.33*5398/360:.1f} m")

print()
print("=== ZNCC の窓の中の視差の差（θ による）: K_w 行の窓で d が変わる量 = β(sinθ/N)·K_w ===")
for th in [0.0, 3.97, 10.0, 19.11, 30.0, 40.0]:
    s = BETA * np.sin(np.deg2rad(th)) / 5.6
    print(f"θ={th}: {s:.5f} px/行, 窓 25 行で {s*25:.3f} px, 窓 51 行で {s*51:.3f} px")

print()
print("=== (行, 距離) の帯の幅: θ による 1 距離あたりの行数（帯 0.45〜5、528 幅相当） ===")
for th in [3.97, 10.0, 19.11, 30.0]:
    s = BETA * np.sin(np.deg2rad(th)) / 5.6
    print(f"θ={th}: 片側 {4.55/s:.0f} 行、下限の幅 ±0.45 で {0.9/s:.0f} 行、平坦域 ±0.78 で {1.56/s:.0f} 行（528 の 352 行に対して）")
