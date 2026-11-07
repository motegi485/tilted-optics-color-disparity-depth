"""主張1・2の検算: 光線追跡（optics.raytrace）で K の θ 不変性、F の v 依存、
ぼけの差と 1/z の幅の比 A·F、(行, 1/z) 面の面積保存を確かめる。式 c=(A/J)(u-r)+AF/z は照合にだけ使う。
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
THETAS = [0.0, 0.5, 1.0, 3.97, 10.0, 19.11, 30.0, 40.0]
ROWS = [360.0, 60.0, 660.0]
ZS = np.array([2e3, 3e3, 5e3, 1e4, 2e4, 5e4, 1e5, 1e6])


def d_at(R, r, Z, x_lat=0.0, n=321):
    """行 r・奥行き Z の物点の d を光線追跡で返す（符号付き）"""
    Y = (C0 - r) * Z / R.F
    rt = op.raytrace(R, x_lat, float(Y), float(Z), n_grid=n)
    return rt["d"], rt["c_x"], rt["c_y"], rt["row0"]


print("=== 1a. v を固定して θ だけ振る: 行ごとに d を 1/Z に当てはめた傾き K_fit（px·mm） ===")
print("v | θ | 行 | K_fit | K_fit/(βAF) | K_fit/K_fit(θ=0) | 切片/β((A/J)(c0-r)+定数項) | 直線残差最大[px]")
for v in [90.22, 109.3]:
    K0 = {}
    for th in THETAS:
        R = op.Rig("t", th, v)
        for r in ROWS:
            d = np.array([d_at(R, r, Z)[0] for Z in ZS])
            x = 1.0 / ZS
            p = np.polyfit(x, d, 1)
            res = d - np.polyval(p, x)
            if th == 0.0:
                K0[r] = p[0]
            pred_int = R.row_coef * (C0 - r) + R.const_term
            ratio_int = p[1] / (BETA * pred_int) if abs(pred_int) > 1e-6 else np.nan
            print(f"{v} | {th} | {r:.0f} | {p[0]:.2f} | {p[0]/(BETA*R.A*R.F):.5f} | {p[0]/K0[r]:.6f} | {ratio_int:.5f} | {np.abs(res).max():.2e}")

print()
print("=== 1b. 横方向の位置 X を振っても同じか（θ=30°, v=109.3, 行 360。X は Z の 10 %） ===")
R = op.Rig("t", 30.0, 109.3)
for xfrac in [0.0, 0.05, 0.1]:
    d = np.array([op.raytrace(R, xfrac * Z, 0.0, float(Z), n_grid=321)["d"] for Z in ZS])
    p = np.polyfit(1 / ZS, d, 1)
    print(f"X/Z={xfrac}: K_fit={p[0]:.2f}, K_fit/(βAF)={p[0]/(BETA*R.A*R.F):.5f}")

print()
print("=== 1c. 距離の基準を変えると係数はどう見えるか（θ=30°, v=109.3）: 同じ行で 1/Z, 1/s(レンズ法線), 1/R(直線距離) ===")
print("行 | K(1/Z) | K(1/s) | K(1/R) | K(1/s)/K(1/Z) と cosθ−tanα·sinθ の照合")
for r in ROWS:
    tanA = (C0 - r) / R.F
    Zs = ZS
    d = np.array([d_at(R, r, Z)[0] for Z in Zs])
    s = Zs * (np.cos(R.theta) - tanA * np.sin(R.theta))
    rng = Zs * np.sqrt(1 + tanA**2)
    pz = np.polyfit(1 / Zs, d, 1)[0]
    ps = np.polyfit(1 / s, d, 1)[0]
    pr = np.polyfit(1 / rng, d, 1)[0]
    print(f"{r:.0f} | {pz:.1f} | {ps:.1f} | {pr:.1f} | {ps/pz:.5f} vs {np.cos(R.theta)-tanA*np.sin(R.theta):.5f}")

print()
print("=== 1d. v の変化で F が変わる分（同じ画素ピッチ p・薄レンズ。F=v/p, K=βAF）。K は px·m ===")
print("構成 | θ | v | F[px@1080] | K[px·m@1080] | K/K(無限遠θ=0) | 縦画角[°]")
cfgs = [
    ("θ=0・無限遠", 0.0, 90.0),
    ("θ=0・z_f=23 m", 0.0, 90.0 * 23000 / (23000 - 90)),
    ("θ=0・z_f=2.1 m", 0.0, 90.0 * 2100 / (2100 - 90)),
    ("θ=0・z_f=0.51 m", 0.0, 109.3),
    ("路面合焦 θ=3.97", 3.97, 90.0 / np.cos(np.deg2rad(3.97))),
    ("水平面(J=180) θ=30", 30.0, 90.0 / np.cos(np.deg2rad(30.0))),
    ("現行 θ=30 v=109.3", 30.0, 109.3),
    ("Fukino2026 実測 θ=20.3 z_id=106.5", 20.3, 106.5),
    ("Fukino2026 同定 θ=19.11 z_id=98.43", 19.11, 98.43),
]
Kref = BETA * (90 / 5.6) * (90.0 / op.P_1080) / 1000
for nm, th, v in cfgs:
    R = op.Rig(nm, th, v)
    K = BETA * R.A * R.F / 1000
    fov = 2 * np.degrees(np.arctan(360 / R.F))
    print(f"{nm} | {th} | {v:.2f} | {R.F:.0f} | {K:.2f} | {K/Kref:.3f} | {fov:.2f}")

print()
print("=== 1e. 前ぼけと後ろぼけで β（d/c_x）は違うか（光線追跡・θ=30°, v=109.3、行 360 付近） ===")
R = op.Rig("t", 30.0, 109.3)
for Z in [1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 5000.0, 20000.0]:
    for r in [60.0, 360.0, 660.0]:
        d, cx, cy, _ = d_at(R, r, Z, n=641)
        if cx > 2:
            print(f"Z={Z:.0f} r={r:.0f}: d={d:+.3f}, c_x={cx:.3f}, |d|/c_x={abs(d)/cx:.5f}, c_y/c_x={cy/cx:.5f}")

print()
print("=== 1f. レンズ傾斜と違い、センサ傾斜では K が行で変わるか（同じ薄レンズの手計算追跡） ===")


def sensor_tilt_d(th_deg, v0, r, Z, p=op.P_1080, f=90.0, A=90 / 5.6):
    """レンズは正対（Z 軸に垂直）、センサ面を θ 傾ける。主光線とセンサ面の交点で行を決める。"""
    th = np.deg2rad(th_deg)
    # センサ面: 点 (y_s cosθ, -v0 + y_s sinθ)、y_s はセンサ面内の座標（mm）
    y_s = (C0 - r) * p
    S = np.array([y_s * np.cos(th), -v0 + y_s * np.sin(th)])
    # 主光線は S とレンズ中心を通るので、物点は P = -S * (Z / -S[1])
    Yobj = -S[0] / (-S[1]) * Z  # S[1]<0
    s = Z
    m = f / (s - f)
    Yp, Zp = -m * Yobj, -m * Z
    ax = np.array([A / 2, -A / 2])
    # 開口 x の両端の光線を、センサ面で切る（y 成分はレンズ中心を通る面内なので x の広がりだけ見る）
    # センサ面の法線 n=(-sinθ, cosθ)。光線 (y,z)=(0,0)+t*(Yp,Zp) の x は ax+(Xp-ax)t で Xp=0
    n = np.array([-np.sin(th), np.cos(th)])
    t = np.dot(S, n) / (Yp * n[0] + Zp * n[1])
    Sx = ax * (1 - t)
    c = (Sx[0] - Sx[1]) / p
    return c * op.BETA, -S[1] / p  # d, 局所の F 相当


for th in [3.97, 20.0, 30.0]:
    print(f"θ={th}: 行 | K_fit/(βA·v0/p) | 主光線方向の局所 F/(v0/p)")
    for r in [0.0, 360.0, 720.0]:
        d = np.array([sensor_tilt_d(th, 90.2, r, Z)[0] for Z in ZS])
        p_ = np.polyfit(1 / ZS, d, 1)[0]
        Floc = sensor_tilt_d(th, 90.2, r, 1e4)[1]
        print(f"  {r:.0f} | {abs(p_)/(op.BETA*(90/5.6)*90.2/op.P_1080):.5f} | {Floc/(90.2/op.P_1080):.5f}")

print()
print("=== 2. 比の検算: 画面内のぼけの差 / 傾きで広がる 1/z の幅 = A·F か（光線追跡で合焦 Z を探す） ===")


def signed_d(R, r, invZ):
    """invZ→d。invZ=0 は扱えないので最小 1e-7 /mm"""
    Z = 1.0 / max(invZ, 1e-7)
    return d_at(R, r, Z)[0]


def pof_invZ(R, r):
    """行 r で d=0 になる 1/Z を 2 分探索。範囲外なら外挿（直線なので 2 点から）"""
    a, b = 1e-6, 1e-3
    da, db = signed_d(R, r, a), signed_d(R, r, b)
    return a - da * (b - a) / (db - da)


print("θ | v | r 範囲 | Δ(1/z)_POF[/m] | 固定 Z=5 m での Δc（光線追跡・符号付き）[px] | Δc/Δ(1/z) | A·F | 比 | 式 N_r/(J·F)[/m]")
for th, v in [(3.97, 90.22), (10.0, 91.0), (19.11, 98.43), (30.0, 109.3), (40.0, 117.0)]:
    R = op.Rig("t", th, v)
    r1, r2 = 0.0, 720.0
    i1, i2 = pof_invZ(R, r1), pof_invZ(R, r2)
    dZ = abs(i2 - i1)
    # c の符号付きは d の符号を流用
    d1, c1, _, _ = d_at(R, r1, 5000.0)
    d2, c2, _, _ = d_at(R, r2, 5000.0)
    sc1 = np.sign(d1) * c1
    sc2 = np.sign(d2) * c2
    dc = abs(sc1 - sc2)
    AF = R.A * R.F
    print(f"{th} | {v} | 0..720 | {dZ*1000:.4f} | {dc:.3f} | {dc/dZ:.1f} | {AF:.1f} | {dc/dZ/AF:.5f} | {720/(R.J*R.F)*1000:.4f}")

print()
print("=== 3. (行, 1/Z) 面での帯の面積は θ で変わらないか（光線追跡の d を格子で数える・帯 0.45<|d|<5・1080 幅の数値例） ===")
print("θ | v | 全面積 ∫∫ 1 dr d(1/Z) [行·/m]（1/Z≥0 内）| 理論 N_r·2·4.55/K | 1/Z の射影の幅[/m] | 1 距離あたりの行数 片側（理論 4.55/(β·A/J)）")
rr = np.linspace(0, 720, 145)
iz = np.linspace(0.0005, 2.5, 5000)  # /m
for th, v in [(0.0, 90.0 * 2100 / 2010), (0.0, 109.3), (3.97, 90.22), (10.0, 92.0), (19.11, 98.43), (30.0, 109.3)]:
    R = op.Rig("t", th, v)
    # 光線追跡を全格子で回すと重いので、行 3 本・1/Z 3 点で式と一致を確かめてから式で数える
    for r in [0.0, 720.0]:
        for z in [1500.0, 5000.0, 30000.0]:
            dd = d_at(R, r, z, n=161)
            if np.abs(dd[1]) > 1:
                fc = R.blur_signed_rowz(r, z)
                assert abs(abs(dd[1]) - abs(fc)) / abs(fc) < 0.01, (th, v, r, z, dd[1], fc)
    RR, II = np.meshgrid(rr, iz, indexing="ij")
    c = R.row_coef * (C0 - RR) + R.const_term + R.A * R.F * II / 1000
    d = BETA * c
    band = (np.abs(d) > 0.45) & (np.abs(d) < 5)
    dr = rr[1] - rr[0]
    di = iz[1] - iz[0]
    area = band.sum() * dr * di
    K = BETA * R.A * R.F / 1000
    th_area = 720 * 2 * 4.55 / K
    proj = band.any(axis=0)
    print(f"{th} | {v:.2f} | {area:.2f} | {th_area:.2f} | {proj.sum()*di:.3f} | {4.55/(BETA*R.row_coef) if R.row_coef>0 else np.inf:.0f}")

print()
print("=== 4. 同じ 1/Z 範囲で見た平均ぼけ・最大ぼけ（行と 1/Z が一様に分布する場合）: 傾けると大きくなるか ===")
print("条件: F=2639 相当の 528 幅、N_r=352、1〜4 m を想定（Δ=0.75 /m）。無傾斜は 1/z の中央に合焦")
AF = (90 / 5.6) * 2639 / 1000  # px·m
Nr = 352
iz = np.linspace(0.25, 1.0, 1501)
rows = np.linspace(-Nr / 2, Nr / 2, 353)
for name, slope, center in [("θ=0", 0.0, 0.625), ("θ=4°", np.sin(np.deg2rad(3.97)) / 5.6, 0.625), ("θ=19°", np.sin(np.deg2rad(19.11)) / 5.6, 0.625), ("θ=30°", 0.5 / 5.6, 0.625)]:
    RR, II = np.meshgrid(rows, iz, indexing="ij")
    c = slope * RR + AF * (II - center)
    print(f"{name}: 平均|c|={np.abs(c).mean():.2f} px, 最大|c|={np.abs(c).max():.2f} px, |c|<11.8 px かつ |c|>1.06 px の割合={(np.logical_and(np.abs(c)<11.8,np.abs(c)>1.06)).mean():.3f}")
