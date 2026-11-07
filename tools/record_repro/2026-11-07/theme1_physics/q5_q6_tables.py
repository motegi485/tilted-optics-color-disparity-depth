"""Q5: 距離の誤差の比較。Q6: 屋内の相似則。Q7: 経路ごとの像の大きさ。"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

import optics as op

Ra = op.rig_parallel("(a)", 0.0)
K1080 = op.BETA * Ra.A * Ra.F
K528 = K1080 / op.R_1080_TO_528
ZS = [20e3, 30e3, 50e3, 70e3, 100e3]

print(f"(a) A={Ra.A:.2f} F={Ra.F:.0f} K@1080={K1080:.0f} K@528={K528:.0f} px·mm")
print("K/z[px@1080 / px@528]:", " | ".join(f"{int(z/1000)}m {K1080/z:.3f}/{K528/z:.3f}" for z in ZS))
print()
print("=== 式: d = β(A/J)(u-r) + K/z。距離の情報は K/z だけ。δz/z = δd·z/K（線形）、z_est = K/(K/z ± δd)（非線形）===")


def err(dd, K, z):
    """視差に +dd / -dd が乗ったときの距離の相対誤差（非線形）。発散したら inf"""
    p = K / (K / z + dd) / z - 1
    den = K / z - dd
    m = (K / den / z - 1) if den > 0 else np.inf
    return p, m


print("シナリオ（δd は視差そのものの誤差。px は処理幅） | " + " | ".join(f"{int(z/1000)} m" for z in ZS) + " | 発散距離 K/δd [m]")
sc = [
    ("前回 δd=0.1 px@1080（仮定）", 0.1, K1080),
    ("時間ノイズ σ_d 中央値 0.021 px@528（B5）", 0.021, K528),
    ("σ_d 平均 0.17 px@528（B5 全距離平均）", 0.17, K528),
    ("B5 の校正範囲内 MAE を視差換算 0.15 px@528", 0.15, K528),
    ("同じ 0.15 px@528 を K 実測 pos で", 0.15, 16283 * 90.22 / 109.3),
    ("合成の外形の偏り 0.05 px@528", 0.05, K528),
    ("合成の外形の偏り 0.19 px@528（50m・青灯）", 0.19, K528),
    ("合成の外形の偏り 0.37 px@528（100m）", 0.37, K528),
    ("4:2:0 の 1080 幅の偏り 0.6 px@1080", 0.6, K1080),
    ("F2.8 にして 0.15 px@528", 0.15, 2 * K528),
    ("静止画 δd=0.15 px@6000 幅（仮定・未検証）", 0.15, K1080 * 6000 / 1080),
    ("静止画 δd=0.5 px@6000 幅（仮定・未検証）", 0.5, K1080 * 6000 / 1080),
]
for name, dd, K in sc:
    cells = []
    for z in ZS:
        p, m = err(dd, K, z)
        ms = "∞" if not np.isfinite(m) else f"{100*m:+.0f}%"
        cells.append(f"{100*p:+.0f}%/{ms}")
    print(f"{name} | " + " | ".join(cells) + f" | {K/dd/1000:.0f}")

print()
print("=== 枠の大きさからの距離: δz/z = δw/w ⊕ σ_W/W ===")
OBJ = {"信号灯器 1.2m": 1200.0, "路側標識 0.6m": 600.0, "片持標識 0.9m": 900.0}
print("対象 | δw | " + " | ".join(f"{int(z/1000)} m" for z in ZS))
for name, W in OBJ.items():
    for lab, fn in [("±2 px（前回）", lambda w: 2.0 / w), ("±5 %（仮定）", lambda w: 0.05), ("±1 px+ぼけの c/4（仮定）", None)]:
        cells = []
        for z in ZS:
            w = Ra.F * W / z
            if fn is None:
                H = {"信号灯器 1.2m": 5000.0, "路側標識 0.6m": 2100.0, "片持標識 0.9m": 5450.0}[name]
                c = float(Ra.blur_signed(H, z))
                e = np.hypot(1.0, c / 4) / w
            else:
                e = fn(w)
            cells.append(f"{100*e:.1f}%")
        print(f"{name} | {lab} | " + " | ".join(cells))
print("実寸の事前分布 σ_W/W を足すと: 5 % → 全距離で 5 % 以上。10 % → 全距離で 10 % 以上。サイズ区分が分からない標識（0.6 と 0.9 の取り違え）→ −33 %／+50 %（距離によらない）")

print()
print("=== 色視差が枠の大きさに並ぶ条件 ===")
print("δd*(px@1080) = K·(δz/z)/z。枠の誤差が『幅の px 一定』なら z によらない（前回の式）が、『割合一定』または『実寸の不確かさ』なら z に反比例して緩む。")
print("対象 | 枠±2px: δd* | 割合 5%: δd*(20/50/100m) | 割合 10%: δd*(20/50/100m)")
for name, W in OBJ.items():
    a = op.BETA * Ra.A * 2.0 / W
    def g(frac):
        return " / ".join(f"{K1080*frac/z:.3f}" for z in [20e3, 50e3, 100e3])
    print(f"{name} | {a:.3f} | {g(0.05)} | {g(0.10)}")
print("同じ δd* を px@528 に直すと 1/2.045 倍。")
print("色視差が割合誤差 ε に勝つ距離 z < K·ε/δd:")
for dd, K, lab in [(0.1, K1080, "0.1 px@1080"), (0.03, K1080, "0.03 px@1080"), (0.15, K528, "0.15 px@528"), (0.05, K528, "0.05 px@528")]:
    print(f"  δd={lab}: ε=5% → {K*0.05/dd/1000:.1f} m, ε=10% → {K*0.10/dd/1000:.1f} m, ε=20% → {K*0.2/dd/1000:.1f} m, 区分取り違え 50% → {K*0.5/dd/1000:.1f} m")

print()
print("=== 枠内集約で下がる誤差・下がらない誤差（信号灯器・528 幅 K=25 相当の独立窓の数）===")
print("z | 像の縦 h@528 | 窓 K=25 より長いか | 独立窓の数 N（2 端 × max(1,h/K)）| 1/√N | 時間・空間のノイズ 0.15 → 0.15/√N | 距離誤差（下がる分だけ・線形）")
for z in ZS:
    h = Ra.F * 450 / z / op.R_1080_TO_528
    Nw = 2 * max(1.0, h / 25.0)
    dd = 0.15 / np.sqrt(Nw)
    print(f"{z/1000:.0f} | {h:.1f} | {'○' if h>=25 else '×'} | {Nw:.1f} | {1/np.sqrt(Nw):.2f} | {dd:.3f} | {100*dd*z/K528:.0f}%")
print("標識（60 cm）の独立窓: h=w なので周の縦エッジの長さ ≈ 2h。文字があれば増える。")
for z in ZS:
    h = Ra.F * 600 / z / op.R_1080_TO_528
    Nw = 2 * max(1.0, h / 25.0)
    print(f"  {z/1000:.0f} m: h@528={h:.1f}, N={Nw:.1f}")

print()
print("=== 下がらない誤差の大きさ（視差への換算 px@1080 と距離の割合） ===")
print("(1) レンズとセンサの間隔の変化 δv: δd = βA·δv/(f p)、δz/z = δv·z/(f v)（A・画素によらない）")
for dv in [0.005, 0.010, 0.020, 0.050, 0.100]:
    dd = op.BETA * Ra.A * dv / (Ra.f * Ra.p)
    print(f"  δv={dv*1000:.0f} µm: δd={dd:.3f} px@1080 | " + " | ".join(f"{int(z/1000)}m {100*dv*z/(Ra.f*Ra.v):.1f}%" for z in ZS))
print("  アルミの線膨張 23 ppm/K [一般知識] で 90 mm は 2.1 µm/K。10 K で 21 µm ≒ 50 m で 13 %。")
print("(2) A/J（傾き）の校正誤差: 行の項の割合を掛ける。(H-Hc)/J 倍で距離に効く")
for R, name in [(op.rig_parallel("a", 0.0), "(a)"), (op.rig_parallel("b", 500.0), "(b0.5)"), (op.rig_parallel("b", 1000.0), "(b1.0)")]:
    J = R.J
    print(f"  {name}: J={J:.0f} mm, A/J を 1 % 誤る → 距離 " + " | ".join(f"{lab} {100*0.01*(H-R.Hc)/J:.1f}%" for H, lab in [(5000.0, "信号灯器"), (5450.0, "片持"), (2100.0, "路側"), (1700.0, "頭")]))
print("  θ の誤差 0.1° → A/J の誤差: (a) " + f"{100*np.deg2rad(0.1)/np.tan(Ra.theta):.1f} %、(b1.0) {100*np.deg2rad(0.1)/np.tan(op.rig_parallel('b',1000.0).theta):.2f} %")
print("(3) u（地平線の行に当たる係数）の誤差 δu: δd=β(A/J)δu。10 行で " + f"{op.BETA*Ra.row_coef*10:.3f} px@1080（(a)）→ 50 m で {100*op.BETA*Ra.row_coef*10*5e4/K1080:.1f} %")
print("    δz/z = δu·z/(F·J) (a): 10 行, 50 m → " + f"{100*10*5e4/(Ra.F*Ra.J):.1f}%")
print("(4) 倍率色収差: R と B の像の定数ずれ。1 µm で " + f"{0.001/Ra.p:.3f} px@1080。校正で吸収できるのは同じ行・列で変わらない成分だけ [推論]")

print()
print("=== 比較相手: 既知の取付け高さからの距離（宙の物体の『接地点』に当たる）===")
print("z = F·(H−Hc)/(u−r)。δz/z = (δr + F·δφ)·z/(F·(H−Hc)) ⊕ δH/(H−Hc)")
for z in ZS:
    t = (2.0 * z / (Ra.F * 3700.0), Ra.F * np.deg2rad(0.5) * z / (Ra.F * 3700.0), 300.0 / 3700.0)
    print(f"  z={z/1000:.0f} m 信号灯器: 行誤差 2 px → {100*t[0]:.1f}%, ピッチ 0.5° → {100*t[1]:.0f}%, 高さ ±0.3 m → {100*t[2]:.1f}%")

print()
print("=== 距離ごとの 3 手法（信号灯器・相対誤差）。色視差は外形の 2 端のみ、枠は幅 ===")
print("z | 色視差 0.1px@1080 | 色視差 0.15/√N px@528（ランダム扱い） | 色視差 0.15 px@528（系統扱い） | 枠 ±2 px | 枠 ±5 % | 枠 ±5 % + 実寸 ±10 % | 取付け高さ（ピッチ既知 ±0.1°・高さ ±0.3 m）")
for z in ZS:
    h = Ra.F * 450 / z / op.R_1080_TO_528
    Nw = 2 * max(1.0, h / 25.0)
    a = 0.1 * z / K1080
    b = 0.15 / np.sqrt(Nw) * z / K528
    c = 0.15 * z / K528
    w = Ra.F * 1200 / z
    e1 = 2 / w
    e2 = 0.05
    e3 = np.hypot(0.05, 0.10)
    e4 = np.hypot(np.hypot(2 * z / (Ra.F * 3700), Ra.F * np.deg2rad(0.1) * z / (Ra.F * 3700)), 300.0 / 3700)
    print(f"{z/1000:.0f} | {100*a:.0f}% | {100*b:.0f}% | {100*c:.0f}% | {100*e1:.1f}% | {100*e2:.0f}% | {100*e3:.0f}% | {100*e4:.0f}%")

print()
print("=== 視差の中の情報の割合: K/z の項と行の項（信号灯器） ===")
for R, name in [(op.rig_parallel("a", 0.0), "(a)"), (op.rig_parallel("b", 1000.0), "(b1.0)"), (op.rig_untilted("c"), "θ=0 無限遠")]:
    s = []
    for z in ZS:
        d = op.BETA * float(R.blur_signed(5000.0, z))
        K = op.BETA * R.A * R.F
        s.append(f"{int(z/1000)}m d={d:.2f} K/z={K/z:.3f}")
    print(name, " | ".join(s))

# -------------------------------------------------------------------
print()
print("=== Q6: 相似則（光線追跡で確かめる）===")
print("実物: 路面合焦 Hc=1.3 m、信号灯器 H=5 m、1.2×0.45 m、F5.6、z")
real = Ra
ref = {}
for z in ZS:
    rt = op.raytrace_world(real, 5000.0, z)
    ref[z] = (rt["c_x"], real.F * 450 / z, rt["d"], rt["c_y"])
print("実物:", " | ".join(f"{int(z/1000)}m c={ref[z][0]:.2f} h={ref[z][1]:.1f} d={ref[z][2]:.2f}" for z in ZS))
print()
print("案 | Hc | H | 物体 W×h | 開口 | θ | v | F | z の縮尺 | c 比(実物比) | h 比 | d 比 | c_y/c_x | δz/z 比 | 枠に入る最小 z | 回折(エアリー径)px@1080 半開口横 | 光量 | 必要な天井（灯器上端） | 廊下 L=40m で届く実距離")
opts = [
    ("A 縦だけ k=2（実物大の灯器・カメラを下げる）", 2.0, 1.0, 1.0),
    ("A' 縦だけ k=2.5", 2.5, 1.0, 1.0),
    ("B 全縮尺 s=2", 2.0, 2.0, 2.0),
    ("B' 全縮尺 s=2.5", 2.5, 2.5, 2.5),
    ("B'' 全縮尺 s=5", 5.0, 5.0, 5.0),
    ("C 全縮尺 s=2.5・開口はそのまま F5.6", 2.5, 2.5, 1.0),
]
for name, kv, sx, sa in opts:
    Hc = 1300.0 / kv
    H = 5000.0 / kv
    W = 1200.0 / sx
    h = 450.0 / sx
    N = 5.6 * sa
    R = op.rig_parallel(name, 0.0, Hc=Hc, N=N)
    zr = []
    for z in ZS:
        zm = z / sx
        rt = op.raytrace_world(R, H, zm)
        zr.append((rt["c_x"] / ref[z][0], (R.F * h / zm) / ref[z][1], rt["d"] / ref[z][2], rt["c_y"] / rt["c_x"]))
    zr = np.array(zr)
    Kmodel = op.BETA * R.A * R.F
    rel = (1 / Kmodel) * (ZS[0] / sx) / ((1 / K1080) * ZS[0])
    zf = R.F * (H + h / 2 - Hc) / 360.0 / 1000.0
    airy_um = 2.44 * 0.55 * N
    airy_half_um = 2.44 * 0.55 * (2 * N)
    p_um = R.p * 1000
    print(f"{name} | {Hc:.0f} | {H:.0f} | {W:.0f}×{h:.0f} | F{N:.1f} | {R.theta_deg:.2f} | {R.v:.2f} | {R.F:.0f} | 1/{sx:g} | {zr[:,0].min():.3f}-{zr[:,0].max():.3f} | {zr[:,1].min():.3f}-{zr[:,1].max():.3f} | {zr[:,2].min():.3f}-{zr[:,2].max():.3f} | {zr[:,3].min():.2f}-{zr[:,3].max():.2f} | {rel:.3f} | {zf:.1f} m | {airy_um/p_um:.2f} / {airy_half_um/p_um:.2f} | -{2*np.log2(sa):.1f} 段 | {(H+h/2)/1000:.2f} m | {40*sx:.0f} m")

print()
print("天井高から逆算（信号灯器の上端 5.225 m を k 分の 1 にする。カメラ高は 1.3/k）")
for ceil in [2400.0, 2700.0, 3000.0, 3500.0]:
    for margin in [100.0]:
        k = 5225.0 / (ceil - margin)
        Hc = 1300.0 / k
        R = op.rig_parallel("x", 0.0, Hc=Hc)
        print(f"天井 {ceil/1000:.1f} m（余裕 0.1 m）: k={k:.2f}, カメラ高 {Hc:.0f} mm, θ={R.theta_deg:.2f}°, v={R.v:.2f} mm, 灯器中心 {5000/k:.0f} mm, 水平で枠に入る z≥{R.F*(5225/k-Hc)/360/1000:.1f} m, レンズ上げ 2mm で z≥{R.F*(5225/k-Hc)/(360+2/R.p)/1000:.1f} m")
print("片持標識（上端 5.9 m）: 天井 2.7 m → k=" + f"{5900/2600:.2f}, カメラ高 {1300/(5900/2600):.0f} mm")

print()
print("v を縮められない場合に残る配置（床と平行な合焦面 ⇔ v=f/cosθ）: J とカメラ高")
for v in [90.22, 91.0, 92.0, 94.0, 96.0, 100.0, 105.0, 109.3]:
    th = np.degrees(np.arccos(90.0 / v))
    J = 90.0 / np.sin(np.radians(th))
    print(f"v={v:.2f}: θ={th:.1f}°, J={J:.0f} mm（床に合わせるならカメラ高={J:.0f} mm。高さ 1.3 m のカメラなら床から {1300-J:.0f} mm の面）, 0°のときの合焦距離 {90*v/(v-90)/1000 if v>90.001 else np.inf:.2f} m")

print()
print("=== Q7: 経路ごとの像の大きさと、ぼけの上限（信号灯器・路面合焦） ===")
paths = [("動画 1080×720 クロップ", 1.0), ("528 幅（現行の処理）", 528 / 1080), ("YOLO 640 入力（1080 を長辺 640）", 640 / 1080), ("YOLO 1280 入力（1080→1280 拡大に相当）", 1280 / 1080), ("静止画 6000×4000 等倍", 6000 / 1080)]
print("経路 | " + " | ".join(f"{int(z/1000)}m w×h, c" for z in ZS) + " | 灯火径(50m)")
for name, s in paths:
    cells = []
    for z in ZS:
        w = Ra.F * 1200 / z * s
        h = Ra.F * 450 / z * s
        c = float(Ra.blur_signed(5000.0, z)) * s
        cells.append(f"{w:.0f}×{h:.0f}, {c:.1f}")
    print(f"{name} | " + " | ".join(cells) + f" | {Ra.F*300/5e4*s:.1f}")
print()
print("等価サイズ（全円の MTF50 を基準・h/(0.71c)。ただし画素の方が粗ければ画素で決まる）: 経路ごとに信号灯器・人（1.7m）の縦")
for name, s in paths:
    cells = []
    for lab, H, hh in [("灯器", 5000.0, 450.0), ("人", 1700.0, 1700.0)]:
        vals = []
        for z in ZS:
            h = Ra.F * hh / z * s
            c = float(Ra.blur_signed(H, z)) * s
            vals.append(h / max(1.0, 0.71 * c))
        cells.append(lab + " " + "/".join(f"{v:.0f}" for v in vals))
    print(f"{name} | " + " | ".join(cells))
print()
print("クロマの解像度（横のサンプル数／センサ幅）: 動画 4:2:0 = 540、静止画 JPEG 4:2:2 = 3000（2300.JPG で確認）、RAW の R/B = 3000（ベイヤ・一般知識）")
print("画素ピッチ: 動画 " + f"{op.P_1080*1000:.2f} µm（校正由来）/{op.P_1080_SENSOR*1000:.2f} µm（22.3 mm/1080）、静止画 {22.3/6000*1000:.2f} µm。F5.6 のエアリー径 {2.44*0.55*5.6:.1f} µm、半開口の横（F11 相当）{2.44*0.55*11.2:.1f} µm")
print(f"灯火の像とクロマ標本: 50 m で灯火径 {Ra.F*300/5e4:.1f} px（1080 幅）= クロマ標本 {Ra.F*300/5e4/2:.1f} 個。クロマ標本が 3 個を切る距離 {Ra.F*300/6/1000:.0f} m")
