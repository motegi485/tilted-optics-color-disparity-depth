"""Q1: 合焦面の置き方ごとのぼけ。Q2: 行だけでぼけは決まるか。"""
import csv
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

import optics as op

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
ZS = [20e3, 30e3, 50e3, 70e3, 100e3]
BAND = (0.45, 5.0)
rigs = op.default_rigs()


def in_band(d, scale=1.0):
    x = abs(d) / scale
    return BAND[0] < x < BAND[1]


rows = []
for R in rigs:
    for key, o in op.OBJECTS.items():
        top, bot = op.obj_top_bot(o)
        for z in ZS:
            c = float(R.blur_signed(o["H_c"], z))
            c_top = float(R.blur_signed(top, z))
            c_bot = float(R.blur_signed(bot, z))
            r_top = float(R.row(top, z))
            r_bot = float(R.row(bot, z))
            h_px = R.F * o["h"] / z
            w_px = R.F * o["w"] / z
            lamp_px = R.F * o["lamp"] / z if np.isfinite(o["lamp"]) else np.nan
            d = op.BETA * c
            # 光線追跡で独立に出したぼけ（中心）
            rt = op.raytrace_world(R, o["H_c"], z)
            # 枠に入る条件（カメラ水平・シフトなし）
            z_fit = max(R.F * max(top - R.Hc, 0) / 360.0, R.F * max(R.Hc - bot, 0) / 360.0) / 1000.0
            rise_px = max(0.0, -r_top)  # 上へはみ出す量
            fall_px = max(0.0, r_bot - 720.0)
            shift_mm_rise = rise_px * R.p
            eq_scale = max(1.0, 0.71 * abs(c))  # MTF50 相当の画素幅
            rows.append(dict(
                rig=R.name, obj=o["label"], z_m=z / 1000, F=R.F, w_px=w_px, h_px=h_px, lamp_px=lamp_px,
                row_top=r_top, row_bot=r_bot, c=c, c_top=c_top, c_bot=c_bot, c_rt=rt["c_x"], c_y_rt=rt["c_y"],
                c_over_h=abs(c) / h_px, c_over_lamp=abs(c) / lamp_px if np.isfinite(lamp_px) else np.nan,
                d=d, d528=d / op.R_1080_TO_528, band1080=in_band(d), band528=in_band(d, op.R_1080_TO_528),
                z_fit_m=z_fit, rise_px=rise_px, rise_mm=shift_mm_rise, fall_px=fall_px,
                in_frame=(r_top >= 0 and r_bot <= 720), h_eq=h_px / eq_scale, w_eq=w_px / eq_scale,
            ))
with open(os.path.join(OUT, "q1_blur_table.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("配置一覧")
print("名前 | θ | v | F | A/J | u | Hc")
for R in rigs:
    print(f"{R.name} | {R.theta_deg:.2f} | {R.v:.2f} | {R.F:.0f} | {R.row_coef:.5f} | {R.u:.0f} | {R.Hc:.0f}")

print()
print("=== 信号灯器（H=5.0m, 1.2×0.45m, 灯火0.3m）距離ごと ===")
print("配置 | z | 像 w×h | 灯火px | ぼけc | c/h | c/灯火 | d@1080 | 帯1080 | 帯528 | 枠内(水平) | rise mm | h等価 w等価")
for r in rows:
    if r["obj"] != "信号灯器":
        continue
    print(f"{r['rig']} | {r['z_m']:.0f} | {r['w_px']:.0f}×{r['h_px']:.0f} | {r['lamp_px']:.1f} | {r['c']:+.2f} | {100*r['c_over_h']:.1f}% | {100*r['c_over_lamp']:.1f}% | {r['d']:+.2f} | {'○' if r['band1080'] else '×'} | {'○' if r['band528'] else '×'} | {'○' if r['in_frame'] else '×'} | {r['rise_mm']:.2f} | {r['h_eq']:.1f} {r['w_eq']:.1f}")

print()
print("=== 他の対象（各距離の ぼけc[px]、c/h、d@1080、帯1080/帯528） ===")
for key in ["sign60", "sign90", "head", "car_top", "car_tail"]:
    lab = op.OBJECTS[key]["label"]
    print(f"--- {lab} ---")
    print("配置 | z_fit[m] | " + " | ".join(f"{int(z/1000)}m: c, c/h, d" for z in ZS))
    for R in rigs:
        sel = [r for r in rows if r["obj"] == lab and r["rig"] == R.name]
        s = " | ".join(f"{r['c']:+.1f}, {100*r['c_over_h']:.1f}%, {r['d']:+.2f}{'○' if r['band1080'] else '×'}{'○' if r['band528'] else '×'}" for r in sel)
        print(f"{R.name} | {sel[0]['z_fit_m']:.1f} | {s}")

print()
print("=== 尾灯のぼけ／灯火径（径 0.15m 仮定） ===")
for R in rigs:
    sel = [r for r in rows if r["obj"] == "車の尾灯(0.9m)" and r["rig"] == R.name]
    print(R.name, " | ".join(f"{int(r['z_m'])}m: lamp {r['lamp_px']:.1f}px c {r['c']:+.1f} c/lamp {100*r['c_over_lamp']:.0f}%" for r in sel))

print()
print("=== ぼけ／対象の縦が距離によらないか（各配置・各対象: 20〜100 m の c/h の max/min） ===")
for R in rigs:
    s = []
    for key, o in op.OBJECTS.items():
        vals = [r["c_over_h"] for r in rows if r["rig"] == R.name and r["obj"] == o["label"]]
        lo, hi = min(vals), max(vals)
        s.append(f"{o['label']} {100*lo:.1f}〜{100*hi:.1f}%")
    print(R.name, " | ".join(s))

print()
print("=== 路面合焦: ぼけ支配と画素支配の境目 z*（0.71c=1 ⇔ c=1.41 px）と、z* より近いときの等価サイズ ===")
Ra = rigs[0]
print("対象 | c·z=定数 [px·m] | z*[m] | 等価サイズ w×h[px]（z<z*で距離によらない） | 16px の縦に落ちる距離[m]")
for key, o in op.OBJECTS.items():
    c1 = float(Ra.blur_signed(o["H_c"], 1e5)) * 100.0
    zs = c1 / 1.41
    h_eq = (Ra.F * o["h"]) / (0.71 * c1 * 1000)
    w_eq = (Ra.F * o["w"]) / (0.71 * c1 * 1000)
    z16 = Ra.F * o["h"] / 16.0 / 1000.0
    print(f"{o['label']} | {c1:.1f} | {zs:.0f} | {w_eq:.0f}×{h_eq:.0f} | {z16:.0f}")

print()
print("=== 寸法を変えたときの結論の感度（路面合焦 (a)・50 m） ===")
print("高さH | 灯器縦h | 灯火径 | c[px] | c/h | c/灯火 | z_fit[m](水平)")
for H in [4500.0, 5000.0, 5500.0, 6000.0]:
    for h, lamp in [(450.0, 300.0), (400.0, 250.0), (350.0, 250.0)]:
        c = float(Ra.blur_signed(H, 5e4))
        hp = Ra.F * h / 5e4
        lp = Ra.F * lamp / 5e4
        zf = Ra.F * (H + h / 2 - Ra.Hc) / 360 / 1000
        print(f"{H/1000:.1f} | {h:.0f} | {lamp:.0f} | {c:.2f} | {100*c/hp:.1f}% | {100*c/lp:.1f}% | {zf:.1f}")
print("カメラ高 Hc の感度（信号灯器・路面合焦・c/h = (A/Hc)(H/h)）")
for Hc in [1000.0, 1300.0, 1500.0, 2000.0]:
    R = op.rig_parallel("x", 0.0, Hc=Hc)
    c = float(R.blur_signed(5000.0, 5e4))
    print(f"Hc={Hc:.0f}: θ={R.theta_deg:.2f}°, F={R.F:.0f}, c={c:.2f}, c/h={100*c/(R.F*450/5e4):.1f}%, z_fit={R.F*(5225-Hc)/360/1000:.1f} m")

print()
print("=== カメラの向きの影響（路面合焦を保ったまま上へ φ・信号灯器） ===")
for phi in [0.0, 3.0, 5.0]:
    R = op.rig_road_pitched("p", phi)
    s = []
    for z in ZS:
        c = float(R.blur_signed(5000.0, z))
        rt = op.raytrace_world(R, 5000.0, z)
        r_t, r_b = float(R.row(5225.0, z)), float(R.row(4775.0, z))
        s.append(f"{int(z/1000)}m: c={c:.2f}(rt {rt['c_x']:.2f}) rows {r_t:.0f}..{r_b:.0f}")
    print(f"φ={phi}° F={R.F:.0f}", " | ".join(s), f" | 路面の行(20m)={float(R.row(0.0, 2e4)):.0f} 地平線={R.u:.0f}")

print()
print("=== 枠に入る条件（カメラ水平・シフトなし／各距離で必要なレンズの上げ量 mm） ===")
for R in [rigs[0], rigs[1], rigs[2], rigs[3], rigs[4]]:
    print(R.name, " | ".join(f"{op.OBJECTS[k]['label']}: z≥{[r for r in rows if r['rig']==R.name and r['obj']==op.OBJECTS[k]['label']][0]['z_fit_m']:.1f}m" for k in op.OBJECTS))
print("信号灯器の上端を枠内へ入れる上げ量 [mm]（路面合焦）:", " | ".join(f"{int(z/1000)}m {max(0, Ra.F*(5225-1300)/z-360)*Ra.p:.2f}" for z in ZS))
print("同じことをカメラの向きで行う角度 [deg]:", " | ".join(f"{int(z/1000)}m {max(0, np.degrees(np.arctan((5225-1300)/z)) - np.degrees(np.arctan(360/Ra.F))):.2f}" for z in ZS))
print("信号灯器と路面（同じ距離の路面の点）が同時に写る最小距離[m]:", f"{Ra.F*5225/720/1000:.1f}", " 上げ量", f"{(Ra.F*(5225-1300)/(Ra.F*5225/720)-360)*Ra.p:.2f} mm")

# ---------------------------------------------------------------------------
# Q2
# ---------------------------------------------------------------------------
print()
print("=== Q2: 同じ行で距離が違うとぼけは何倍変わるか（行 r 固定・z を 20〜100 m で変える） ===")
print("式: c = (A/J)(u-r) + AF/z。H<0（路面より下）になる組は物理的に存在しないので除く。")
ZS2 = [20e3, 30e3, 50e3, 70e3, 100e3]
qrows = []
for R in rigs:
    print(f"--- {R.name}: A/J={R.row_coef:.5f}, AF={R.A*R.F:.0f} px·mm, u={R.u:.1f} ---")
    for r in [0, 120, 240, 359, 480, 600, 719]:
        vals = []
        txt = []
        for z in ZS2:
            Y = (op.C0 - r) * z / R.F
            H = R.Hc + Y
            if H < 0:
                txt.append("--")
                continue
            c = float(R.blur_signed_rowz(r, z))
            # 検算
            rt = op.raytrace(R, 0.0, float(Y), float(z))
            assert abs(rt["c_x"] - abs(c)) < 0.02 * max(1, abs(c)), (R.name, r, z, c, rt["c_x"])
            vals.append(c)
            txt.append(f"{c:+.2f}(H={H/1000:.1f})")
        if vals:
            a = np.abs(vals)
            ratio = a.max() / a.min() if a.min() > 1e-6 else np.inf
            row_term = R.row_coef * (op.C0 - r) + R.const_term
            qrows.append(dict(rig=R.name, r=r, row_term=row_term, ratio=ratio, c_min=a.min(), c_max=a.max()))
            print(f"r={r:3d} 行項={row_term:+7.2f} | " + " | ".join(txt) + f" | max/min={ratio:.2f}")
with open(os.path.join(OUT, "q2_row_vs_depth.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(qrows[0].keys()))
    w.writeheader()
    w.writerows(qrows)

print()
print("=== Q2: ぼけのうち「行で決まる項」の割合（合焦面が路面と平行: 割合=(H-Hc)/(H-p)） ===")
print("配置 | 信号灯器 | 片持0.9 | 路側0.6 | 頭 | 車上端 | 尾灯")
for R in [rigs[0], rigs[1], rigs[2]]:
    s = []
    for k in ["signal", "sign90", "sign60", "head", "car_top", "car_tail"]:
        H = op.OBJECTS[k]["H_c"]
        rterm = R.row_coef * (R.u - float(R.row(H, 5e4)))
        tot = float(R.blur_signed(H, 5e4))
        s.append(f"{100*rterm/tot:.0f}%")
    print(R.name, " | ".join(s))

print()
print("=== Q2: 行を入力にした PSF 推定の誤差（同じ行で z=50 m の値を代表にしたときの 20 m・100 m での誤差） ===")
for R in rigs:
    if R.name.startswith("対照"):
        continue
    s = []
    for r in [0, 180, 359]:
        c50 = float(R.blur_signed_rowz(r, 5e4))
        c20 = float(R.blur_signed_rowz(r, 2e4))
        c100 = float(R.blur_signed_rowz(r, 1e5))
        s.append(f"r={r}: c50={c50:+.1f}, 20m {c20:+.1f}, 100m {c100:+.1f}, Δc {c20-c50:+.1f}/{c100-c50:+.1f}")
    print(R.name, " | ".join(s))
print("Δc の大きさは行によらず AF(1/z-1/50m): 20m " + f"{rigs[0].A*rigs[0].F*(1/2e4-1/5e4):+.2f} px、100m {rigs[0].A*rigs[0].F*(1/1e5-1/5e4):+.2f} px（(a)）")

print()
print("=== 枠の大きさ＋既知の実寸で AF/z を置き換えられるか: AF/z = A·w/W ===")
for z in ZS:
    w = Ra.F * 1200 / z
    print(f"z={z/1000:.0f} m: w={w:.1f} px, A·w/W={Ra.A*w/1200:.3f}, AF/z={Ra.A*Ra.F/z:.3f}")
