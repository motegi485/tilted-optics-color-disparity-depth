"""既存記録 disparity_band_fraction_b5_20261001.csv（読み取りのみ）から、
帯の下側（合焦行の近く）と上側（強いぼけ）の誤差を視差換算で比べ、全誤差への寄与を出す。
δd = MAE·K/z²（K=v/|ā|）。枝の内訳は CSV にないので、下側は 2 枝の平均 K、上側は近距離 pos・遠距離 neg とする（仮定）。
"""
import csv
import sys

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

P = r"C:\Users\t5424001\dev\tilted_optics_color_disparity_depth\project_knowledge\records\2026-10-01\disparity_band_fraction_b5_20261001.csv"
v_pos, u_pos, a_pos = 432599.32942224736, -63.927380677321885, -26.568072891235353
v_neg, u_neg, a_neg = 423627.23803714506, -21.368758126996866, -22.562290573120116
Kp, Kn = v_pos / abs(a_pos), v_neg / abs(a_neg)
Kavg = 0.5 * (Kp + Kn)
print(f"K_pos={Kp:.0f}, K_neg={Kn:.0f} px·mm@528。v_neg/v_pos={v_neg/v_pos:.4f}, |a_pos|/|a_neg|={abs(a_pos)/abs(a_neg):.4f}（幾何の v はそろい、差は傾き ā＝視差のスケールにある）")
print("枝の切片の差（y0_neg−y0_pos）: " + ", ".join(f"{z}: {(v_neg/z+u_neg)-(v_pos/z+u_pos):.1f} 行" for z in [1400, 2000, 2600, 3200, 4000]))
gap = np.mean([(v_neg / z + u_neg) - (v_pos / z + u_pos) for z in [1400, 2000, 2600, 3200]])
print(f"校正範囲の平均 {gap:.1f} 行 = 視差換算 {gap/abs(a_pos):.2f}〜{gap/abs(a_neg):.2f} px（±{gap/abs(a_pos)/2:.2f}〜{gap/abs(a_neg)/2:.2f} px 相当の平坦域）")
print()

rows = list(csv.DictReader(open(P, encoding="utf-8")))
print("z | 下側: 割合, MAE[mm], δd[px] | 帯内: 割合, MAE, δd | 上側: 割合, MAE, δd(pos換算/neg換算) | 誤差寄与（割合×MAE）下側:帯内:上側 | 帯なし MAE の再構成")
tot = []
for r in rows:
    z = float(r["gt_mm"])
    fL, fI, fH = float(r["below_band_fraction"]), float(r["in_band_fraction"]), float(r["above_band_fraction"])
    mL, mI, mH = float(r["mae_below_band_mm"]), float(r["mae_in_band_mm"]), float(r["mae_above_band_mm"])
    dL = mL * Kavg / z**2
    dI = mI * Kavg / z**2
    dHp = mH * Kp / z**2
    dHn = mH * Kn / z**2
    cL, cI, cH = fL * mL, fI * mI, fH * mH
    s = cL + cI + cH
    tot.append((z, fL, fI, fH, dL, dI, dHp, dHn, cL, cI, cH, s))
    print(f"{z:.0f} | {fL:.3f}, {mL:.1f}, {dL:.3f} | {fI:.3f}, {mI:.1f}, {dI:.3f} | {fH:.3f}, {mH:.1f}, {dHp:.3f}/{dHn:.3f} | {cL/s:.2f}:{cI/s:.2f}:{cH/s:.2f} | {s:.1f}")

tot = np.array(tot)
sel = (tot[:, 0] >= 1450) & (tot[:, 0] <= 3050)
print()
print("校正範囲内 9 距離の平均: δd 下側 {:.3f}, 帯内 {:.3f}, 上側 pos換算 {:.3f}/neg換算 {:.3f} px".format(*[tot[sel, i].mean() for i in (4, 5, 6, 7)]))
print("校正範囲内 9 距離の誤差寄与の平均比（下側:帯内:上側）= {:.2f}:{:.2f}:{:.2f}".format(*[(tot[sel, i] / tot[sel, 11]).mean() for i in (8, 9, 10)]))
print("校正範囲内 9 距離の画素割合の平均（下側:帯内:上側）= {:.3f}:{:.3f}:{:.3f}".format(*[tot[sel, i].mean() for i in (1, 2, 3)]))
print("z≥2650 の平均比（下側:帯内:上側）= {:.2f}:{:.2f}:{:.2f}".format(*[(tot[tot[:,0]>=2650, i] / tot[tot[:,0]>=2650, 11]).mean() for i in (8, 9, 10)]))
print("z≤2050 の平均比（下側:帯内:上側）= {:.2f}:{:.2f}:{:.2f}".format(*[(tot[tot[:,0]<=2050, i] / tot[tot[:,0]<=2050, 11]).mean() for i in (8, 9, 10)]))

print()
print("=== 2026-09-18 §6.3 の推論帯の感度（T=20・held-out 9 距離）: 下限と上限のどちらが MAE を動かすか ===")
print("上限を 14.5→5（下限 0.45 固定）: 79.5→50.2 mm（−29.3 mm、有効率 0.406→0.313）")
print("下限を 0.2→0.45（上限 5 固定）: 53.7→50.2 mm（−3.5 mm、有効率 0.346→0.313）")
print("下限を 0.45→0.8（上限 5 固定）: 50.2→50.3 mm（+0.1 mm、有効率 0.313→0.280）")
print("→ 同じ 3 割台の有効率でも、MAE を動かすのは上限（強いぼけ側）が 8 倍以上大きい（ここは記録の値の引き算）")

print()
print("=== 報告書の帯の表を live 係数で再計算（2026-09-15 固定値との差） ===")
fix = dict(vp=426165.15, up=-61.04, ap=-26.43, vn=411311.86, un=-17.39, an=-22.76)
liv = dict(vp=v_pos, up=u_pos, ap=a_pos, vn=v_neg, un=u_neg, an=a_neg)


def rows_in_band(par, z):
    # y = u + v/z + a d; pos: d>0, neg: d<0
    yp = [par["up"] + par["vp"] / z + par["ap"] * d for d in (0.45, 5.0)]
    yn = [par["un"] + par["vn"] / z + par["an"] * d for d in (-0.45, -5.0)]
    def clip(a, b):
        lo, hi = sorted((a, b))
        lo2, hi2 = max(lo, 0), min(hi, 351)
        return (round(lo2), round(hi2)) if lo2 <= hi2 else None
    return clip(*yp), clip(*yn)


print("z[m] | 固定値 pos, neg | live pos, neg | 1 m 差（pos/neg）固定値→live")
for z in [1, 2, 3, 4, 5, 7, 9, 19]:
    zmm = z * 1000.0
    f = rows_in_band(fix, zmm)
    l = rows_in_band(liv, zmm)
    df = (fix["vp"] / abs(fix["ap"]) * (1 / zmm - 1 / (zmm + 1000)), fix["vn"] / abs(fix["an"]) * (1 / zmm - 1 / (zmm + 1000)))
    dl = (Kp * (1 / zmm - 1 / (zmm + 1000)), Kn * (1 / zmm - 1 / (zmm + 1000)))
    print(f"{z} | {f[0]}, {f[1]} | {l[0]}, {l[1]} | {df[0]:.3f}/{df[1]:.3f} → {dl[0]:.3f}/{dl[1]:.3f}")

print()
print("=== 1 m 差が δd を下回る距離（視差ノイズとの照合。K は px·m@528） ===")
for name, K in [("pos", Kp / 1000), ("neg", Kn / 1000)]:
    for dd in [0.021, 0.1, 0.15, 0.45]:
        # K(1/z-1/(z+1))=dd → z≈(-1+sqrt(1+4K/dd))/2
        z = (-1 + np.sqrt(1 + 4 * K / dd)) / 2
        print(f"{name} K={K:.2f}: δd={dd} px → 1 m 差と等しくなるのは z={z:.1f} m")
