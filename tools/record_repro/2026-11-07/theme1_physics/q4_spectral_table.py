"""Q4: 狭帯域の灯火が、どの半開口とどのチャネルを通るか。視差が出るか。"""
import csv
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

import spectral as sp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
src = sp.sources()
rows = []
print("値: 各チャネルの (Y半開口の寄与, C半開口の寄与)。単位は白(平坦な光=1) で合計 1。e = 実効重心（+1 全部Y側, -1 全部C側, 0 全開口）。")
print("基線の相対値 b_RB=(e_R-e_B)/2 ... 白で 1.0 が理想の R–B（0.85·r）。b_RG=(e_R-e_G)/2, b_GB=(e_G-e_B)/2。")
for cname, cfg in sp.FILTER_CASES.items():
    ty, tc = sp.filter_pair(**cfg)
    g = sp.wb_gains(ty, tc)
    print()
    print(f"=== フィルタ {cname} ===")
    print("光源 | R(Y,C) | G(Y,C) | B(Y,C) | e_R e_G e_B | b_RB b_RG b_GB | 最小チャネル／最大チャネル(信号比)")
    for sname, E in src.items():
        # 発光体は最大チャネルを 1 に正規化して比較する
        W = sp.weights(E, ty, tc, g)
        tot = {k: W[k][0] + W[k][1] for k in sp.CH}
        if sname.startswith("LED"):
            m = max(tot.values())
            W = {k: (W[k][0] / m, W[k][1] / m) for k in sp.CH}
            tot = {k: W[k][0] + W[k][1] for k in sp.CH}
        e = {k: sp.centroid_e(*W[k]) for k in sp.CH}
        def b(a, c):
            return (e[a] - e[c]) / 2
        rows.append(dict(filter=cname, source=sname, RY=W["R"][0], RC=W["R"][1], GY=W["G"][0], GC=W["G"][1], BY=W["B"][0], BC=W["B"][1],
                         e_R=e["R"], e_G=e["G"], e_B=e["B"], b_RB=b("R", "B"), b_RG=b("R", "G"), b_GB=b("G", "B")))
        print(f"{sname} | {W['R'][0]:.3f},{W['R'][1]:.3f} | {W['G'][0]:.3f},{W['G'][1]:.3f} | {W['B'][0]:.3f},{W['B'][1]:.3f} | "
              f"{e['R']:+.2f} {e['G']:+.2f} {e['B']:+.2f} | {b('R','B'):+.3f} {b('R','G'):+.3f} {b('G','B'):+.3f} | {min(tot.values())/max(tot.values()):.4f}")
with open(os.path.join(OUT, "q4_spectral_weights.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# 灯火ピークを振ったときの実効基線（フィルタの端にかかると視差が出るか）
print()
print("=== 灯火のピーク波長・半値幅を振ったときの、チャネル間の実効基線 b_RG, b_GB（基準フィルタ） ===")
ty, tc = sp.filter_pair(**sp.FILTER_CASES["基準(Y500/C580)"])
g = sp.wb_gains(ty, tc)
print("λ_peak | FWHM | 信号の強い 2 チャネル | 比（弱/強）| 実効重心の差 Δe/2 （理想 R–B=1.0）| 片方でも 5 % 未満で照合不能なら ×")
for mu in [500, 505, 510, 520, 575, 580, 585, 590, 595, 630]:
    for fw in [15, 30]:
        E = sp.gauss(float(mu), float(fw))
        W = sp.weights(E, ty, tc, g)
        tot = {k: W[k][0] + W[k][1] for k in sp.CH}
        order = sorted(sp.CH, key=lambda k: -tot[k])
        a, b_ = order[0], order[1]
        ea, eb = sp.centroid_e(*W[a]), sp.centroid_e(*W[b_])
        ratio = tot[b_] / tot[a]
        print(f"{mu} | {fw} | {a},{b_} | {ratio:.3f} | {abs(ea-eb)/2:.3f} | {'○' if ratio>=0.05 else '×'}")

# 塗料・印刷の縁のコントラスト（チャネルごとの 8bit 差）
def enc(x):
    x = np.clip(x, 0, 1)
    return 255 * np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)

print()
print("=== 縁の明暗差（ガンマ後 8bit・照明は平坦な白・露出は白=0.8）。使えるチャネル対の判定（差 ≥ 20 を仮に「写る」） ===")
pairs = [("白", "赤塗料", "白地の赤枠"), ("白", "青塗料", "白地の青文字/青地の白文字"), ("白", "緑塗料", "緑地の白文字"), ("白", "黄塗料", "黄地"), ("白", "黒", "白地の黒文字"),
         ("黄塗料", "黒", "黄地の黒記号"), ("赤塗料", "黒", "赤と黒"), ("青塗料", "黒", "青と黒")]
W_all = {k: sp.weights(v, ty, tc, g) for k, v in src.items()}
sky = 0.75 * np.ones_like(sp.LAM)
W_sky = sp.weights(sky, ty, tc, g)
for a, b_, lab in pairs:
    da = {}
    for k in sp.CH:
        va = 0.8 * sum(W_all[a][k]) / 0.85
        vb = 0.8 * sum(W_all[b_][k]) / 0.85
        da[k] = abs(enc(va) - enc(vb))
    usable = [p for p in ("RB", "RG", "GB") if da[p[0]] >= 20 and da[p[1]] >= 20]
    print(f"{lab}: ΔR={da['R']:.0f} ΔG={da['G']:.0f} ΔB={da['B']:.0f} | 使える対: {usable}")
print("空(平坦) vs 黒い灯器の外形:", {k: round(float(abs(enc(0.75*sum(W_sky[k])) - enc(0.8*sum(W_all['黒'][k])/0.85)))) for k in sp.CH})

# 印刷の灯火と LED の違い
print()
print("=== 印刷の灯火が狭帯域の問題を隠す: ガンマ後 8bit の縁（黒い灯器の中の灯火。照明は平坦な白・白=0.8 の露出） ===")
for name in ["印刷赤", "印刷黄", "印刷青緑"]:
    wa = W_all[name]
    wb_ = W_all["黒"]
    d = {k: abs(enc(0.8 * sum(wa[k]) / 0.85) - enc(0.8 * sum(wb_[k]) / 0.85)) for k in sp.CH}
    e = {k: sp.centroid_e(*wa[k]) for k in sp.CH}
    print(f"{name}: ΔR={d['R']:.0f} ΔG={d['G']:.0f} ΔB={d['B']:.0f} | e_R={e['R']:+.2f} e_G={e['G']:+.2f} e_B={e['B']:+.2f}")
for name, tgt in [("LED赤630", 0.9), ("LED黄590", 0.9), ("LED青505", 0.9)]:
    E = src[name]
    W = sp.weights(E, ty, tc, g)
    m = max(sum(W[k]) for k in sp.CH)
    wb_ = W_all["黒"]
    vals = {k: 0.8 * sum(wb_[k]) / 0.85 + tgt * sum(W[k]) / m for k in sp.CH}
    d = {k: abs(enc(vals[k]) - enc(0.8 * sum(wb_[k]) / 0.85)) for k in sp.CH}
    e = {k: sp.centroid_e(*W[k]) for k in sp.CH}
    print(f"{name}: ΔR={d['R']:.0f} ΔG={d['G']:.0f} ΔB={d['B']:.0f} | e_R={e['R']:+.2f} e_G={e['G']:+.2f} e_B={e['B']:+.2f}")
