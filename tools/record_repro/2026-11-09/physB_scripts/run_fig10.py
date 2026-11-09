"""Kaneko2025 Fig.10 の傾向（縦エッジだけの画像を回すと視差の誤差が増え、60° 超で測れない）が合成で再現するか（physB・2026-11-09）

目的: 合成器の妥当性の目安。現行の推定（(i): R–B の横方向 ZNCC・開発リポの Rust 実装そのもの・K25 S41 T20）を、
      左右分割の開口（現行 C/Y と Kaneko2025 型）で、1 次元のランダムなバーコードを 0〜90° 回した画像に当てる。
入力: なし
出力: physB_out/fig10.csv, physB_out/fig10.png, physB_out/fig10.txt（UTF-8）
仮定: apsim.py 冒頭のとおり（幾何光学・理想フィルタ・無彩色・雑音 σ=1.5 階調）。ぼけは画面内で一定（アオリの行ごとの変化なし）。
      φ はエッジの法線の角度（0°=縦エッジ、90°=横エッジ）。Kaneko2025 の「回転角」と同じ意味と読む [推論]。
評価: 全開口で描いた画像の |∇I|>20 の画素（ROI 内）を分母に、有効率（ピークが立った割合）、
      良好率（|ĉ−c|≤1 px）、偏り（中央値）、ばらつき（1.4826·MAD）、MAE（有効画素）。ĉ = 2d/(p_R−p_B)_x。
"""
import csv
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import apsim as ap

OUT = os.path.join(os.path.dirname(HERE), "physB_out")
H = W = 192
SS = 4
ROI = (slice(40, 152), slice(40, 152))
PHIS = [0, 10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90]
CS = [4.0, 8.0, 12.0]
SEEDS = [11, 12, 13]
SIGMA = 1.5

rows = []
LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


for shape in ["cy_lr", "kaneko"]:
    fn = ap.SHAPES[shape][0]
    p = ap.centroid_img(ap.shape_stats(fn))
    bx = p["R"][0] - p["B"][0]
    for c in CS:
        kf = None
        for phi in PHIS:
            agg = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed * 1000 + phi)
                sc = ap.Scene(ap.scene_barcode(H, W, SS, phi, rng), SS)
                if kf is None:
                    kf = ap.kernel_ffts(fn, c, sc.shape, SS)
                lin = ap.render(sc, fn, c, kfft=kf)
                u8 = ap.to_u8(lin, SIGMA, np.random.default_rng(seed + 7))
                d = ap.est_i_rust(u8)
                chat = 2.0 * d / bx
                m, _, _ = ap.eval_mask_from_full(sc, c, roi=ROI)
                e = chat[m] - c
                agg.append((m.sum(), e))
            n = sum(a for a, _ in agg)
            e = np.concatenate([b for _, b in agg])
            ok = np.isfinite(e)
            ev = e[ok]
            med = np.median(ev) if ev.size else np.nan
            sd = 1.4826 * np.median(np.abs(ev - med)) if ev.size else np.nan
            r = dict(shape=shape, c=c, phi=phi, n=int(n), valid=ok.mean(), good=(np.abs(ev) <= 1.0).sum() / n,
                     bias=med, sd=sd, mae=np.mean(np.abs(ev)) if ev.size else np.nan)
            rows.append(r)

log("=== Kaneko2025 Fig.10 の再現の試み（(i) 現行の R–B 横 ZNCC・バーコードを回す・σ=1.5）===")
log("φ: 法線の角度（0=縦エッジ）。値: 有効率 / 良好率(|err|≤1px) / 偏り / ばらつき(MAD) / MAE  [ĉ の px]")
for shape in ["cy_lr", "kaneko"]:
    for c in CS:
        log(f"--- {ap.SHAPES[shape][1]}  c={c} px（R–B 視差 {ap.BETA * c:.2f} px 相当・現行形状）")
        base = None
        for r in rows:
            if r["shape"] != shape or r["c"] != c:
                continue
            if base is None:
                base = r["mae"]
            log(f"  φ={r['phi']:2d}: {r['valid']:.3f} / {r['good']:.3f} / {r['bias']:+.3f} / {r['sd']:.3f} / {r['mae']:.3f}"
                f"  MAE比(0°比) {r['mae'] / base:.2f}  1/cosφ={1 / max(np.cos(np.deg2rad(r['phi'])), 1e-9):.2f}")

with open(os.path.join(OUT, "fig10.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(OUT, "fig10.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(LOG) + "\n")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
cols = {4.0: "#1f77b4", 8.0: "#ff7f0e", 12.0: "#2ca02c"}
for shape, ls in [("cy_lr", "-"), ("kaneko", "--")]:
    for c in CS:
        rr = [r for r in rows if r["shape"] == shape and r["c"] == c]
        ph = [r["phi"] for r in rr]
        axs[0].plot(ph, [r["valid"] for r in rr], ls, color=cols[c], label=f"{ap.SHAPES[shape][1]} c={c:g}")
        axs[1].plot(ph, [r["good"] for r in rr], ls, color=cols[c])
        axs[2].plot(ph, [r["mae"] for r in rr], ls, color=cols[c])
axs[0].set_ylabel("有効率")
axs[1].set_ylabel("良好率（|ĉ−c|≤1 px）")
axs[2].set_ylabel("MAE（有効画素）[px of c]")
axs[2].set_yscale("log")
for a in axs:
    a.set_xlabel("エッジ法線の角度 φ [°]（0=縦エッジ）")
    a.axvline(60, color="gray", lw=0.6, ls=":")
    a.grid(alpha=0.3)
axs[0].legend(fontsize=7)
fig.suptitle("[計算] 左右分割開口＋現行の R–B 横 ZNCC（Rust 実装・K25 S41 T20）でバーコードを回したとき", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig10.png"), dpi=130)
