"""開口形状 × エッジの向き × ぼけ量で、視差（ぼけ量 c）の推定誤差を比べる（physB・2026-11-09）

目的: 机上で形状を絞るための最初の結果。向きを 0〜165°（15° 刻み）で振ったバーコード（1 次元のランダムな縞）
      または 1 本の段差エッジを、各開口で合成して推定する。
使い方: uv run --frozen python -u run_orient.py barcode|step
入力: なし
出力: physB_out/orient_<scene>.csv, physB_out/orient_<scene>.png, physB_out/orient_<scene>.txt（UTF-8）
推定法（結果の method 列）:
  i_hzncc : (i) 現行の R–B 横方向 ZNCC（開発リポの Rust 実装・K25 S41 T20）。R–B の基線が横の形状だけ。ĉ=2d/(p_R−p_B)_x
  ii_zsum : (ii) スカラー c で各チャネルを (c/2)p_k だけ 2 次元に戻し、|∇I| のチャネル対 ZNCC（3 対）の平均を最大化
  ii_bando: (ii) 同じ戻し方で 1 − det(相関行列)（Bando 2008 の color-line 指標と同じ値）を最大化
  iii_zw  : (iii) 窓内の構造テンソルで対ごとに重み w_ij=b_ijᵀT b_ij/trT を付けた ZNCC の和を最大化
  (ii)(iii) は候補 c=−47〜47（1 px 刻み・(i) の S=41 が現行形状で覆う |c|≤47 に合わせた）＋放物線。窓 K=25、選択は |∇I|>20。
校正の模擬（cal 列）: 等方 1/f 雑音で推定法ごとの比 k=median(ĉ/c) を求め、ĉ/k で評価する（apsim.iso_gain）。
      向きによらない一定の比は実機の校正で吸収される、とみなした値。raw は割らない値。
仮定: apsim.py 冒頭のとおり（幾何光学・理想フィルタ・無彩色・雑音 σ=1.5 階調・ぼけは画面内で一定）。
評価: 全開口で描いた画像の |∇I|>20 の画素（中央 112×112 の ROI）を分母に、有効率・良好率（|ĉ−c|≤1 px）・偏り（中央値）・
      ばらつき（1.4826·MAD・有効画素）。
"""
import csv
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import apsim as ap

OUT = os.path.join(os.path.dirname(HERE), "physB_out")
SCENE = sys.argv[1] if len(sys.argv) > 1 else "barcode"
H = W = 192
SS = 4
ROI = (slice(40, 152), slice(40, 152))
PHIS = list(range(0, 180, 15))
CS = [3.0, 6.0, 12.0]
SIGMA = 1.5
SHAPES = ["cy_lr", "cy_rot45", "rgb3", "cmy3", "ycg3", "kaneko", "cy_msec", "cy_topW"]
CANDS = np.arange(-47, 48, 1.0)
METHODS = ["i_hzncc", "ii_zsum", "ii_bando", "iii_zw"]

t0 = time.time()
scenes = {}
for phi in PHIS:
    rng = np.random.default_rng(5000 + phi)
    img = ap.scene_barcode(H, W, SS, phi, rng) if SCENE == "barcode" else ap.scene_step(H, W, SS, phi)
    scenes[phi] = ap.Scene(img, SS)
masks = {(phi, c): ap.eval_mask_from_full(scenes[phi], c, roi=ROI)[0] for phi in PHIS for c in CS}

rows = []
gains = {}
for shape in SHAPES:
    fn = ap.SHAPES[shape][0]
    p = ap.centroid_img(ap.shape_stats(fn))
    use_i = shape in ap.LR_SHAPES
    for c in CS:
        k = ap.iso_gain(fn, p, c, CANDS, use_i)
        gains[(shape, c)] = k
        kf = ap.kernel_ffts(fn, c, scenes[0].shape, SS)
        for phi in PHIS:
            lin = ap.render(scenes[phi], fn, c, kfft=kf)
            u8 = ap.to_u8(lin, SIGMA, np.random.default_rng(int(c * 100 + phi)))
            m = masks[(phi, c)]
            for mt, e in ap.run_methods(u8, p, CANDS, use_i).items():
                rows.append(dict(scene=SCENE, phi=phi, shape=shape, c=c, method=mt, cal="raw", k=1.0, **ap.stats(e, c, m)))
                rows.append(dict(scene=SCENE, phi=phi, shape=shape, c=c, method=mt, cal="cal", k=k[mt], **ap.stats(e / k[mt], c, m)))
    print(f"{shape} 済 ({time.time() - t0:.0f} s)")

with open(os.path.join(OUT, f"orient_{SCENE}.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)

# 要約 --------------------------------------------------------------------
LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


def sel(shape, c, mt, cal):
    return [r for r in rows if r["shape"] == shape and r["c"] == c and r["method"] == mt and r["cal"] == cal]


log(f"=== 校正の模擬で求めた比 k = median(ĉ/c)（等方 1/f・σ={SIGMA}）===")
for shape in SHAPES:
    log(f"{ap.SHAPES[shape][1]}: " + " | ".join(f"c={c:g}: " + ", ".join(f"{mt}={v:.3f}" for mt, v in gains[(shape, c)].items()) for c in CS))

for cal in ["cal", "raw"]:
    log()
    log(f"=== 向きの掃引（{SCENE}・σ={SIGMA}・{'校正の比で割った値' if cal == 'cal' else '割らない値'}）: 形状 × 推定法ごとの要約（12 方向）===")
    log("各セル: 良好率の最小（最悪の向き φ）/ 良好率の平均 / 偏りの幅（向きごとの中央値の max−min）/ ばらつきの最大 [px of c]")
    for c in CS:
        log(f"--- c = {c:g} px")
        log("形状 | " + " | ".join(METHODS))
        for shape in SHAPES:
            cells = []
            for mt in METHODS:
                rr = sel(shape, c, mt, cal)
                if not rr:
                    cells.append("—")
                    continue
                g = np.array([r["good"] for r in rr])
                b = np.array([r["bias"] for r in rr], dtype=float)
                s = np.array([r["sd"] if np.isfinite(r["sd"]) else np.inf for r in rr])
                iw = int(np.argmin(g))
                brange = (np.nanmax(b) - np.nanmin(b)) if np.isfinite(b).any() else np.nan
                cells.append(f"{g.min():.2f}(φ={rr[iw]['phi']}) / {g.mean():.2f} / {brange:.2f} / {s.max():.2f}")
            log(f"{ap.SHAPES[shape][1]} | " + " | ".join(cells))

log()
log("向きごとの良好率（校正の比で割った値）")
for mt in METHODS:
    for c in CS:
        log(f"--- {mt} c={c:g}: φ = " + " ".join(f"{p_:>4d}" for p_ in PHIS))
        for shape in SHAPES:
            rr = sel(shape, c, mt, "cal")
            if rr:
                log(f"  {ap.SHAPES[shape][1]:<14s}: " + " ".join(f"{r['good']:.2f}" for r in rr))
log()
log("向きごとの偏り（中央値）とばらつき（校正の比で割った値・c=6）")
for mt in ["iii_zw", "ii_bando", "i_hzncc"]:
    log(f"--- {mt}")
    for shape in SHAPES:
        rr = sel(shape, 6.0, mt, "cal")
        if not rr:
            continue
        log(f"  {ap.SHAPES[shape][1]:<14s}: 偏り " + " ".join(f"{r['bias']:+.2f}" for r in rr))
        log(f"  {'':<14s}  ばらつき " + " ".join(f"{r['sd']:.2f}" for r in rr))
log(f"所要 {time.time() - t0:.0f} s")
with open(os.path.join(OUT, f"orient_{SCENE}.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(LOG) + "\n")

# 図 ----------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
colors = dict(cy_lr="#222222", cy_rot45="#8c564b", rgb3="#d62728", cmy3="#17becf", ycg3="#2ca02c",
              kaneko="#7f7f7f", cy_msec="#9467bd", cy_topW="#ff7f0e")
fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
for j, c in enumerate(CS):
    for shape in SHAPES:
        for mt, ls in [("ii_bando", "-"), ("i_hzncc", ":")]:
            rr = sel(shape, c, mt, "cal")
            if not rr:
                continue
            lab = f"{ap.SHAPES[shape][1]}" + (" (i)" if mt == "i_hzncc" else "")
            axs[0, j].plot(PHIS, [r["good"] for r in rr], ls, color=colors[shape], label=lab, lw=1.3)
            axs[1, j].plot(PHIS, [r["sd"] for r in rr], ls, color=colors[shape], lw=1.3)
    axs[0, j].set_title(f"c = {c:g} px（実線 (ii) color-line・点線 (i) 現行）", fontsize=9)
    axs[1, j].set_yscale("log")
    axs[1, j].set_xlabel("エッジ法線の角度 φ [°]（0=縦エッジ・90=横エッジ）")
    for a in axs[:, j]:
        a.grid(alpha=0.3)
        a.set_xticks(range(0, 181, 30))
axs[0, 0].set_ylabel("良好率（|ĉ/k−c|≤1 px）")
axs[1, 0].set_ylabel("ばらつき 1.4826·MAD [px of c]")
axs[0, 0].legend(fontsize=6.5, ncol=2)
fig.suptitle(f"[計算] {SCENE}: 開口形状ごとの良好率とばらつき（校正の比で割った値・σ=1.5・幾何光学・理想フィルタ）", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"orient_{SCENE}.png"), dpi=120)
