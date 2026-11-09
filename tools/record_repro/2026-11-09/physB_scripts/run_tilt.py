"""アオリとの関係: 行ごとにぼけ量が変わる画像で、縦の基線の推定が横の基線よりどれだけ悪くなるか（physB・2026-11-09）

目的: アオリではぼけ c が行ごとに変わる（c の勾配 g = sinθ/N [px/行]・研究方針_検出と測距.md §3.1, §14）。
      窓の中で視差が行ごとに変わると、横の基線ではせん断、縦の基線では行方向の伸縮になる。
      さらに PSF の縦が cosθ に縮む（run_geometry.py の光線追跡で c_y/c_x≈cosθ を確認）。
      現行 C/Y 左右（横の基線）と、それを 90° 回した上下分割（縦の基線）を同じ推定法で比べる。
入力: なし
出力: physB_out/tilt.csv, physB_out/tilt.txt（UTF-8）
条件: なし（g=0, sy=1）／θ=3.97°（路面合焦・g=sin3.97°/5.6=0.0124, sy=cos=0.9976）／θ=30°（現行・g=0.0893, sy=0.866）
      ／θ=30° の縦の縮みだけ（g=0, sy=0.866）／θ=30° の行の変化だけ（g=0.0893, sy=1）。
      平面の被写体（奥行き一定）を仮定し、c(y) = c0 + g·(y − 96)（c0 = 6, 10 px・192 行の画像・評価は 40〜151 行）。
      528×352 の処理画像と同じ画素の尺度とみなす（g は解像度によらず px/行で同じ）。
推定法: (i) 現行の横方向 ZNCC（Rust）／(i)ᵀ 同じ関数を転置画像に当てた縦方向版（上下分割用）／
      (ii) 1 対（R–B）の 2 次元推定（|∇I| ZNCC・推定側は sy を知っている）。真値は各画素の行の c(y)。
仮定: apsim.py 冒頭のとおり。行ごとの描画は c を 0.25 px 刻みで描いて行ごとに線形補間（PSF の広がりの中の c の変化は無視）。
      c_y/c_x の行による ±2 % の変化（光線追跡）は入れない。
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
H = W = 192
SS = 4
ROI = (slice(40, 152), slice(40, 152))
N_ = 5.6
CONDS = [
    ("なし", 0.0, 1.0),
    ("θ=3.97°", np.sin(np.deg2rad(3.97)) / N_, np.cos(np.deg2rad(3.97))),
    ("θ=30°", np.sin(np.deg2rad(30.0)) / N_, np.cos(np.deg2rad(30.0))),
    ("θ=30° 縮みのみ", 0.0, np.cos(np.deg2rad(30.0))),
    ("θ=30° 勾配のみ", np.sin(np.deg2rad(30.0)) / N_, 1.0),
]
C0S = [6.0, 10.0]
SCENES = [("bar0", 0.0), ("bar90", 90.0), ("bar45", 45.0), ("iso", None)]
SEEDS = [1, 2]
CANDS = np.arange(-47, 48, 1.0)

t0 = time.time()
rows = []
yy = np.arange(H, dtype=float)
for sname, phi in SCENES:
    for seed in SEEDS:
        rng = np.random.default_rng(900 + seed * 10 + (0 if phi is None else int(phi)))
        img = ap.scene_1overf(H, W, SS, rng) if phi is None else ap.scene_barcode(H, W, SS, phi, rng)
        sc = ap.Scene(img, SS)
        for c0 in C0S:
            m, _, _ = ap.eval_mask_from_full(sc, c0, roi=ROI)
            for cname, g, sy in CONDS:
                c_rows = c0 + g * (yy - 96.0)
                truth = np.repeat(c_rows[:, None], W, axis=1)
                for shape in ["cy_lr", "cy_tb"]:
                    fn = ap.SHAPES[shape][0]
                    p = ap.centroid_img(ap.shape_stats(fn), sy)
                    if g == 0:
                        lin = ap.render(sc, fn, c0, sy)
                    else:
                        lin = ap.render_rowvarying(sc, fn, c_rows, sy, step=0.25)
                    u8 = ap.to_u8(lin, 1.5, np.random.default_rng(seed * 31 + int(c0)))
                    if shape == "cy_lr":
                        d = ap.est_i_rust(u8)
                        chat = 2.0 * d / (p["R"][0] - p["B"][0])
                        rows.append(dict(scene=sname, seed=seed, c0=c0, cond=cname, shape=shape, method="i", **ap.stats(chat, truth, m)))
                    else:
                        uT = {k: np.ascontiguousarray(u8[k].T) for k in ap.CH}
                        dT = ap.est_i_rust(uT).T
                        chat = 2.0 * dT / (p["R"][1] - p["B"][1])
                        rows.append(dict(scene=sname, seed=seed, c0=c0, cond=cname, shape=shape, method="i", **ap.stats(chat, truth, m)))
                    res, _ = ap.est_scalar2d(u8, p, CANDS, pairs=(("R", "B"),), use_bando=False, metrics=("zsum",))
                    rows.append(dict(scene=sname, seed=seed, c0=c0, cond=cname, shape=shape, method="ii_RB", **ap.stats(res["zsum"][0], truth, m)))
        print(f"{sname} seed{seed} 済 ({time.time() - t0:.0f} s)")

with open(os.path.join(OUT, "tilt.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)

LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


def agg(scene, c0, cond, shape, method):
    rr = [r for r in rows if r["scene"] == scene and r["c0"] == c0 and r["cond"] == cond and r["shape"] == shape and r["method"] == method]
    n = sum(r["n"] for r in rr)
    return dict(good=sum(r["good"] * r["n"] for r in rr) / n, valid=sum(r["valid"] * r["n"] for r in rr) / n,
                bias=np.nanmean([r["bias"] for r in rr]), sd=np.nanmean([r["sd"] for r in rr]), mae=np.nanmean([r["mae"] for r in rr]))


log("=== アオリ（行ごとのぼけの変化＋縦の縮み）: 横の基線（現行 C/Y 左右）vs 縦の基線（上下分割）===")
log("値: 良好率 / 偏り / ばらつき(1.4826·MAD) / MAE  [px of c]。σ=1.5。真値は各画素の行の c(y)")
log("横の基線は縦エッジ（bar0）、縦の基線は横エッジ（bar90）が得意な向き。bar45 と iso は両方に共通")
for method in ["i", "ii_RB"]:
    for c0 in C0S:
        log(f"--- 推定法 {method}（i: 現行の 1 次元 ZNCC・上下分割は転置して同じ関数）, c0={c0:g}")
        log("条件 | 横の基線 bar0 | 縦の基線 bar90 | 横 bar45 | 縦 bar45 | 横 iso | 縦 iso")
        for cname, g, sy in CONDS:
            cells = []
            for sname, shape in [("bar0", "cy_lr"), ("bar90", "cy_tb"), ("bar45", "cy_lr"), ("bar45", "cy_tb"), ("iso", "cy_lr"), ("iso", "cy_tb")]:
                a = agg(sname, c0, cname, shape, method)
                cells.append(f"{a['good']:.2f} / {a['bias']:+.2f} / {a['sd']:.2f} / {a['mae']:.2f}")
            log(f"{cname}（g={g:.4f}, sy={sy:.3f}） | " + " | ".join(cells))
log()
log("縦の基線の悪化の倍率（ばらつき・MAE の 縦/横 の比。bar0 の横 と bar90 の縦、iso どうし）")
for method in ["i", "ii_RB"]:
    for c0 in C0S:
        for cname, g, sy in CONDS:
            a1 = agg("bar0", c0, cname, "cy_lr", method)
            a2 = agg("bar90", c0, cname, "cy_tb", method)
            b1 = agg("iso", c0, cname, "cy_lr", method)
            b2 = agg("iso", c0, cname, "cy_tb", method)
            log(f"  {method} c0={c0:g} {cname}: 縞 sd比 {a2['sd'] / a1['sd']:.2f}, MAE比 {a2['mae'] / a1['mae']:.2f} | iso sd比 {b2['sd'] / b1['sd']:.2f}, MAE比 {b2['mae'] / b1['mae']:.2f}")
log(f"所要 {time.time() - t0:.0f} s")
with open(os.path.join(OUT, "tilt.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(LOG) + "\n")
