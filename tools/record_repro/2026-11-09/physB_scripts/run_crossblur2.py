"""PSF 合わせ (iv) の頑健さ: 推定側の PSF モデルが実物とずれたとき・色のある被写体のとき（physB・2026-11-09）

目的: run_crossblur.py の (iv) は、合成と推定に同じ PSF モデルを使っている（いわゆる inverse crime）ので楽観的。
      推定側は理想の形状のまま、合成側だけを次のようにずらしたときに (iv) と (ii) color-line がどう崩れるかを見る。
        none   : ずれなし
        rot5   : フィルタが開口面内で 5° 回って取り付けられている
        blur07 : 全チャネルに σ=0.7 px のガウスぼけ（収差・回折の代わり）
        leak10 : R と B のチャネルが、自分のフィルタ以外の領域からも 10 % 光を受ける（フィルタの漏れ）
        g73    : 現行形状だけ: G が Y 側 0.7・C 側 0.3 の重み（synth.py の measured_like_filter の白色光の重みに合わせた [仮定]）
        aspect : PSF の縦横比が 0.95（推定側は 1 と仮定）
      あわせて色のある被写体（R・B に独立な ±25 % の色むら）で (iv) を見る。
入力: なし
出力: physB_out/crossblur2.txt（UTF-8）
評価: 校正の模擬は「ずれた合成」の等方 1/f（別の種）で比 k を求め、ĉ/k で評価（実機の校正はずれた実物で行うため）。
      被写体は向き 0/45/90/135° のバーコードと等方 1/f（評価用の種）。c = 6, 12。σ=1.5。
仮定: apsim.py 冒頭のとおり。
"""
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cv2
import numpy as np
import scipy.fft as sfft

import apsim as ap

OUT = os.path.join(os.path.dirname(HERE), "physB_out")
H = W = 192
SS = 4
ROI = (slice(40, 152), slice(40, 152))
CANDS = np.arange(-30, 31, 1.0)
SHAPES = ["cy_lr", "rgb3", "cmy3", "ycg3", "cy_msec"]
PERTS = ["none", "rot5", "blur07", "leak10", "g73", "aspect"]
SCN = [("iso", None), ("bar0", 0.0), ("bar45", 45.0), ("bar90", 90.0), ("bar135", 135.0)]
LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


def perturbed(fn, pert, shape):
    if pert == "rot5":
        return ap.rotated(fn, 5.0)
    if pert == "leak10":
        def g(U, V):
            T = fn(U, V)
            return dict(R=T["R"] + 0.1 * (1 - T["R"]), G=T["G"], B=T["B"] + 0.1 * (1 - T["B"]))
        return g
    if pert == "g73" and shape == "cy_lr":
        def g(U, V):
            T = fn(U, V)
            return dict(R=T["R"], G=0.7 * (U < 0) + 0.3 * (U >= 0), B=T["B"])
        return g
    return fn


def render_p(sc, fn, c, pert, shape, scenes_color=None):
    fr = perturbed(fn, pert, shape)
    sy = 0.95 if pert == "aspect" else 1.0
    ks = ap.psf_kernels(fr, c, SS, sy)
    out = {}
    for k in ap.CH:
        s = sc if scenes_color is None else scenes_color[k]
        img = sfft.irfft2(s.F * ap.kernel_fft(ks[k], s.shape), s=s.shape, workers=-1)
        out[k] = ap.box_down(img, SS)
    if pert == "blur07":
        out = {k: cv2.GaussianBlur(v, (0, 0), 0.7) for k, v in out.items()}
    return out


def estimate(u8, fn, p):
    res, _ = ap.est_scalar2d(u8, p, CANDS, metrics=("bando",))
    e4, _, _ = ap.est_crossblur(u8, fn, CANDS)
    return {"ii_bando": res["bando"][0], "iv_xblur": e4}


t0 = time.time()
cal_scenes = [ap.Scene(ap.scene_1overf(H, W, SS, np.random.default_rng(s)), SS) for s in (9101, 9102)]
ev_scenes = {}
for name, phi in SCN:
    rng = np.random.default_rng(6262 + (0 if phi is None else int(phi)))
    ev_scenes[name] = ap.Scene(ap.scene_1overf(H, W, SS, rng) if phi is None else ap.scene_barcode(H, W, SS, phi, rng), SS)

log("=== (iv) PSF 合わせ と (ii) color-line の、PSF モデルのずれへの頑健さ（校正の比で割った値・σ=1.5）===")
log("各セル: 良好率（|ĉ/k−c|≤1）/ ばらつき [px]  被写体の順: " + ", ".join(n for n, _ in SCN) + "  ・k は校正の比")
for c in [6.0, 12.0]:
    log(f"--- c={c:g}")
    masks = {n: ap.eval_mask_from_full(s, c, roi=ROI)[0] for n, s in ev_scenes.items()}
    cmasks = [ap.eval_mask_from_full(s, c, roi=ROI)[0] for s in cal_scenes]
    for shape in SHAPES:
        fn = ap.SHAPES[shape][0]
        p = ap.centroid_img(ap.shape_stats(fn))
        for pert in PERTS:
            if pert == "g73" and shape != "cy_lr":
                continue
            # 校正
            ratios = {}
            for sc, cm in zip(cal_scenes, cmasks):
                u8 = ap.to_u8(render_p(sc, fn, c, pert, shape), 1.5, np.random.default_rng(3))
                for mt, e in estimate(u8, fn, p).items():
                    v = e[cm]
                    ratios.setdefault(mt, []).append(v[np.isfinite(v)] / c)
            k = {mt: float(np.median(np.concatenate(v))) for mt, v in ratios.items()}
            cells = {mt: [] for mt in k}
            for name, _ in SCN:
                u8 = ap.to_u8(render_p(ev_scenes[name], fn, c, pert, shape), 1.5, np.random.default_rng(4))
                for mt, e in estimate(u8, fn, p).items():
                    st = ap.stats(e / k[mt], c, masks[name])
                    cells[mt].append(f"{st['good']:.2f}/{st['sd']:.2f}")
            for mt in k:
                log(f"  {ap.SHAPES[shape][1]:<12s} {pert:<7s} {mt:<9s} k={k[mt]:.3f}: " + " | ".join(cells[mt]))
    print(f"c={c} 済 ({time.time() - t0:.0f} s)")

# 色のある被写体 ---------------------------------------------------------------
log()
log("=== 色のある被写体（R・B に独立な 1/f の色むら ±25 %・σ）: 良好率/ばらつき（raw）===")
rng = np.random.default_rng(301)
L = ap.scene_1overf(H, W, SS, rng)
chi_r = ap.scene_1overf(H, W, SS, rng, mean=0.0, std=1.0, clip=False)
chi_b = ap.scene_1overf(H, W, SS, rng, mean=0.0, std=1.0, clip=False)
for amp in [0.10, 0.25]:
    col = {"R": ap.Scene(np.clip(L * (1 + amp * chi_r), 0.02, 0.9), SS), "G": ap.Scene(L, SS),
           "B": ap.Scene(np.clip(L * (1 + amp * chi_b), 0.02, 0.9), SS)}
    for c in [3.0, 6.0, 12.0]:
        m = ap.eval_mask_from_full(col["G"], c, roi=ROI)[0]
        cells = []
        for shape in SHAPES:
            fn = ap.SHAPES[shape][0]
            p = ap.centroid_img(ap.shape_stats(fn))
            u8 = ap.to_u8(render_p(None, fn, c, "none", shape, scenes_color=col), 1.5, np.random.default_rng(5))
            ests = estimate(u8, fn, p)
            if shape == "cy_lr":
                d = ap.est_i_rust(u8)
                ests["i_hzncc"] = 2.0 * d / (p["R"][0] - p["B"][0])
            cells.append(f"{ap.SHAPES[shape][1]}: " + ", ".join(f"{mt} {ap.stats(e, c, m)['good']:.2f}/{ap.stats(e, c, m)['sd']:.2f}" for mt, e in ests.items()))
        log(f"色むら ±{amp * 100:.0f}% c={c:g}: " + " | ".join(cells))
log(f"所要 {time.time() - t0:.0f} s")
with open(os.path.join(OUT, "crossblur2.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(LOG) + "\n")
