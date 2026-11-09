# -*- coding: utf-8 -*-
"""(b) 通常シーン b5_scene_01（筒 1430 mm・車を印刷した紙 2800 mm・記録票 2026-10-01 §4）で推定法を比べる。読み取りのみ・非公式の再解析。

領域（処理画像 528×352 の座標）: 筒の輪郭 x∈[95,112]∪[190,206]（frame 10 のエッジ列 101〜106・196〜199〈scene_cylinder_edge_rows.py〉を含む・
背景の紙が混ざりうる）／筒の内側 x∈[113,189]／紙 x∈[33,85]∪[216,495) かつ y<295（筒の輪郭から 16 px 以上離す・下端の白い余白を除く）。
フレーム 10, 60, ..., 460（10 枚・静止した配置）。誤差は推論帯を通った後の深度（gate2 後）で、紙は 2800 mm、筒の輪郭は 1430 mm を正解とする。
出力: verReal_out/v3_scene.txt・v3_scene_rows.csv・v3_scene_frame10.png
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vr_lib as L
import vr_eval as E
import vr_scene as SC

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verReal_out")
SES = "b5_scene_01"
FRAMES = list(range(10, 470, 50))
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
log = open(os.path.join(OUT, "v3_scene.txt"), "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")
    log.flush()


yy, xx = np.indices((L.H, L.W))
REG = {
    "紙": (((xx >= 33) & (xx <= 85)) | ((xx >= 216) & (xx < 495))) & (yy >= 12) & (yy < 295),
    "筒の輪郭": (((xx >= 95) & (xx <= 112)) | ((xx >= 190) & (xx <= 206))) & L.ROI,
    "筒の内側": (xx >= 113) & (xx <= 189) & L.ROI,
}
GT = {"紙": 2800.0, "筒の輪郭": 1430.0}

t0 = time.time()
store = {c[:2]: [] for c in SC.SCOMBOS}   # フレームごとの推論帯後の d
zs = {c[:2]: [] for c in SC.SCOMBOS}
feats_list = []
img10 = None
for idx in FRAMES:
    proc = L.load_proc(SES, idx)
    f, d = SC.compute(proc)
    feats_list.append(f)
    if idx == 10:
        img10 = proc
    for (m, sel, _) in SC.SCOMBOS:
        dm = E.combo_disp(f, d, m, sel)
        z = L.to_depth(dm)
        db = L.banded(dm).astype(np.float64)
        db[~np.isfinite(z)] = np.nan
        store[(m, sel)].append(db)
        zs[(m, sel)].append(z)
p(f"計算 {time.time() - t0:.0f}s（{len(FRAMES)} フレーム）")

# (i) の紙の深度の空間的な広がり（正解 2800 mm の目安の妥当性）
zi = np.stack(zs[("i", "S_i")])
p("\n(i) の紙の深度の中央値（列帯 × 行帯・10 フレーム）: 紙が一様な距離かの確認")
for (x0, x1) in ((33, 86), (216, 300), (300, 400), (400, 495)):
    parts = []
    for (y0, y1) in ((12, 90), (90, 150), (150, 220), (220, 295)):
        sub = zi[:, y0:y1, x0:x1]
        v = sub[np.isfinite(sub)]
        parts.append(f"y{y0}-{y1}: {np.median(v):.0f}（{v.size}）" if v.size else f"y{y0}-{y1}: —")
    p(f"  x{x0}-{x1}: " + " / ".join(parts))

rows = []
nf = len(FRAMES)
base_n = {}
for (m, sel, label) in SC.SCOMBOS:
    D = np.stack(store[(m, sel)])
    Z = np.stack(zs[(m, sel)])
    nvalid = np.isfinite(D).sum(0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tstd = np.nanstd(D, axis=0)
    tstd[nvalid < 5] = np.nan
    sdev = np.stack([SC.local_dev(D[k]) for k in range(nf)])
    di = np.stack(store[("i", "S_i")])
    for rname, rmask in REG.items():
        for sub in ["all"] + [f"h:{lab}" for lab in E.HB_LABEL] + ["inSi", "outSi"]:
            msk = np.zeros((nf, L.H, L.W), bool)
            for k, f in enumerate(feats_list):
                mm = rmask.copy()
                if sub.startswith("h:"):
                    kb = E.HB_LABEL.index(sub[2:])
                    hb, _ = E.bins_of(f)
                    mm &= hb == kb
                elif sub == "inSi":
                    mm &= f["S_i"]
                elif sub == "outSi":
                    mm &= ~f["S_i"]
                msk[k] = mm
            v = msk & np.isfinite(Z)
            n = int(v.sum())
            row = dict(region=rname, method=m, sel=sel, label=label, subset=sub, n_per_frame=n / nf)
            if rname in GT and n:
                st = E.err_stats(Z[v] - GT[rname], GT[rname])
                row.update(bias=st["bias"], median=st["median"], mae=st["mae"], good10=st["good10"],
                           depth_median=float(np.median(Z[v])))
            else:
                row.update(bias=np.nan, median=np.nan, mae=np.nan, good10=np.nan,
                           depth_median=float(np.median(Z[v])) if n else np.nan)
            # 時間方向のばらつき（5 フレーム以上有効な画素の d の std の中央値）
            tm = msk[0] & np.isfinite(tstd)
            row["tstd_median"] = float(np.median(tstd[tm])) if tm.any() else np.nan
            row["n_tstd_px"] = int(tm.sum())
            sv = sdev[v[:, :, :] & np.isfinite(sdev)]
            row["sdev_median"] = float(np.median(sv)) if sv.size else np.nan
            both = v & np.isfinite(di)
            row["agree_i_absdiff_median"] = float(np.median(np.abs(D[both] - di[both]))) if both.any() else np.nan
            row["n_both_i"] = int(both.sum()) / nf
            rows.append(row)

with open(os.path.join(OUT, "v3_scene_rows.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)


def get(region, m, sel, sub="all"):
    return [r for r in rows if r["region"] == region and r["method"] == m and r["sel"] == sel and r["subset"] == sub][0]


for rname in REG:
    p(f"\n=== {rname}（正解 {GT.get(rname, '—')} mm）===")
    b = get(rname, "i", "S_i")
    for (m, sel, label) in SC.SCOMBOS:
        r = get(rname, m, sel)
        ro = get(rname, m, sel, "outSi")
        p(f"  {label:26s} 有効 {r['n_per_frame']:7.0f}/フレーム（×{r['n_per_frame'] / max(b['n_per_frame'], 1e-9):.2f}）"
          f" 深度中央値 {r['depth_median']:6.0f} MAE {r['mae']:7.1f} 中央値の偏り {r['median']:+7.1f} 良好10% {r['good10']:.3f}"
          f" | 時間std中央値 {r['tstd_median']:.3f} px 空間偏差 {r['sdev_median']:.3f} px (i)との差 {r['agree_i_absdiff_median']:.3f} px"
          f" | S_i の外 {ro['n_per_frame']:6.0f} 深度中央値 {ro['depth_median']:6.0f} MAE {ro['mae']:7.1f} 良好10% {ro['good10']:.3f}")
p("\n=== 紙（2800 mm）を窓の h 別に ===")
for (m, sel, label) in SC.SCOMBOS:
    parts = []
    for lab in E.HB_LABEL:
        r = get("紙", m, sel, f"h:{lab}")
        parts.append(f"{lab}: n {r['n_per_frame']:5.0f} 中央 {r['depth_median']:5.0f} MAE {r['mae']:6.0f} 良10 {r['good10']:.2f} tstd {r['tstd_median']:.2f}")
    p(f"  {label:26s} " + " | ".join(parts))

# 図: frame 10 の推論帯後の深度
fig, axs = plt.subplots(3, 3, figsize=(15, 10.5))
gray = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
light = 0.45 + 0.55 * gray
for ax, (m, sel, label) in zip(axs.ravel(), SC.SCOMBOS):
    z = zs[(m, sel)][0]
    ax.imshow(light, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    im = ax.imshow(np.ma.masked_invalid(z), cmap="viridis", vmin=1000, vmax=4000, interpolation="nearest")
    ax.set_title(f"{label}（{int(np.isfinite(z).sum())} 画素）", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.03, pad=0.02, label="深度 [mm]（推論帯の後・1000〜4000 で表示）")
fig.savefig(os.path.join(OUT, "v3_scene_frame10.png"), dpi=90)
p(f"\n合計 {time.time() - t0:.0f}s")
log.close()
