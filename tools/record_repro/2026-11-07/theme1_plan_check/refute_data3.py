# 読み取りのみ: b5_scene_01 の印刷紙（2800 mm）での距離誤差を色の彩度別に集計し、ノイズ板と比べる
import glob
import os
import sys
from pathlib import Path

REPO = Path(r"C:\Users\t5424001\dev\tilted_optics_color_disparity_depth")
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

import cv2
import numpy as np

from tools.analysis_common import preprocess_to

sp = sorted(glob.glob(str(REPO / "data" / "captures" / "b5_scene_01" / "*.png")))
sn = sorted(glob.glob(str(REPO / "outputs" / "b5_scene_01" / "*.npy")))
idx = list(range(10, 557, 5))
GT = 2800.0
K_ref = 18.0e3  # px*mm（理論値）

imgs = []
errs = []
rows_all = []
chroma_all = []
vals_all = []
fr_all = []
for fi, i in enumerate(idx):
    raw = cv2.imread(sp[i])
    proc = preprocess_to(raw, (528, 352)).astype(np.float32)
    # 白地（紙の余白）で正規化した R-B の彩度
    R = cv2.blur(proc[..., 2], (9, 9)) / 0.838
    B = cv2.blur(proc[..., 0], (9, 9)) / 0.788
    ch = (R - B) / np.maximum(R + B, 1)
    z = np.load(sn[i]).astype(np.float64)
    mask = np.isfinite(z)
    mask[:12] = False
    mask[295:] = False  # 印刷の余白より下を除く
    mask[:, 92:208] = False  # 筒とその縁を除く
    ys, xs = np.nonzero(mask)
    vals_all.append(z[ys, xs])
    chroma_all.append(ch[ys, xs])
    rows_all.append(ys)
    fr_all.append(np.full(ys.size, fi))
z = np.concatenate(vals_all)
ch = np.concatenate(chroma_all)
rw = np.concatenate(rows_all)
fr = np.concatenate(fr_all)
print("印刷紙の有効画素 合計", z.size, " /フレーム", z.size / len(idx), " 使ったフレーム数", len(idx), "（静止シーンなので独立な標本は 1 シーン）")


def stat(sel, name):
    x = z[sel]
    if x.size < 30:
        print(f"{name}: n={x.size}")
        return
    med = np.median(x)
    mae = np.mean(np.abs(x - GT))
    print(
        f"{name}: n/フレーム={x.size/len(idx):7.1f} 中央値={med:7.1f} ({100*(med/GT-1):+5.1f}%) MAE={mae:7.1f} mm  bias={np.mean(x-GT):+7.1f}  |誤差|>10%={100*np.mean(np.abs(x-GT)>280):5.1f}%  |誤差|>30%={100*np.mean(np.abs(x-GT)>840):5.1f}%  視差換算(K=18)の中央値={K_ref*(1/med-1/GT):+.3f}px"
    )


stat(np.ones_like(z, bool), "全体")
print("彩度の分位", np.percentile(ch, [1, 5, 25, 50, 75, 95, 99]).round(3))
stat(np.abs(ch) < 0.05, "|彩度|<0.05（無彩色）")
stat((np.abs(ch) >= 0.05) & (np.abs(ch) < 0.15), "0.05<=|彩度|<0.15")
stat(ch >= 0.15, "彩度>=0.15（赤より）")
stat(ch <= -0.15, "彩度<=-0.15（青より）")
for th in (0.3, 0.5):
    stat(ch >= th, f"彩度>={th}")
print("--- 行帯別（無彩色のみ）")
for r0, r1 in [(12, 60), (60, 100), (100, 130), (130, 170), (170, 230), (230, 295)]:
    stat((rw >= r0) & (rw < r1) & (np.abs(ch) < 0.05), f"行{r0}-{r1}")
print("--- 行帯別（赤より >=0.15）")
for r0, r1 in [(12, 60), (60, 100), (100, 130), (130, 170), (170, 230), (230, 295)]:
    stat((rw >= r0) & (rw < r1) & (ch >= 0.15), f"行{r0}-{r1}")

# 筒の縁（シルエット）: 列 92-108 と 190-208
vals = []
for i in idx:
    z0 = np.load(sn[i]).astype(np.float64)
    m = np.isfinite(z0)
    m2 = np.zeros_like(m)
    m2[:, 92:108] = True
    m2[:, 190:208] = True
    vals.append(z0[m & m2])
tz = np.concatenate(vals)
tz1 = tz[(tz > 1000) & (tz < 2000)]
print(f"筒の縁: n/フレーム={tz.size/len(idx):.0f} 1000-2000mm に入る割合={tz1.size/tz.size:.3f} 中央値={np.median(tz1):.1f} MAD={np.median(np.abs(tz1-np.median(tz1))):.1f} 5/95%={np.percentile(tz1,[5,95]).round(1)}")
print("筒の幅: 約 94 px → 1430 mm で", round(94 * 1430 / 2639), "mm（最近面 1430 mm。縁は半径ぶん奥）")
