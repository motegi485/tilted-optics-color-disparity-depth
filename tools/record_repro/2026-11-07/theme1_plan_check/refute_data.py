# 読み取りのみ: 既存録画の色かぶりと、実物（筒・印刷紙）での物体ごとの距離誤差
import glob
import os

import cv2
import numpy as np

root = r"C:\Users\t5424001\dev\tilted_optics_color_disparity_depth"
scene_png = sorted(glob.glob(os.path.join(root, "data", "captures", "b5_scene_01", "*.png")))
scene_npy = sorted(f for f in glob.glob(os.path.join(root, "outputs", "b5_scene_01", "*.npy")))
print("frames", len(scene_png), len(scene_npy))

# 1. 色かぶり: ノイズ板（白黒印刷）と紙の余白
for sess in ["b5_dist_1450", "b5_dist_2250", "b5_dist_3050"]:
    ps = sorted(glob.glob(os.path.join(root, "data", "captures", sess, "*.png")))
    if not ps:
        print(sess, "なし")
        continue
    im = cv2.imread(ps[20]).astype(np.float64)
    b, g, r = im[..., 0].mean(), im[..., 1].mean(), im[..., 2].mean()
    print(f"{sess}: 平均 B={b:.1f} G={g:.1f} R={r:.1f}  R/G={r/g:.3f} B/G={b/g:.3f}  飽和率 G>=250: {(im[...,1]>=250).mean():.4f}")

im = cv2.imread(scene_png[100]).astype(np.float64)
print("scene shape", im.shape)
# 紙の余白（記録票: 処理画像の 301 行以下、筒以外）
roi = im[310:345, 300:500]
b, g, r = roi[..., 0].mean(), roi[..., 1].mean(), roi[..., 2].mean()
print(f"紙の余白（行310-345, 列300-500）: B={b:.1f} G={g:.1f} R={r:.1f} R/G={r/g:.3f} B/G={b/g:.3f}")

# 2. 物体ごとの距離: 筒 1400 mm、紙 2800 mm（公称。b5 の配置は scene_01 と同じとの記録）
frames = range(10, min(len(scene_npy), 557))
stack = []
for i in list(frames)[::10]:
    arr = np.load(scene_npy[i])
    stack.append(arr)
D = np.stack(stack).astype(np.float64)
print("depth shape", D.shape)
valid = np.isfinite(D)
print("有効画素/フレーム 平均", valid.sum(axis=(1, 2)).mean())
# 列ごとの有効数の分布で筒の位置を見る
vals_all = D[valid]
hist, edges = np.histogram(vals_all, bins=[0, 800, 1000, 1200, 1300, 1400, 1500, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3600, 4000, 6000, 10000, 1e9])
for h, a, b_ in zip(hist, edges[:-1], edges[1:]):
    print(f"  {a:8.0f}-{b_:8.0f}: {h:7d} {100*h/hist.sum():5.1f}%")

tube_cols = (100, 198)
tube = D[:, :, tube_cols[0]:tube_cols[1]]
# 筒の列の中で 1000-2000 mm を筒候補、列外で 2000-4000 mm を紙候補
tube_v = tube[np.isfinite(tube) & (tube > 800) & (tube < 2000)]
paper = np.concatenate([D[:, :, :tube_cols[0]], D[:, :, tube_cols[1]:]], axis=2)
paper_v = paper[np.isfinite(paper) & (paper > 2000) & (paper < 4000)]
paper_all = paper[np.isfinite(paper)]
tube_all = tube[np.isfinite(tube)]
for name, gt, sel, allv in [("筒", 1400.0, tube_v, tube_all), ("紙", 2800.0, paper_v, paper_all)]:
    med = np.median(sel)
    mad = np.median(np.abs(sel - med))
    q = np.percentile(sel, [5, 25, 75, 95])
    print(f"{name}: GT(公称)={gt:.0f} 範囲内中央値={med:.1f} MAD={mad:.1f} 5/25/75/95%={q.round(1)} 範囲内/全={sel.size}/{allv.size} 全体の中央値={np.median(allv):.1f}")
    print(f"   相対誤差={(med-gt)/gt*100:+.2f}%  1/z 誤差換算 K=18: {18000*(1/med-1/gt):+.3f} px")
# フレームごとの中央値のばらつき
tm = [np.nanmedian(np.where((t > 800) & (t < 2000), t, np.nan)) for t in tube]
pm = [np.nanmedian(np.where((p > 2000) & (p < 4000), p, np.nan)) for p in paper]
print("フレーム中央値の範囲: 筒", np.nanmin(tm), np.nanmax(tm), " 紙", np.nanmin(pm), np.nanmax(pm))
# 紙の行ごとの中央値（合焦近傍の帯が見えるか）
print("紙（列外）の行帯別 中央値と画素数")
for r0, r1 in [(0, 60), (60, 120), (120, 180), (180, 240), (240, 300), (300, 352)]:
    x = paper[:, r0:r1, :]
    x = x[np.isfinite(x)]
    if x.size:
        print(f"  行{r0:3d}-{r1:3d}: n={x.size:7d} 中央値={np.median(x):7.1f} 5%={np.percentile(x,5):7.1f} 95%={np.percentile(x,95):7.1f}")
