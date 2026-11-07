# 読み取りのみ: 実物（印刷紙と筒）の行別の距離誤差を、ノイズ板（B5）と同じ考え方で見る
import glob
import os
import sys

import cv2
import numpy as np

root = r"C:\Users\t5424001\dev\tilted_optics_color_disparity_depth"
out = __import__("tempfile").gettempdir()
vp, up, ap = 432599.32942224736, -63.927380677321885, -26.568072891235353
vn, un, an = 423627.23803714506, -21.368758126996866, -22.562290573120116

sp = sorted(glob.glob(os.path.join(root, "data", "captures", "b5_scene_01", "*.png")))
sn = sorted(glob.glob(os.path.join(root, "outputs", "b5_scene_01", "*.npy")))
raw = cv2.imread(sp[100])
# 共有の前処理と同じ形（3:2 中央クロップ → 528×352）。確認用の図のみ
h, w = raw.shape[:2]
cw = int(round(h * 3 / 2))
x0 = (w - cw) // 2
crop = raw[:, x0:x0 + cw]
small = cv2.resize(crop, (528, 352), interpolation=cv2.INTER_AREA)
D = np.stack([np.load(sn[i]) for i in range(10, 557, 10)]).astype(np.float64)
valid = np.isfinite(D)
count = valid.mean(axis=0)
med = np.nanmedian(np.where(valid, D, np.nan), axis=0)
vis = small.copy()
mask = np.isfinite(med) & (count > 0.2)
cm = cv2.applyColorMap(np.clip(((np.nan_to_num(med) - 1000) / 3000 * 255), 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
vis[mask] = cm[mask]
both = np.hstack([small, vis])
both = cv2.resize(both, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(os.path.join(out, "scene_check.png"), both)

# 色かぶり: 紙の余白（処理画像 310〜345 行、列 300〜500）
roi = small[310:345, 300:500].astype(np.float64)
b, g, r = roi[..., 0].mean(), roi[..., 1].mean(), roi[..., 2].mean()
print(f"紙の余白(処理画像 行310-345, 列300-500): B={b:.1f} G={g:.1f} R={r:.1f} R/G={r/g:.3f} B/G={b/g:.3f}")

# 列プロファイル（筒の範囲の確認）
print("列ごとの有効画素の割合（16 列おき）")
cp = valid.mean(axis=(0, 1))
print(" ".join(f"{i}:{cp[i]:.2f}" for i in range(0, 528, 32)))

# 行別・列帯別の中央値
for label, c0, c1 in [("左 40-90", 40, 90), ("右 240-300", 240, 300), ("右 300-400", 300, 400), ("右 400-490", 400, 490), ("筒 110-190", 110, 190)]:
    print("---", label)
    for r0, r1 in [(12, 40), (40, 70), (70, 100), (100, 130), (130, 160), (160, 190), (190, 220), (220, 260), (260, 300), (300, 340)]:
        x = D[:, r0:r1, c0:c1]
        x = x[np.isfinite(x)]
        if x.size < 50:
            print(f"  行{r0:3d}-{r1:3d}: n={x.size}")
            continue
        m = np.median(x)
        rm = 0.5 * (r0 + r1)
        dtp = (rm - (vp / 2800 + up)) / ap
        dtn = (rm - (vn / 2800 + un)) / an
        print(f"  行{r0:3d}-{r1:3d}: n={x.size:6d} 中央値={m:7.1f} 誤差(2800基準)={100*(m/2800-1):+6.1f}% q5={np.percentile(x,5):7.1f} q95={np.percentile(x,95):7.1f} d_true pos={dtp:+.2f} neg={dtn:+.2f}")
