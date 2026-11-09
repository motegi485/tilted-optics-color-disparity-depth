# -*- coding: utf-8 -*-
"""撮影セッションの中身を目で確かめるための縮小一覧を作る（読み取りのみ）。
入力: argv[1]=開発リポ root, argv[2]=出力 PNG
出力: 各セッションの中央フレームを 3:2 中央クロップ→264x176 に縮小して並べた PNG
"""
import sys
from pathlib import Path
import cv2
import numpy as np

root = Path(sys.argv[1])
out = Path(sys.argv[2])
tiles = []
for d in sorted((root / "data" / "captures").iterdir()):
    pngs = sorted(d.glob("*.png")) if d.is_dir() else []
    if not pngs or d.name.startswith("b5_dist") or d.name.startswith("dist_"):
        continue
    im = cv2.imread(str(pngs[len(pngs) // 2]))
    h, w = im.shape[:2]
    x0 = (w - 1080) // 2
    im = cv2.resize(im[:, x0:x0 + 1080], (264, 176), interpolation=cv2.INTER_AREA)
    cv2.putText(im, d.name, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    tiles.append(im)
# 校正板の代表として b5_dist_2050 も 1 枚
p = sorted((root / "data" / "captures" / "b5_dist_2050").glob("*.png"))[10]
im = cv2.imread(str(p)); x0 = (im.shape[1] - 1080) // 2
im = cv2.resize(im[:, x0:x0 + 1080], (264, 176), interpolation=cv2.INTER_AREA)
cv2.putText(im, "b5_dist_2050", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
tiles.append(im)
t = cv2.imread(str(root / "data" / "test" / "2300.JPG"))
t = cv2.resize(t, (264, 176), interpolation=cv2.INTER_AREA)
cv2.putText(t, "test/2300.JPG", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
tiles.append(t)
cols = 5
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
cv2.imwrite(str(out), np.vstack(rows))
print("tiles", len(tiles))
