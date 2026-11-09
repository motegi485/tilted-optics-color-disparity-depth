# -*- coding: utf-8 -*-
"""フォルダ内の画像を番号付きで 1 枚に並べる（読み取りのみ）。
入力: argv[1]=画像フォルダ, argv[2]=出力 PNG
"""
import sys
from pathlib import Path
import cv2
import numpy as np

d = Path(sys.argv[1])
tiles = []
for f in sorted(d.iterdir()):
    im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
    if im is None:
        print("skip", f.name)
        continue
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:
        a = im[:, :, 3:4].astype(float) / 255
        im = (im[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    print(f.name, im.shape)
    s = 220 / max(im.shape[:2])
    im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((240, 240, 3), 200, np.uint8)
    canvas[18:18 + im.shape[0], 10:10 + im.shape[1]] = im
    cv2.putText(canvas, f.name[:28], (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    tiles.append(canvas)
cols = 5
while len(tiles) % cols:
    tiles.append(np.full_like(tiles[0], 200))
cv2.imwrite(sys.argv[2], np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
