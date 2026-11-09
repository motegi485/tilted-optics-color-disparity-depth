# -*- coding: utf-8 -*-
"""撮影データの画像サイズを確かめる下調べ（読み取りのみ）。
入力: data/captures の各セッションの先頭 PNG、data/test/2300.JPG
出力: 標準出力（各セッションの shape と平均色）
"""
import sys
from pathlib import Path
import cv2
import numpy as np

root = Path(sys.argv[1])
for d in sorted((root / "data" / "captures").iterdir()):
    if not d.is_dir():
        continue
    pngs = sorted(d.glob("*.png"))
    if not pngs:
        continue
    im = cv2.imread(str(pngs[0]), cv2.IMREAD_COLOR)
    if im is None:
        print(d.name, "read fail")
        continue
    m = im.reshape(-1, 3).mean(0)
    print(f"{d.name:20s} n={len(pngs):5d} shape={im.shape} meanBGR={m.round(1)}")
im = cv2.imread(str(root / "data" / "test" / "2300.JPG"))
print("test/2300.JPG", None if im is None else im.shape)
