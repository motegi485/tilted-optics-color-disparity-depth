# -*- coding: utf-8 -*-
"""2026-07〜08 のセッションの中央フレームを並べた一覧（どれがブラインド等の横縞の多い場面かを目で選ぶため）。読み取りのみ。"""
import os
import sys

import cv2
import numpy as np

ROOT = "C:/Users/t5424001/dev/tilted_optics_color_disparity_depth"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verReal_out")
CAP = os.path.join(ROOT, "data", "captures")
sessions = sorted(s for s in os.listdir(CAP) if s.startswith("202607") or s.startswith("202608") or s == "clip92")
tiles = []
for s in sessions:
    fs = sorted(f for f in os.listdir(os.path.join(CAP, s)) if f.endswith(".png"))
    img = cv2.imread(os.path.join(CAP, s, fs[len(fs) // 2]))
    h, w = img.shape[:2]
    print(s, len(fs), w, h)
    if (w, h) == (1280, 720):
        img = img[:, 100:1180]
    t = cv2.resize(img, (324, 216), interpolation=cv2.INTER_AREA)
    cv2.putText(t, s, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    tiles.append(t)
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
cv2.imwrite(os.path.join(OUT, "v0_contact.png"), np.vstack(rows))
