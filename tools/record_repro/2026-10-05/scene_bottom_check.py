"""scene_bottom_check.py — b5_scene_01 の処理画像 301 行目以下にある有効画素の深度と列の分布。

保存 npy の全フレーム（先頭 10 を除く）で確かめる。
読み取りのみ（outputs/b5_scene_01/*.npy・git 追跡外）。新しい評価指標ではない。

実行（リポジトリ root）: uv run python tools/record_repro/2026-10-05/scene_bottom_check.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

ROW0 = 301            # 確認する境界（処理画像の行）
SKIP = 10             # 集計の --skip-first 10 と同じ

files = sorted((config.OUTPUTS_DIR / "b5_scene_01").glob("[0-9][0-9][0-9][0-9][0-9][0-9].npy"))
print(f"npy {len(files)} frames, used {len(files) - SKIP} (skip first {SKIP}), rows >= {ROW0}")
counts, vals, cols, totals = [], [], [], []
for p in files[SKIP:]:
    full = np.load(p)
    d = full[ROW0:, :]
    ok = np.isfinite(d)
    counts.append(int(ok.sum()))
    totals.append(int(np.isfinite(full).sum()))
    vals.append(d[ok])
    cols.append(np.nonzero(ok)[1])
v = np.concatenate(vals)
x = np.concatenate(cols)
print("valid pixels per frame (rows >= %d): mean %.1f, min %d, max %d; share of all valid pixels %.3f"
      % (ROW0, np.mean(counts), min(counts), max(counts), sum(counts) / sum(totals)))
q = np.percentile(v, [0, 1, 5, 25, 50, 75, 95, 99, 100])
print("depth [mm]: min %.0f p1 %.0f p5 %.0f p25 %.0f median %.0f p75 %.0f p95 %.0f p99 %.0f max %.0f" % tuple(q))
print("fraction within 1425-1492 mm: %.3f" % ((v >= 1425) & (v <= 1492)).mean())
qx = np.percentile(x, [0, 5, 50, 95, 100])
print("column x' (processed image): min %d p5 %d median %d p95 %d max %d" % tuple(qx.astype(int)))
