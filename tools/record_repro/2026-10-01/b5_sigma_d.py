"""B5: 視差の時間的安定性 σ_d。

定義: tools.analyze_disparity_settings.stability_summary
（tools.analyze_disparity_stability.pixel_stats）を、各セッションの frame 10..39（list_frames(skip_first=10,
max_frames=30)）の**ゲートなしの生視差**（disparity_of_proc・gate1 も推論帯も掛けない）に適用する。
画素の選別は valid_frac ≥ 0.5 かつ n_valid ≥ 2、全セッションの画素をプールした中央値・平均と符号反転率。

自己検証: 同じ関数を既存キャッシュ outputs/cache/disp/K25_S41_T20/（frame 10..39）
に掛け、P2/edge_th の T=20 行の値を再現することを確かめる（読み取りのみ）。
B5 側の視差はキャッシュに書かずメモリ上で計算する。
実行（開発リポの root）: uv run python -u tools/record_repro/2026-10-01/b5_sigma_d.py --manifest outputs/distance_manifest_b5.csv --out <csv>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import cv2
import numpy as np

import config
from checks.evaluate_distance_accuracy import load_manifest
from tools.analysis_common import DispSettings, disparity_of_proc, list_frames, load_stack, preprocess_to
from tools.analyze_disparity_settings import stability_summary

OLD_MANIFEST = Path("outputs/distance_manifest_all.csv")
CACHE = Path(config.OUTPUTS_DIR) / "cache" / "disp" / "K25_S41_T20"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = DispSettings()
    if s.key() != "K25_S41_T20":
        print(f"[ERROR] 設定が B5 と異なります: {s.key()}")
        return 2
    rows = []
    # 自己検証: 旧データのキャッシュ（読み取りのみ）
    old = {}
    for sess, _gt, _note in load_manifest(OLD_MANIFEST):
        files = sorted((CACHE / sess).glob("??????.npy"))
        if len(files) != 30:
            print(f"[ERROR] キャッシュ {sess} が 30 枚ではありません: {len(files)}")
            return 2
        old[sess] = load_stack(files)
    st_old = stability_summary(old)
    rows.append({"set": "2026-09-15/16 cache K25_S41_T20 (17 sessions, frames 10..39) [self-check]", **st_old})
    print("[self-check old cache]", st_old, flush=True)
    # B5
    stacks, groups = {}, {}
    for sess, gt, note in load_manifest(Path(args.manifest)):
        frames = list_frames(Path(config.CAPTURES_DIR) / sess, skip_first=10, max_frames=30)
        stacks[sess] = np.stack([disparity_of_proc(preprocess_to(cv2.imread(str(p)), s.image_size), s) for p in frames])
        groups.setdefault(note.split(" ")[0] + (" near" if "near" in note else " far" if "far" in note else ""),
                          []).append(sess)
        one = stability_summary({sess: stacks[sess]})
        rows.append({"set": f"{sess} (gt {gt})", **one})
        print(f"[b5] {sess}: {one}", flush=True)
    for g, ss in groups.items():
        rows.append({"set": f"B5 {g} ({len(ss)} sessions)", **stability_summary({k: stacks[k] for k in ss})})
    rows.append({"set": "B5 all 15 sessions", **stability_summary(stacks)})
    print("[b5 all]", rows[-1], flush=True)
    with open(args.out, "x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
