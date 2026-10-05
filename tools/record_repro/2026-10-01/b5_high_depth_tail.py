"""B5: 推論帯あり（保存 npy）の有効画素に含まれる高深度の裾を数え、RMSE・MAE への寄与を記述する。

**評価方法の変更ではない**: ここでの「裾を除いた値」は感度の記述に限る。

定義:
  有効画素 : gate2（config.DEPTH_MIN < z < config.DEPTH_MAX）を通った画素（checks.evaluate_distance_accuracy と同じ）
  対象     : manifest の各セッションの npy（list_depth_files の昇順）から先頭 --skip-first 枚を除いた全フレーム・全画面
  裾       : z > --threshold-mm（既定 10000 mm。対象レンジの上限 4000 mm の 2.5 倍という記述用の目安）
  参考     : z > 2×gt の画素数
  統計     : checks.evaluate_distance_accuracy.summarize_errors（pooled）
自己照合: 全有効画素の mean_err・MAE・RMSE・有効画素数が --official-csv の同じセッションの値と一致しなければ exit 1。

実行（開発リポの root）:
  uv run python -u tools/record_repro/2026-10-01/b5_high_depth_tail.py \
      --manifest project_knowledge/records/2026-10-01/distance_manifest_b5.csv \
      --official-csv project_knowledge/records/2026-10-01/distance_accuracy_b5_20261001.csv \
      --skip-first 10 --out project_knowledge/records/2026-10-01/high_depth_tail_b5_20261001.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np

import config
from checks.evaluate_distance_accuracy import list_depth_files, load_manifest, summarize_errors

COLUMNS = ["session", "gt_mm", "frames_used", "n_valid_total", "threshold_mm", "n_tail", "tail_ppm",
           "frames_with_tail", "tail_row_min", "tail_row_p10", "tail_row_median", "tail_row_p90", "tail_row_max",
           "tail_depth_max_mm", "n_gt_2x_gt", "mean_err_all_mm", "mae_all_mm", "rmse_all_mm", "median_err_all_mm",
           "mean_err_excl_tail_mm", "mae_excl_tail_mm", "rmse_excl_tail_mm", "median_err_excl_tail_mm",
           "official_check"]


def tail_stats(out_dir: Path, gt_mm: int, *, skip_first: int, threshold_mm: float) -> dict:
    files = list_depth_files(out_dir)[skip_first:]
    if not files:
        raise ValueError(f"{out_dir}: 集計フレームが 0 件")
    errors, tail_rows = [], []
    n_valid = n_gt2x = frames_with_tail = 0
    tail_max = float("nan")
    for p in files:
        depth = np.load(str(p)).astype(np.float64)
        mask = (depth > config.DEPTH_MIN) & (depth < config.DEPTH_MAX)
        valid = depth[mask]
        n_valid += valid.size
        n_gt2x += int((valid > 2 * gt_mm).sum())
        tail = mask & (depth > threshold_mm)
        k = int(tail.sum())
        if k:
            frames_with_tail += 1
            tail_rows.append(np.nonzero(tail)[0])
            m = float(depth[tail].max())
            tail_max = m if math.isnan(tail_max) else max(tail_max, m)
        errors.append(valid - float(gt_mm))
    err = np.concatenate(errors)
    keep = (err + gt_mm) <= threshold_mm
    all_s = summarize_errors(err)
    excl_s = summarize_errors(err[keep])
    rows = np.concatenate(tail_rows) if tail_rows else np.empty(0, dtype=np.int64)
    n_tail = int(rows.size)

    def q(v: float) -> float:
        return float(np.percentile(rows, v)) if n_tail else float("nan")

    return {"gt_mm": gt_mm, "frames_used": len(files), "n_valid_total": n_valid, "threshold_mm": threshold_mm,
            "n_tail": n_tail, "tail_ppm": n_tail / n_valid * 1e6, "frames_with_tail": frames_with_tail,
            "tail_row_min": float(rows.min()) if n_tail else float("nan"), "tail_row_p10": q(10),
            "tail_row_median": q(50), "tail_row_p90": q(90),
            "tail_row_max": float(rows.max()) if n_tail else float("nan"), "tail_depth_max_mm": tail_max,
            "n_gt_2x_gt": n_gt2x,
            "mean_err_all_mm": all_s["mean_err_mm"], "mae_all_mm": all_s["mae_mm"], "rmse_all_mm": all_s["rmse_mm"],
            "median_err_all_mm": all_s["median_err_mm"],
            "mean_err_excl_tail_mm": excl_s["mean_err_mm"], "mae_excl_tail_mm": excl_s["mae_mm"],
            "rmse_excl_tail_mm": excl_s["rmse_mm"], "median_err_excl_tail_mm": excl_s["median_err_mm"]}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--official-csv", type=Path, required=True)
    ap.add_argument("--skip-first", type=int, default=10)
    ap.add_argument("--threshold-mm", type=float, default=10000.0)
    ap.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    with open(a.official_csv, encoding="utf-8", newline="") as f:
        official = {r["session"]: r for r in csv.DictReader(f)}
    rows, ok = [], True
    for session, gt_mm, _note in load_manifest(a.manifest):
        r = {"session": session, **tail_stats(a.outputs_dir / session, gt_mm, skip_first=a.skip_first,
                                              threshold_mm=a.threshold_mm)}
        o = official[session]
        checks = [int(o["n_valid_total"]) == r["n_valid_total"], int(o["frames_used"]) == r["frames_used"]]
        checks += [math.isclose(float(o[k]), r[f"{k[:-3]}_all_mm"], rel_tol=1e-12, abs_tol=1e-9)
                   for k in ("mean_err_mm", "mae_mm", "rmse_mm", "median_err_mm")]
        r["official_check"] = "match" if all(checks) else "MISMATCH"
        ok &= all(checks)
        rows.append(r)
        print(f"{session:14s} valid={r['n_valid_total']:9d} tail(>{a.threshold_mm:.0f})={r['n_tail']:6d} "
              f"({r['tail_ppm']:6.1f} ppm, {r['frames_with_tail']:3d}/{r['frames_used']} frames, "
              f"row med {r['tail_row_median']}) >2gt={r['n_gt_2x_gt']:6d} | all MAE={r['mae_all_mm']:.2f} "
              f"RMSE={r['rmse_all_mm']:.2f} | excl MAE={r['mae_excl_tail_mm']:.2f} RMSE={r['rmse_excl_tail_mm']:.2f}"
              f" | official={r['official_check']}")
    with open(a.out, "x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nofficial self-check all match: {ok}\nCSV: {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
