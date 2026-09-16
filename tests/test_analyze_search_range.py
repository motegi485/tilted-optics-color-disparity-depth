"""tools/analyze_search_range.py の HW 非依存テスト（合成スタック・tmp_path・monkeypatch）。"""
import csv
import json
import math

import cv2
import numpy as np
import pytest

import config
import tools.analysis_common as common
from core.params import DepthParams
from tools.analyze_search_range import (
    BENCH_COLUMNS,
    CMP_COLUMNS,
    DIST_COLUMNS,
    PRED_COLUMNS,
    bench_search_ranges,
    compare_row,
    distribution_row,
    main,
    predicted_max_disparity,
)

W, H = config.IMAGE_SIZE
PARAMS = DepthParams(v_pos=426165.15, u_pos=-61.04, a_pos=-26.43, w_pos=0.0,
                     v_neg=411311.86, u_neg=-17.39, a_neg=-22.76, w_neg=0.0)


def test_distribution_row_percentiles_and_fractions():
    stack = np.full((2, 4, 5), np.nan, dtype=np.float32)
    stack[0, 0, :] = [0.5, -1.0, 2.0, 8.0, 14.45]
    stack[1, 1, :] = [0.5, 1.0, -2.0, 8.0, 14.45]
    r = distribution_row("s", 2000, stack, reach_max=14.5, high=7.5)
    assert set(r) == set(DIST_COLUMNS) and r["n_valid"] == 10 and r["frames"] == 2
    assert r["p50"] == pytest.approx(np.percentile(np.abs(stack[~np.isnan(stack)]), 50))
    assert r["max"] == pytest.approx(14.45) and r["frac_ge_reach"] == 0.2 and r["frac_gt_high"] == 0.4
    empty = distribution_row("s", 2000, np.full((1, 2, 2), np.nan, dtype=np.float32), reach_max=14.5, high=7.5)
    assert empty["n_valid"] == 0 and math.isnan(empty["p99"])


def test_compare_row_changed_lost_gained():
    base = np.array([[[1.0, 2.0, np.nan, 5.0]]], dtype=np.float32)
    other = np.array([[[1.2, 2.0, 3.0, np.nan]]], dtype=np.float32)   # 1 画素が 0.2 変化・1 画素喪失・1 画素獲得
    other[0, 0, 0] = 1.8                                                # |Δd| = 0.8 > 0.5 → changed
    r = compare_row("s", 2000, base, other, search_range_x=41, reach_max=19.5, base_reach=14.5)
    assert set(r) == set(CMP_COLUMNS)
    assert r["n_valid_base"] == 3 and r["n_valid"] == 3 and r["n_both"] == 2
    assert r["frac_changed"] == 0.5 and r["frac_lost"] == pytest.approx(1 / 3) and r["frac_gained"] == pytest.approx(1 / 3)
    assert r["frac_beyond_base_reach"] == 0.0 and r["max_abs_d"] == 3.0
    with pytest.raises(ValueError):
        compare_row("s", 2000, base, other[:, :, :2], search_range_x=41, reach_max=19.5, base_reach=14.5)


def test_predicted_max_disparity_signs_and_monotonic():
    rows = predicted_max_disparity(PARAMS, np.array([1000.0, 2000.0, 4000.0]), H)
    assert [r["z_mm"] for r in rows] == [1000.0, 2000.0, 4000.0] and set(rows[0]) == set(PRED_COLUMNS)
    for r in rows:
        assert r["y0_pos"] == pytest.approx(PARAMS.v_pos / r["z_mm"] + PARAMS.u_pos)
        assert r["d_top_pos"] == pytest.approx((0 - r["y0_pos"]) / PARAMS.a_pos)
        assert r["d_bottom_neg"] == pytest.approx((H - 1 - r["y0_neg"]) / PARAMS.a_neg)
        assert r["d_max_abs"] == max(abs(r["d_top_pos"]), abs(r["d_bottom_neg"]))
    # 近距離ほど y0 が下（大きい行）→ 上端の pos 視差は大きく、下端の neg 視差は小さい
    assert rows[0]["d_top_pos"] > rows[2]["d_top_pos"] and abs(rows[0]["d_bottom_neg"]) < abs(rows[2]["d_bottom_neg"])


def test_bench_rows(monkeypatch):
    monkeypatch.setattr(common, "calcDisparityByZNCC", lambda red, blue, k, s, t: np.full((H, W), np.nan, np.float32))
    proc = np.random.default_rng(0).integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    rows = bench_search_ranges(proc, [common.DispSettings(), common.DispSettings(search_range_x=41)], n=3)
    assert [r["search_range_x"] for r in rows] == [31, 41] and set(rows[0]) | {"frame"} == set(BENCH_COLUMNS)
    assert all(r["n"] == 3 and r["min_ms"] <= r["mean_ms"] <= r["p95_ms"] + 1e-9 for r in rows)
    assert rows[0]["edge_px"] == rows[1]["edge_px"] > 0


def test_cli_end_to_end(tmp_path, monkeypatch):
    def fake_zncc(red, blue, k, s, t):
        reach = (s - 1) / 2 - 0.5
        d = np.full((H, W), np.nan, dtype=np.float32)
        d[:, :100] = 3.0
        d[:, 100:110] = min(12.0, reach)            # S=21 では 9.5 に張り付く
        d[:, 110:115] = min(20.0, reach)            # S=31 では 14.5 に張り付き、S=51 では 20.0
        return d

    monkeypatch.setattr(common, "calcDisparityByZNCC", fake_zncc)
    cap = tmp_path / "captures" / "dist_2000"
    cap.mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(cap / f"{i:06d}.png"),
                    np.random.default_rng(i).integers(0, 256, size=(720, 1280, 3), dtype=np.uint8))
    params_csv = tmp_path / "fit_result.csv"
    with params_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "value"])
        for k in ("v_pos", "u_pos", "w_pos", "a_pos", "v_neg", "u_neg", "w_neg", "a_neg"):
            w.writerow([k, getattr(PARAMS, k)])
    out = tmp_path / "out"
    rc = main(["--sessions", "dist_2000", "--captures-dir", str(tmp_path / "captures"),
               "--cache-root", str(tmp_path / "cache"), "--out-dir", str(out), "--no-plots",
               "--search-ranges", "21", "51", "--frames-per-session", "2", "--bench-n", "2",
               "--params", str(params_csv), "--z-min", "1000", "--z-max", "2000", "--z-step", "500"])
    assert rc == 0
    with (out / "distribution.csv").open(encoding="utf-8", newline="") as f:
        dist = list(csv.DictReader(f))
    assert len(dist) == 1 and float(dist[0]["max"]) == 14.5 and float(dist[0]["frac_ge_reach"]) == pytest.approx(5 / 115)
    with (out / "search_range_compare.csv").open(encoding="utf-8", newline="") as f:
        cmp_rows = {r["search_range_x"]: r for r in csv.DictReader(f)}
    assert set(cmp_rows) == {"21", "51"} and cmp_rows["21"]["frames"] == "2"
    assert float(cmp_rows["21"]["frac_changed"]) == pytest.approx(15 / 115)      # 12→9.5 と 14.5→9.5 が変化
    assert float(cmp_rows["51"]["frac_changed"]) == pytest.approx(5 / 115)       # 14.5→20 だけ変化
    assert float(cmp_rows["51"]["frac_beyond_base_reach"]) == pytest.approx(5 / 115)
    with (out / "predicted_max_disparity.csv").open(encoding="utf-8", newline="") as f:
        pred = list(csv.DictReader(f))
    assert [float(r["z_mm"]) for r in pred] == [1000.0, 1500.0, 2000.0]
    with (out / "bench.csv").open(encoding="utf-8", newline="") as f:
        bench = list(csv.DictReader(f))
    assert [r["search_range_x"] for r in bench] == ["31", "21", "51"] and bench[0]["frame"] == "dist_2000/000000.png"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["base"]["reach_max"] == 14.5 and summary["compared_search_ranges"] == [21, 51]
    assert main(["--sessions", "dist_2000", "--captures-dir", str(tmp_path / "captures"), "--search-ranges", "20",
                 "--params", str(params_csv)]) == 2                                  # 偶数 S は拒否
