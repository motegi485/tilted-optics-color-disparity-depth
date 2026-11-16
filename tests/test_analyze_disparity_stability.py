"""tools/analyze_disparity_stability.py の HW 非依存テスト（合成スタック・tmp_path・monkeypatch）。"""
import csv
import json
import math

import cv2
import numpy as np
import pytest

import config
import tools.analysis_common as common
from tools.analyze_disparity_stability import (
    DISP_COLUMNS,
    EDGE_COLUMNS,
    bin_by_disparity,
    bin_by_edge,
    main,
    pixel_stats,
    reference_thresholds,
)

W, H = config.IMAGE_SIZE
NAN = np.nan


def test_pixel_stats_known_values():
    stack = np.array([
        [[1.0, 0.3, NAN], [2.0, 0.5, -1.0]],
        [[1.2, -0.3, NAN], [NAN, -0.5, -1.0]],
        [[0.8, 0.2, NAN], [NAN, NAN, -1.0]],
        [[1.0, NAN, NAN], [NAN, NAN, -1.0]],
    ], dtype=np.float32)
    st = pixel_stats(stack)
    assert st["n_valid"].tolist() == [[4, 3, 0], [1, 2, 4]]
    assert st["mu"][0, 0] == pytest.approx(1.0) and st["mu"][0, 1] == pytest.approx(0.2 / 3, abs=1e-6)
    assert math.isnan(st["mu"][0, 2])
    assert st["sigma"][0, 0] == pytest.approx(math.sqrt(0.08 / 3), abs=1e-6)
    assert math.isnan(st["sigma"][1, 0]) and st["sigma"][1, 2] == 0.0      # n_valid<2 は NaN・一定値は 0
    assert st["flip_rate"][0, 0] == 0.0 and st["flip_rate"][0, 1] == pytest.approx(1 / 3)
    assert math.isnan(st["flip_rate"][0, 2]) and math.isnan(st["flip_rate"][1, 1])   # 無効・多数派なし
    assert st["flip_rate"][1, 0] == 0.0 and st["flip_rate"][1, 2] == 0.0
    assert st["valid_frac"].tolist() == [[1.0, 0.75, 0.0], [0.25, 0.5, 1.0]]
    with pytest.raises(ValueError):
        pixel_stats(stack[0])


def test_binning_and_reference_thresholds():
    rng = np.random.default_rng(1)
    n, h, w = 20, 10, 40
    # 列ごとに |μ| を 0.05〜7.95 で変え、σ は |μ|<0.5 で大きく・>6 で再び大きくする
    mu = np.linspace(0.05, 7.95, w)[None, :].repeat(h, axis=0)
    sigma_true = np.where(mu < 0.5, 0.30, np.where(mu > 6.0, 0.25, 0.08))
    stack = mu[None] + rng.normal(0, 1, size=(n, h, w)) * sigma_true[None]
    stack = stack.astype(np.float32)
    stack[:, 0, :] = np.nan                                              # 1 行は無効（有効率 0）
    st = pixel_stats(stack)
    edge = np.linspace(0, 250, w)[None, :].repeat(h, axis=0)
    rows = bin_by_disparity("s", st, edge, disp_bin=0.5, disp_max=15.0, min_valid_frac=0.5)
    assert [r["bin_lo"] for r in rows[:3]] == [0.0, 0.5, 1.0] and len(rows) == 30
    assert sum(r["n_px"] for r in rows) == (h - 1) * w                   # 無効行は除外
    assert set(rows[0]) == set(DISP_COLUMNS)
    first = rows[0]
    assert first["sigma_median"] > rows[4]["sigma_median"]               # |μ|<0.5 は σ が大きい
    assert rows[-1]["n_px"] == 0 and math.isnan(rows[-1]["sigma_median"])
    erows = bin_by_edge("s", st, edge, edge_bin=50.0, min_valid_frac=0.5)
    assert [r["bin_lo"] for r in erows] == [0.0, 50.0, 100.0, 150.0, 200.0, 250.0]
    assert set(erows[0]) == set(EDGE_COLUMNS) and sum(r["n_px"] for r in erows) == (h - 1) * w
    ref = reference_thresholds(rows, flip_ref=0.05, sigma_ratio=2.0, min_px=5)
    assert ref["sigma_floor"] < 0.15 and 0.5 <= ref["sigma_floor_bin_lo"] < 6.0
    assert ref["sigma_exceeds_ratio_from_px"] is not None and ref["sigma_exceeds_ratio_from_px"] >= 6.0
    assert ref["flip_below_ref_from_px"] is not None
    assert reference_thresholds([], flip_ref=0.05) == {"flip_ref": 0.05, "sigma_ratio": 2.0, "min_px": 200}


def test_cli_end_to_end(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_zncc(red, blue, k, s, t):
        calls["n"] += 1
        rng = np.random.default_rng(calls["n"])
        d = np.full((H, W), np.nan, dtype=np.float32)
        d[:, :100] = 2.0 + rng.normal(0, 0.1, size=(H, 100))
        d[:, 100:150] = 0.1 * rng.normal(0, 1, size=(H, 50))            # 符号が揺れる小視差
        return d

    monkeypatch.setattr(common, "calcDisparityByZNCC", fake_zncc)
    cap = tmp_path / "captures" / "dist_2000"
    cap.mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(cap / f"{i:06d}.png"),
                    np.random.default_rng(i).integers(0, 256, size=(720, 1280, 3), dtype=np.uint8))
    out = tmp_path / "out"
    rc = main(["--sessions", "dist_2000", "--captures-dir", str(tmp_path / "captures"),
               "--cache-root", str(tmp_path / "cache"), "--out-dir", str(out), "--no-plots",
               "--disp-bin", "0.5", "--min-valid-frac", "0.5"])
    assert rc == 0
    with (out / "by_disparity.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["session"] for r in rows} == {"all", "dist_2000"}
    assert list(rows[0].keys()) == DISP_COLUMNS
    all_rows = [r for r in rows if r["session"] == "all"]
    assert int(all_rows[0]["n_px"]) == H * 50                                              # |d| < 0.5
    assert int(all_rows[3]["n_px"]) + int(all_rows[4]["n_px"]) == H * 100                 # μ≈2.0 は 1.5–2.5 に収まる
    assert float(all_rows[0]["flip_rate_mean"]) > 0.1 and float(all_rows[4]["flip_rate_mean"]) == 0.0
    assert (out / "by_edge.csv").exists()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["frames_used"] == {"dist_2000": 3} and summary["settings"]["key"] == common.DispSettings().key()
    cached = common.cache_root(common.DispSettings(), tmp_path / "cache") / "dist_2000" / "000000.npy"
    assert cached.exists() and cached.with_suffix(".json").exists()      # キャッシュは入力画像を識別するメタ json とセットで置かれる
    assert main(["--sessions", "dist_9999", "--captures-dir", str(tmp_path / "captures"),
                 "--cache-root", str(tmp_path / "cache"), "--no-plots"]) == 2
    assert main(["--sessions", "dist_2000", "--min-valid-frac", "0", "--captures-dir", str(tmp_path / "captures")]) == 2
