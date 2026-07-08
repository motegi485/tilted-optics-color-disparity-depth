"""calibration/calibrate.py・calibration/fit_params.py の HW 非依存テスト
（合成 numpy 配列・tmp_path・monkeypatch のみ。実校正画像・実 params/ には触れない）。
"""
import sys

import numpy as np
import pandas as pd
import pytest
import cv2

import config
from calibration.calibrate import extract_sign_points
from calibration import calibrate
from calibration import fit_params


LOW, HEIGHT = 0.45, 7.5


# ===== extract_sign_points =====

def test_extract_sign_points_pos_band():
    disp_map = np.array([
        [0.1, 1.0, 10.0],
        [np.nan, 2.0, -1.0],
        [0.5, 0.44, 0.46],
        [7.4, 7.5, 7.6],
    ], dtype=np.float64)
    d_vals, rows = extract_sign_points(disp_map, +1, LOW, HEIGHT)
    pairs = set(zip(rows.tolist(), d_vals.tolist()))
    assert pairs == {(0.0, 1.0), (1.0, 2.0), (2.0, 0.5), (2.0, 0.46), (3.0, 7.4)}


def test_extract_sign_points_neg_band():
    disp_map = np.array([
        [-0.1, -1.0, -10.0],
        [np.nan, -2.0, 1.0],
        [-0.5, -0.44, -0.46],
        [-7.4, -7.5, -7.6],
    ], dtype=np.float64)
    d_vals, rows = extract_sign_points(disp_map, -1, LOW, HEIGHT)
    pairs = set(zip(rows.tolist(), d_vals.tolist()))
    assert pairs == {(0.0, -1.0), (1.0, -2.0), (2.0, -0.5), (2.0, -0.46), (3.0, -7.4)}


def test_extract_sign_points_excludes_nan():
    disp_map = np.array([[np.nan, 5.0], [np.nan, np.nan]], dtype=np.float64)
    d_vals, rows = extract_sign_points(disp_map, +1, LOW, HEIGHT)
    assert d_vals.tolist() == [5.0]
    assert rows.tolist() == [0.0]


# ===== calibrate.py C-3 入力ガード =====

def test_calibrate_rejects_mixed_extensions(tmp_path, monkeypatch):
    (tmp_path / "1400.png").touch()
    (tmp_path / "1600.jpg").touch()
    monkeypatch.setattr(config, "CALIB_IMGS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["calibrate.py"])
    with pytest.raises(ValueError, match="混在"):
        calibrate.main()


def test_calibrate_rejects_duplicate_distance(tmp_path, monkeypatch):
    # 同一拡張子グループ(.jpg/.jpeg)内での重複にして、規格混在チェックとは独立に検証する
    (tmp_path / "1400.jpg").touch()
    (tmp_path / "1400.jpeg").touch()
    monkeypatch.setattr(config, "CALIB_IMGS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["calibrate.py"])
    with pytest.raises(ValueError, match="重複"):
        calibrate.main()


def test_calibrate_rejects_bad_aspect_ratio(tmp_path, monkeypatch):
    bad = np.zeros((720, 1280, 3), dtype=np.uint8)   # 16:9。3:2から大きく逸脱
    cv2.imwrite(str(tmp_path / "9999.png"), bad)
    monkeypatch.setattr(config, "CALIB_IMGS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["calibrate.py"])
    with pytest.raises(ValueError, match="アスペクト比"):
        calibrate.main()


# ===== fit_params.py C-2 isfinite フィルタ =====

def test_filter_finite_keeps_negative_intercept_drops_nan():
    df = pd.DataFrame({"zero_disp_pos": [238.5, -50.0, np.nan, 5.0]})
    out = fit_params._filter_finite(df, "zero_disp_pos")
    assert out["zero_disp_pos"].tolist() == [238.5, -50.0, 5.0]
