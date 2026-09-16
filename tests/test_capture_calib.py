"""calibration/capture_calib.py の HW 非依存テスト（保存関数のみ・tmp_path・monkeypatch。webcam は使わない）。"""
import cv2
import numpy as np
import pytest

import config
import calibration.capture_calib as cc
from calibration.capture_calib import save_calib_png, save_calib_raw

W, H = config.IMAGE_SIZE
RW, RH = config.WEBCAM_NATIVE_SIZE


def test_save_calib_png_and_raw_write_lossless_png(tmp_path):
    proc = np.random.default_rng(0).integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    raw = np.random.default_rng(1).integers(0, 256, size=(RH, RW, 3), dtype=np.uint8)
    assert save_calib_png(proc, 2000, tmp_path / "imgs", allow_overwrite=False) is True
    assert save_calib_raw(raw, 2000, tmp_path / "raw", allow_overwrite=False) is True
    assert np.array_equal(cv2.imread(str(tmp_path / "imgs" / "2000.png")), proc)      # 可逆
    assert np.array_equal(cv2.imread(str(tmp_path / "raw" / "2000.png")), raw)


def test_save_functions_reject_wrong_shape(tmp_path):
    with pytest.raises(ValueError):
        save_calib_png(np.zeros((RH, RW, 3), dtype=np.uint8), 2000, tmp_path, allow_overwrite=True)   # raw を proc に誤投入
    with pytest.raises(ValueError):
        save_calib_raw(np.zeros((H, W, 3), dtype=np.uint8), 2000, tmp_path, allow_overwrite=True)     # proc を raw に誤投入
    with pytest.raises(ValueError):
        save_calib_raw(np.zeros((RH, RW, 3), dtype=np.float32), 2000, tmp_path, allow_overwrite=True)  # dtype


def test_overwrite_prompt_is_honored(tmp_path, monkeypatch):
    raw = np.zeros((RH, RW, 3), dtype=np.uint8)
    assert save_calib_raw(raw, 1500, tmp_path, allow_overwrite=False) is True
    raw2 = np.full((RH, RW, 3), 7, dtype=np.uint8)
    monkeypatch.setattr(cc, "_ask_yes_no", lambda msg: False)
    assert save_calib_raw(raw2, 1500, tmp_path, allow_overwrite=False) is False        # 否 → 保存しない
    assert np.array_equal(cv2.imread(str(tmp_path / "1500.png")), raw)
    assert save_calib_raw(raw2, 1500, tmp_path, allow_overwrite=True) is True          # 撮り直しは明示上書き
    assert np.array_equal(cv2.imread(str(tmp_path / "1500.png")), raw2)


def test_capture_one_distance_saves_raw_only_when_requested(tmp_path, monkeypatch):
    raw = np.random.default_rng(2).integers(0, 256, size=(RH, RW, 3), dtype=np.uint8)
    monkeypatch.setattr(cc, "capture_burst", lambda source, n: raw)
    monkeypatch.setattr(cc, "_ask_yes_no", lambda msg: False)            # チェックしない・撮り直さない
    monkeypatch.setattr(config, "CALIB_IMGS_DIR", tmp_path / "imgs")
    monkeypatch.setattr(config, "CALIB_RAW_DIR", tmp_path / "raw")
    cc._capture_one_distance(source=None, dist=1800, n=4)
    assert (tmp_path / "imgs" / "1800.png").exists() and not (tmp_path / "raw").exists()
    cc._capture_one_distance(source=None, dist=2000, n=4, save_raw=True)
    assert (tmp_path / "imgs" / "2000.png").exists() and (tmp_path / "raw" / "2000.png").exists()
    assert np.array_equal(cv2.imread(str(tmp_path / "raw" / "2000.png")), raw)
    assert np.array_equal(cv2.imread(str(tmp_path / "imgs" / "2000.png")), cc.preprocess(raw))
