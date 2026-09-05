"""realtime/pipeline.py::process_frame の HW 非依存テスト（合成 numpy 配列のみ）。"""
import numpy as np
import pytest

import config
from core.params import DepthParams
from realtime.pipeline import process_frame


_H, _W = config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]   # (H,W)=(352,528)。数値は config から導く


def _params() -> DepthParams:
    return DepthParams(v_pos=400000.0, u_pos=50.0, a_pos=-30.0, w_pos=0.0,
                       v_neg=400000.0, u_neg=50.0, a_neg=-30.0, w_neg=0.0)


# ===== process_frame: 形状・dtype の入力検証（2026-09 レビュー） =====

def test_process_frame_wrong_height_raises():
    bad = np.zeros((_H + 8, _W, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        process_frame(bad, _params())


def test_process_frame_missing_channel_axis_raises():
    bad = np.zeros((_H, _W), dtype=np.uint8)
    with pytest.raises(ValueError):
        process_frame(bad, _params())


def test_process_frame_wrong_channel_count_raises():
    bad = np.zeros((_H, _W, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        process_frame(bad, _params())


def test_process_frame_wrong_dtype_raises():
    bad = np.zeros((_H, _W, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        process_frame(bad, _params())


def test_process_frame_non_ndarray_raises():
    bad = np.zeros((_H, _W, 3), dtype=np.uint8).tolist()   # list（np.ndarray でない）
    with pytest.raises(ValueError):
        process_frame(bad, _params())


# ===== process_frame: 正しい入力での成功系 =====

def test_process_frame_valid_input_returns_expected_shape_and_dtype():
    proc_bgr = np.zeros((_H, _W, 3), dtype=np.uint8)
    depth = process_frame(proc_bgr, _params())
    assert depth.shape == (_H, _W)
    assert depth.dtype == np.float32
