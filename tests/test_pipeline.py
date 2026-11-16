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


# ===== process_frame: 形状・dtype の入力検証 =====

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


# ===== process_frame: 視差ゲート（視差上限ゲート＋推論帯）=====

def test_process_frame_applies_gate1_and_inference_band(monkeypatch):
    """ZNCC を既知の視差を返す fake に差し替え、視差上限ゲート（|d| > DISP_MAX）と推論帯
    （|d| <= INFER_BAND_LOW または |d| >= INFER_BAND_HIGH）の外側だけが NaN になることを確認する。
    境界値は両方とも「落ちる」側（校正帯と同じ厳密不等号）。"""
    import realtime.pipeline as mod

    low, high, dmax = config.INFER_BAND_LOW, config.INFER_BAND_HIGH, config.DISP_MAX
    cases = {   # 行 → (視差, 有効であるべきか)
        10: (0.0, False),                    # d == 0 は compute_depth が NaN にする
        20: (low, False),                    # 境界 |d| == LOW → 落ちる
        30: (-low, False),
        40: (low + 0.05, True),              # 帯内（pos）
        50: (-(low + 0.05), True),           # 帯内（neg）
        60: ((low + high) / 2, True),        # 帯の中央
        70: (high - 0.05, True),
        80: (high, False),                   # 境界 |d| == HIGH → 落ちる
        90: (-high, False),
        100: (high + 1.0, False),            # 帯の上側（視差上限ゲートは通るが推論帯で落ちる）
        110: (dmax, False),                  # 視差上限ゲートの境界は通るが推論帯で落ちる（HIGH < DISP_MAX）
        120: (dmax + 1.0, False),            # 視差上限ゲートで落ちる
        130: (-(dmax + 1.0), False),
    }
    assert high < dmax                       # 「視差上限ゲートは推論帯に吸収される」の前提

    def fake_zncc(red, blue, kernel_size, search_range_x, th_value):
        assert (kernel_size, search_range_x, th_value) == (config.KERNEL_SIZE, config.SEARCH_RANGE_X, config.EDGE_TH)
        d = np.full((_H, _W), np.nan, dtype=np.float32)
        for row, (val, _) in cases.items():
            d[row, 100:110] = val
        return d

    monkeypatch.setattr(mod, "calcDisparityByZNCC", fake_zncc)
    depth = process_frame(np.zeros((_H, _W, 3), dtype=np.uint8), _params())
    for row, (val, should_be_valid) in cases.items():
        got = ~np.isnan(depth[row, 100:110])
        assert got.all() == should_be_valid and got.any() == should_be_valid, f"row {row}: d={val}"
    assert np.isnan(depth[0, 100:110]).all()        # 視差 NaN の画素は NaN のまま
    # 有効画素数 = 帯内のケース数 × 10 画素
    assert int(np.count_nonzero(~np.isnan(depth))) == 10 * sum(1 for _, ok in cases.values() if ok)
