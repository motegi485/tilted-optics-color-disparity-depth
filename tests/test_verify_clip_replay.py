"""checks/verify_clip_replay.py::_compare の HW 非依存テスト（合成 numpy 配列のみ）。"""
import numpy as np

from checks.verify_clip_replay import _compare


def test_compare_dtype_mismatch_returns_false():
    """live が float64、recomputed が float32 で数値が同一でも dtype 不一致で FAIL 扱い
    （canonical な depth は float32 のため・2026-09 レビュー）。"""
    live = np.array([[1.0, 2.0]], dtype=np.float64)
    recomputed = live.astype(np.float32)

    ok, msg = _compare(live, recomputed)

    assert ok is False
    assert "dtype" in msg


def test_compare_exact_match_returns_true():
    live = np.array([[1.0, np.nan, 3.0]], dtype=np.float32)
    recomputed = live.copy()

    ok, _msg = _compare(live, recomputed)

    assert ok is True


def test_compare_both_all_nan_returns_false():
    """両方全 NaN は空集合同士の比較で match してしまう穴を無条件 FAIL で塞ぐガード。"""
    live = np.full((2, 2), np.nan, dtype=np.float32)
    recomputed = np.full((2, 2), np.nan, dtype=np.float32)

    ok, _msg = _compare(live, recomputed)

    assert ok is False
