"""checks/verify_disparity_rust.py::_passes の HW 非依存テスト（結果 dict を直接組み立てるのみ。
_rt_disparity_native は import 時にモジュール直下で参照されるため、Rust 拡張未導入の環境では
このファイルの import 自体が失敗する。core/disparity.py と同様に導入済みが前提・docs/verification-and-tools.md）。
"""
from checks.verify_disparity_rust import MAX_DELTA_TOL, OVER_FINE_RATE_MAX, _passes


def _good_result(**overrides) -> dict:
    """_passes の全基準を満たす結果 dict（各テストで1項目だけ崩す土台）。"""
    r = {
        "nan_match": True,
        "n_nan_mismatch": 0,
        "max_delta": 0.0,
        "n_over_fine": 0,
        "over_fine_rate": 0.0,
        "determinism_match": True,
        "n_valid_py": 100,
        "dtype_ok": True,
        "writable_ok": True,
    }
    r.update(overrides)
    return r


def test_passes_false_when_over_fine_rate_exceeds_max_even_within_main_criterion():
    """副基準: max_delta は主基準内でも over_fine_rate が上限超なら FAIL。"""
    r = _good_result(max_delta=MAX_DELTA_TOL / 2, over_fine_rate=OVER_FINE_RATE_MAX * 2)
    assert _passes(r, name="some_fixture") is False


def test_passes_false_when_max_delta_exceeds_main_tolerance():
    """主基準: max_delta が許容超なら FAIL。"""
    r = _good_result(max_delta=MAX_DELTA_TOL * 2)
    assert _passes(r, name="some_fixture") is False


def test_passes_false_for_known_shift_with_zero_valid_pixels():
    """既知シフトは両実装が揃って全 NaN でも合格し得るため、n_valid_py>0 を独立に要求する。"""
    r = _good_result(n_valid_py=0)
    assert _passes(r, name="known_shift_1") is False


def test_passes_true_for_fully_flat_both_with_zero_valid_pixels():
    """全面平坦は全 NaN が正解の縮退 fixture なので、n_valid_py==0 でも他基準を満たせば合格。"""
    r = _good_result(n_valid_py=0)
    assert _passes(r, name="fully_flat_both") is True
