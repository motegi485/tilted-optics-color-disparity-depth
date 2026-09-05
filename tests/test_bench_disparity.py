"""checks/bench_disparity.py::main の HW 非依存テスト（synthetic 入力のみ・data/ 非依存）。"""
from checks.bench_disparity import main


def test_main_returns_2_for_zero_n():
    assert main(["--n", "0"]) == 2


def test_main_returns_2_for_negative_n():
    assert main(["--n", "-3"]) == 2


def test_main_returns_0_for_synthetic_native_within_budget():
    """synthetic 入力・native 実装は実測 p95~9ms（2026-09 実測）で 33ms バジェットに余裕があり、
    通常環境では PASS(0) を期待する。"""
    assert main(["--source", "synthetic", "--n", "1", "--impl", "native"]) == 0
