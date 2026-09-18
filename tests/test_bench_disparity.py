"""checks/bench_disparity.py::main の HW 非依存テスト（synthetic 入力のみ・data/ 非依存）。"""
import checks.bench_disparity as mod
from checks.bench_disparity import main


def test_main_returns_2_for_zero_n():
    assert main(["--n", "0"]) == 2


def test_main_returns_2_for_negative_n():
    assert main(["--n", "-3"]) == 2


def _fixed_bench(p95_ms: float):
    """_bench の差し替え: 関数は 1 回だけ実行し（disp_holder を埋めるため）、計測値は固定で返す。"""
    def fake(fn, n):
        fn()
        return p95_ms, p95_ms, p95_ms
    return fake


def test_main_exit_code_follows_budget_verdict(monkeypatch):
    """exit code は 30fps バジェット判定（p95 < 33.33 ms → 0・それ以外 → 1）に従う（計測を固定値に差し替えた HW 非依存の契約テスト）。

    2026-09-18 の設定変更（K25/S41/T20）後、synthetic 入力（E≈40k・ノイズ板全画面に近い最悪ケース）は実測で
    予算超（p95 ≈ 58 ms・記録票 2026-09-18 §8）になった。30fps の基準は典型シーン（scene_01・§7）なので、
    実測値そのものはここでは検証せず（旧テストは synthetic で PASS(0) を前提にしていた）、判定と終了コードの接続だけを見る。
    """
    monkeypatch.setattr(mod, "_bench", _fixed_bench(10.0))
    assert main(["--source", "synthetic", "--n", "1", "--impl", "native"]) == 0
    monkeypatch.setattr(mod, "_bench", _fixed_bench(40.0))
    assert main(["--source", "synthetic", "--n", "1", "--impl", "native"]) == 1
