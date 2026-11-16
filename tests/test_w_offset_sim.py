"""tools/w_offset_sim.py の HW 非依存テスト（合成係数・tmp_path のみ）。"""
import numpy as np
import pytest

from tools.w_offset_sim import main, simulate

# 実機の live 校正で得た pos 係数に近い値。
# 係数の全桁一致は目的ではなく、「校正範囲で δ≈0.1 m の影響が px・mm でどの程度か」の桁を固定する。
V_LIVE = 4.228e5
U_LIVE = -63.81
Z_LIVE = np.arange(1400, 3201, 200, dtype=float)


def test_delta_zero_is_exact():
    """δ=0 なら真のモデルと w=0 モデルは同形なので残差・深度誤差ともゼロ。"""
    res = simulate(Z_LIVE, V_LIVE, U_LIVE, 0.0)
    assert res["resid_max_px"] == pytest.approx(0.0, abs=1e-6)
    assert res["err_max_mm"] == pytest.approx(0.0, abs=1e-3)
    assert res["v_fit"] == pytest.approx(V_LIVE, rel=1e-6)
    assert res["u_fit"] == pytest.approx(U_LIVE, abs=1e-6)


def test_endpoints_match_live_curve():
    """真のモデルは校正範囲の両端で現行 live 曲線に一致させる（合焦面位置を動かさない前提）。"""
    res = simulate(Z_LIVE, V_LIVE, U_LIVE, 122.0)
    y_live = V_LIVE / Z_LIVE + U_LIVE
    assert res["y_true"][0] == pytest.approx(y_live[0])
    assert res["y_true"][-1] == pytest.approx(y_live[-1])


def test_delta_of_lens_sensor_offset_stays_small_in_calibrated_range():
    """δ=122 mm（関連論文のセンサ–レンズ距離の目安）で、残差は 2 px 未満・深度誤差は 30 mm 未満に
    収まる（assert の範囲がその桁。桁が変わったらモデルの前提を見直す）。"""
    res = simulate(Z_LIVE, V_LIVE, U_LIVE, 122.0)
    assert 1.0 < res["resid_max_px"] < 2.0
    assert 20.0 < res["err_max_mm"] < 30.0
    assert res["err_rms_mm"] < res["err_max_mm"]
    # 誤差は遠端（3200 mm）で最大になる（形の不一致が 1/z の曲率差として遠方に出る）
    assert int(np.argmax(np.abs(res["err_mm"]))) == Z_LIVE.size - 1


def test_error_grows_with_delta():
    r110 = simulate(Z_LIVE, V_LIVE, U_LIVE, 110.0)
    r122 = simulate(Z_LIVE, V_LIVE, U_LIVE, 122.0)
    assert r122["resid_max_px"] > r110["resid_max_px"] > 0.0
    assert r122["err_max_mm"] > r110["err_max_mm"] > 0.0


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        simulate(np.array([1400.0, 1600.0]), V_LIVE, U_LIVE, 0.0)          # 3 点未満
    with pytest.raises(ValueError):
        simulate(np.array([100.0, 200.0, 300.0]), V_LIVE, U_LIVE, 150.0)   # z <= delta
    with pytest.raises(ValueError):
        simulate(Z_LIVE, V_LIVE, U_LIVE, -1.0)                              # 負の delta
    with pytest.raises(ValueError):
        simulate(Z_LIVE, float("nan"), U_LIVE, 0.0)                         # 非有限の係数
    with pytest.raises(ValueError):
        simulate(np.array([1400.0, 1400.0, 1600.0]), V_LIVE, U_LIVE, 0.0)  # 重複距離


def test_main_reads_params_csv_and_prints(tmp_path, capsys):
    p = tmp_path / "fit_result.csv"
    p.write_text(
        "param,value\n"
        f"v_pos,{V_LIVE}\nu_pos,{U_LIVE}\nw_pos,0.0\na_pos,-27.37\n"
        "v_neg,427144.2\nu_neg,-33.90\nw_neg,0.0\na_neg,-22.92\n",
        encoding="utf-8",
    )
    rc = main(["--params", str(p), "--delta", "0", "122", "--sign", "pos"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[pos] delta=   0.0 mm" in out
    assert "[pos] delta= 122.0 mm" in out
    assert "[neg]" not in out
