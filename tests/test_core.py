"""core/disparity.py・core/depth.py・core/params.py の HW 非依存テスト（合成 numpy 配列のみ）。"""
import numpy as np
import pytest

from core.disparity import calcDisparityByZNCC
from core.depth import compute_depth
from core.params import DepthParams


K, S, TH = 17, 31, 30


# ===== calcDisparityByZNCC: 入力検証 =====

def test_calc_disparity_even_kernel_raises():
    img = np.zeros((40, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        calcDisparityByZNCC(img, img, 16, S, TH)


def test_calc_disparity_even_search_raises():
    img = np.zeros((40, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        calcDisparityByZNCC(img, img, K, 30, TH)


def test_calc_disparity_non_2d_raises():
    img3d = np.zeros((40, 100, 3), dtype=np.uint8)
    img2d = np.zeros((40, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        calcDisparityByZNCC(img3d, img2d, K, S, TH)


# ===== calcDisparityByZNCC: 既知シフトでの符号・値の再現 =====

def _make_pattern(width: int, height: int, edge_col: int) -> np.ndarray:
    """縦エッジ1本＋微小ノイズ(±3)を持つ列パターンを全行に複製する。
    ノイズは EDGE_TH=30 未満に抑え、エッジ以外を ZNCC タイにしない(C-1)ためのもの。
    """
    rng = np.random.default_rng(42)
    base = 100 + rng.integers(-3, 4, size=width)
    base = base.astype(np.int32)
    base[edge_col:] += 150
    pattern = np.clip(base, 0, 255).astype(np.uint8)
    return np.broadcast_to(pattern, (height, width)).copy()


@pytest.mark.parametrize("shift", [1, -1, 3, -3, 7, -7])
def test_calc_disparity_known_shift(shift):
    """blue(c) = red(c+shift) の関係を作ると d ≈ shift になる(§2.7 の視差定義どおり)。"""
    width, height, edge_col = 140, 40, 70
    red = _make_pattern(width, height, edge_col)
    blue = _make_pattern(width, height, edge_col - shift)

    disp = calcDisparityByZNCC(red, blue, K, S, TH)
    valid = ~np.isnan(disp)
    assert int(valid.sum()) > 0, "有効画素が0（パターン設計を見直すこと）"
    vals = disp[valid]
    assert np.all(np.abs(vals - shift) < 0.1), f"d が shift={shift} から乖離: {vals}"


# ===== calcDisparityByZNCC: C-1 タイ・プラトー → NaN =====

@pytest.mark.parametrize("blue_const", [128, 255])
def test_calc_disparity_flat_blue_becomes_all_nan(blue_const):
    """blue が全面定数（平坦・飽和）だと全シフト同値のタイになり、有効画素は0になる(C-1)。"""
    height, width = 60, 100
    red = np.full((height, width), 50, dtype=np.uint8)
    red[:, 50:] = 200   # 縦エッジ1本
    blue = np.full((height, width), blue_const, dtype=np.uint8)

    disp = calcDisparityByZNCC(red, blue, K, S, TH)
    assert int(np.count_nonzero(~np.isnan(disp))) == 0


# ===== compute_depth =====

def _params(v_pos=400000.0, u_pos=50.0, a_pos=-30.0,
            v_neg=400000.0, u_neg=50.0, a_neg=-30.0) -> DepthParams:
    return DepthParams(v_pos=v_pos, u_pos=u_pos, a_pos=a_pos, w_pos=0.0,
                       v_neg=v_neg, u_neg=u_neg, a_neg=a_neg, w_neg=0.0)


def _compute(disp: np.ndarray, p: DepthParams, eps_den: float = 1.0) -> np.ndarray:
    return compute_depth(disp, p.v_pos, p.u_pos, p.a_pos, p.w_pos,
                         p.v_neg, p.u_neg, p.a_neg, p.w_neg, eps_den)


def test_compute_depth_nan_propagates():
    disp = np.array([[np.nan, 5.0]], dtype=np.float32)
    depth = _compute(disp, _params())
    assert np.isnan(depth[0, 0])


def test_compute_depth_d_zero_is_nan():
    disp = np.array([[0.0, 5.0]], dtype=np.float32)
    depth = _compute(disp, _params())
    assert np.isnan(depth[0, 0])
    assert not np.isnan(depth[0, 1])


def test_compute_depth_eps_den_guard():
    # y=0 の行で pos: den = y - a*d - u = 0 - (-30)*d - 50。d=50/30 のとき den=0 に近づく。
    params = _params(a_pos=-30.0, u_pos=50.0)
    d_zero_den = 50.0 / 30.0
    disp = np.array([[d_zero_den]], dtype=np.float32)  # 1行1列 -> y=0
    depth = _compute(disp, params)
    assert np.isnan(depth[0, 0]), "分母がほぼ0の画素は NaN になるべき"


def test_compute_depth_pos_neg_branch_selection():
    """d>0 は pos 係数、d<0 は neg 係数を使う（v を変えて分岐を判別）。"""
    params = _params(v_pos=100000.0, u_pos=0.0, a_pos=0.0,
                     v_neg=999999.0, u_neg=0.0, a_neg=0.0)
    # y=10 の行に d=+2（pos）と d=-2（neg）を置く。a=0 なので分母は y-u=10 で共通。
    disp = np.full((11, 2), np.nan, dtype=np.float32)
    disp[10, 0] = 2.0
    disp[10, 1] = -2.0
    depth = _compute(disp, params)
    expected_pos = 100000.0 / 10.0
    expected_neg = 999999.0 / 10.0
    assert depth[10, 0] == pytest.approx(expected_pos, rel=1e-5)
    assert depth[10, 1] == pytest.approx(expected_neg, rel=1e-5)


# ===== DepthParams.from_dict =====

def test_depthparams_from_dict_missing_key_raises():
    mapping = {"v_pos": 1, "u_pos": 1, "a_pos": 1, "w_pos": 0,
               "v_neg": 1, "u_neg": 1, "a_neg": 1}  # w_neg 欠損
    with pytest.raises(KeyError):
        DepthParams.from_dict(mapping)


def test_depthparams_from_dict_ok():
    mapping = {"v_pos": "1.5", "u_pos": "2.5", "a_pos": "3.5", "w_pos": "0",
               "v_neg": "4.5", "u_neg": "5.5", "a_neg": "6.5", "w_neg": "0"}
    params = DepthParams.from_dict(mapping)
    assert params.v_pos == 1.5
    assert params.w_neg == 0.0
    assert isinstance(params.v_pos, float)
