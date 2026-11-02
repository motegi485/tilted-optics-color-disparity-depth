"""core/disparity.py・core/depth.py・core/params.py の HW 非依存テスト（合成 numpy 配列のみ）。

calcDisparityByZNCC の挙動テストは core.disparity（Rust 実装への薄いラッパー）と core._disparity_reference（凍結 Python 参照実装）の両方でパラメトライズする。
Rust↔Python の数値照合そのものは checks/verify_disparity_rust.py が広いコーパスで担うが、
ここでの二重化は「どちらの実装も契約（ValueError・既知シフト・タイ→NaN）を満たす」ことを
pytest の通常回帰網の中で恒久的に守るためのもの（2026-07 のコードレビュー指摘）。
"""
import numpy as np
import pytest

from core.disparity import calcDisparityByZNCC as calc_rust
from core._disparity_reference import calcDisparityByZNCC as calc_python
from core.depth import compute_depth
from core.params import DepthParams, load_params


K, S, TH = 17, 31, 30

_IMPLS = [
    pytest.param(calc_rust, id="rust"),
    pytest.param(calc_python, id="python"),
]


# ===== calcDisparityByZNCC: 入力検証 =====

@pytest.mark.parametrize("calc_fn", _IMPLS)
def test_calc_disparity_even_kernel_raises(calc_fn):
    img = np.zeros((40, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        calc_fn(img, img, 16, S, TH)


@pytest.mark.parametrize("calc_fn", _IMPLS)
def test_calc_disparity_even_search_raises(calc_fn):
    img = np.zeros((40, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        calc_fn(img, img, K, 30, TH)


@pytest.mark.parametrize("calc_fn", _IMPLS)
def test_calc_disparity_non_2d_raises(calc_fn):
    img3d = np.zeros((40, 100, 3), dtype=np.uint8)
    img2d = np.zeros((40, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        calc_fn(img3d, img2d, K, S, TH)


# ===== calcDisparityByZNCC: ラッパー(core.disparity)のみの入力契約（2026-07-13 レビュー） =====
# 参照実装(core._disparity_reference)は凍結のため対象外。参照実装のこれらの入力への挙動は
# 契約外（float64 は cv2.error、負の kernel_size は負インデックス回り込み等）であり、
# ラッパーで ValueError に統一する（docs/known-issues.md #10。PyO3 の TypeError/OverflowError 漏れの封止）。

def test_calc_disparity_wrapper_non_uint8_raises():
    img32 = np.zeros((40, 100), dtype=np.float32)
    img64 = np.zeros((40, 100), dtype=np.float64)
    for img in (img32, img64):
        with pytest.raises(ValueError):
            calc_rust(img, img, K, S, TH)


def test_calc_disparity_wrapper_negative_params_raise():
    img = np.zeros((40, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        calc_rust(img, img, -3, S, TH)   # -3 % 2 == 1 なので奇数チェックだけでは通過してしまう
    with pytest.raises(ValueError):
        calc_rust(img, img, K, -5, TH)


def test_calc_disparity_wrapper_non_integer_params_raise():
    img = np.zeros((40, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        calc_rust(img, img, 17.0, S, TH)
    with pytest.raises(ValueError):
        calc_rust(img, img, K, 31.0, TH)


def test_calc_disparity_wrapper_empty_image_raises():
    empty = np.zeros((5, 0), dtype=np.uint8)
    with pytest.raises(ValueError):
        calc_rust(empty, empty, 1, 1, TH)


def test_calc_disparity_wrapper_shape_mismatch_raises():
    red = np.zeros((40, 100), dtype=np.uint8)
    blue = np.zeros((40, 90), dtype=np.uint8)
    with pytest.raises(ValueError):
        calc_rust(red, blue, K, S, TH)


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


@pytest.mark.parametrize("calc_fn", _IMPLS)
@pytest.mark.parametrize("shift", [1, -1, 3, -3, 7, -7])
def test_calc_disparity_known_shift(shift, calc_fn):
    """blue(c) = red(c+shift) の関係を作ると d ≈ shift になる(docs/theory.md の視差定義どおり)。"""
    width, height, edge_col = 140, 40, 70
    red = _make_pattern(width, height, edge_col)
    blue = _make_pattern(width, height, edge_col - shift)

    disp = calc_fn(red, blue, K, S, TH)
    valid = ~np.isnan(disp)
    assert int(valid.sum()) > 0, "有効画素が0（パターン設計を見直すこと）"
    vals = disp[valid]
    assert np.all(np.abs(vals - shift) < 0.1), f"d が shift={shift} から乖離: {vals}"


# ===== calcDisparityByZNCC: 入力のメモリ配置（2026-10-30 監査 IMPL-003） =====

def _textured_pair(h: int = 60, w: int = 120, shift: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """行ごとに異なる乱数テクスチャと、それを水平に shift ずらした青（配置を取り違えると結果が変わる入力）。"""
    red = np.random.default_rng(20261030).integers(0, 256, size=(h, w), dtype=np.uint8)
    return red, np.roll(red, shift, axis=1)


def _non_c_layouts(img: np.ndarray) -> dict[str, np.ndarray]:
    """同じ論理画素値を持つ C 連続でない配列（F 連続・転置 view・非連続 view）。"""
    big = np.zeros((img.shape[0] * 2, img.shape[1] * 2), dtype=np.uint8)
    big[::2, ::2] = img
    return {"fortran": np.asfortranarray(img),
            "transposed_view": np.ascontiguousarray(img.T).T,
            "strided_view": big[::2, ::2]}


@pytest.mark.parametrize("calc_fn", _IMPLS)
def test_calc_disparity_independent_of_memory_layout(calc_fn):
    """F 連続入力を Rust 側が行優先と誤読して別の視差を返していた（NaN 位置・値とも不一致）。"""
    red, blue = _textured_pair()
    ref = calc_fn(red, blue, K, S, TH)
    assert int(np.count_nonzero(~np.isnan(ref))) > 0
    reds, blues = _non_c_layouts(red), _non_c_layouts(blue)
    for name in reds:
        assert np.array_equal(reds[name], red) and not reds[name].flags.c_contiguous
        got = calc_fn(reds[name], blues[name], K, S, TH)
        assert np.array_equal(np.isnan(got), np.isnan(ref)), name
        assert np.array_equal(got[~np.isnan(got)], ref[~np.isnan(ref)]), name


def test_calc_disparity_wrapper_passes_c_contiguous_input_without_copy(monkeypatch):
    """本番経路の C 連続入力（cv2.split の出力）はそのままネイティブへ渡る。F 連続は C 連続のコピーにして渡す。"""
    import core.disparity as wrapper
    seen = []

    class _FakeNative:
        @staticmethod
        def calc_disparity_by_zncc(red, blue, *args):
            seen.append((red, blue))
            return np.full(red.shape, np.nan, dtype=np.float32)

    monkeypatch.setattr(wrapper, "_native", _FakeNative)
    red, blue = _textured_pair()
    calc_rust(red, blue, K, S, TH)
    assert seen[-1][0] is red and seen[-1][1] is blue
    calc_rust(np.asfortranarray(red), np.asfortranarray(blue), K, S, TH)
    assert all(a.flags.c_contiguous for a in seen[-1])
    assert np.array_equal(seen[-1][0], red) and np.array_equal(seen[-1][1], blue)


# ===== calcDisparityByZNCC: C-1 タイ・プラトー → NaN =====

@pytest.mark.parametrize("calc_fn", _IMPLS)
@pytest.mark.parametrize("blue_const", [128, 255])
def test_calc_disparity_flat_blue_becomes_all_nan(blue_const, calc_fn):
    """blue が全面定数（平坦・飽和）だと全シフト同値のタイになり、有効画素は0になる(C-1)。"""
    height, width = 60, 100
    red = np.full((height, width), 50, dtype=np.uint8)
    red[:, 50:] = 200   # 縦エッジ1本
    blue = np.full((height, width), blue_const, dtype=np.uint8)

    disp = calc_fn(red, blue, K, S, TH)
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


# ===== DepthParams.from_dict: 係数の妥当性検証（2026-09 レビューで追加されたガード） =====

@pytest.mark.parametrize("bad_key,bad_value", [
    ("v_pos", "nan"), ("u_pos", "inf"), ("a_pos", "-inf"),
    ("v_neg", "nan"), ("a_neg", "inf"),
])
def test_depthparams_from_dict_non_finite_coefficient_raises(bad_key, bad_value):
    mapping = {"v_pos": "1", "u_pos": "1", "a_pos": "1", "w_pos": "0",
               "v_neg": "1", "u_neg": "1", "a_neg": "1", "w_neg": "0"}
    mapping[bad_key] = bad_value
    with pytest.raises(ValueError):
        DepthParams.from_dict(mapping)


@pytest.mark.parametrize("bad_key", ["w_pos", "w_neg"])
def test_depthparams_from_dict_nonzero_w_raises(bad_key):
    mapping = {"v_pos": "1", "u_pos": "1", "a_pos": "1", "w_pos": "0",
               "v_neg": "1", "u_neg": "1", "a_neg": "1", "w_neg": "0"}
    mapping[bad_key] = "1.0"
    with pytest.raises(ValueError):
        DepthParams.from_dict(mapping)


def test_depthparams_from_dict_all_finite_w_zero_succeeds_as_float():
    mapping = {"v_pos": "1.0", "u_pos": "2.0", "a_pos": "3.0", "w_pos": "0.0",
               "v_neg": "4.0", "u_neg": "5.0", "a_neg": "6.0", "w_neg": "0.0"}
    params = DepthParams.from_dict(mapping)
    for value in (params.v_pos, params.u_pos, params.a_pos, params.w_pos,
                  params.v_neg, params.u_neg, params.a_neg, params.w_neg):
        assert isinstance(value, float)


# ===== load_params（tmp_path に自作 CSV。実 params/ ファイルには触れない） =====

def _write_fit_csv(path, overrides=None) -> None:
    values = {"v_pos": "1.0", "u_pos": "2.0", "a_pos": "3.0", "w_pos": "0.0",
              "v_neg": "4.0", "u_neg": "5.0", "a_neg": "6.0", "w_neg": "0.0"}
    if overrides:
        values.update(overrides)
    lines = ["param,value"] + [f"{k},{v}" for k, v in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_params_duplicate_param_raises(tmp_path):
    path = tmp_path / "fit_result.csv"
    lines = ["param,value", "v_pos,1.0", "v_pos,2.0", "u_pos,1.0", "a_pos,1.0", "w_pos,0.0",
             "v_neg,1.0", "u_neg,1.0", "a_neg,1.0", "w_neg,0.0"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_params(path)


def test_load_params_ok(tmp_path):
    path = tmp_path / "fit_result.csv"
    _write_fit_csv(path)
    params = load_params(path)
    assert params.v_pos == 1.0
    assert params.w_neg == 0.0
