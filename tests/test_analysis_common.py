"""tools/analysis_common.py の HW 非依存テスト（合成画像・合成視差・tmp_path のみ）。"""
import math

import cv2
import numpy as np
import pytest

import config
from core.params import DepthParams
from realtime.camera import preprocess
from realtime.pipeline import process_frame
from tools.analysis_common import (
    CalibResult,
    DispSettings,
    apply_gates,
    calib_image_to_proc,
    calibrate_from_disparity,
    cache_root,
    depth_of,
    disparity_of_proc,
    edge_strength_of_proc,
    ensure_calib_disparity,
    ensure_calib_disparity_from_raw,
    ensure_capture_disparity,
    evaluate_session,
    list_frames,
    load_stack,
    preprocess_to,
    round_to_odd,
    scale_factor,
    scaled_settings,
    summarize_holdout,
)

W, H = config.IMAGE_SIZE
RNG = np.random.default_rng(0)


# ---------- 設定 ----------

def test_settings_defaults_key_and_limits():
    s = DispSettings()
    assert (s.kernel_size, s.search_range_x, s.edge_th) == (config.KERNEL_SIZE, config.SEARCH_RANGE_X, config.EDGE_TH)
    assert s.key() == f"K{config.KERNEL_SIZE}_S{config.SEARCH_RANGE_X}_T{config.EDGE_TH}"
    assert s.disp_max == config.DISP_MAX and s.reach_max == config.DISP_MAX - 0.5   # gate1 上限と到達上限の差 0.5
    assert DispSettings(image_size=(264, 176)).key().endswith("_W264xH176")
    for bad in (dict(kernel_size=16), dict(search_range_x=0), dict(kernel_size=True), dict(edge_th=-1),
                dict(image_size=(0, 10))):
        with pytest.raises(ValueError):
            DispSettings(**bad)


SIZES = [(264, 176), (396, 264), (528, 352), (660, 440), (792, 528), (1056, 704)]   # B3 の 6 水準（3:2）


def test_round_to_odd_and_scale_factor():
    assert [round_to_odd(x) for x in (8.5, 12.75, 17, 21.25, 25.5, 34)] == [9, 13, 17, 21, 25, 35]     # K=17 × r
    assert [round_to_odd(x) for x in (20.5, 30.75, 41, 51.25, 61.5, 82)] == [21, 31, 41, 51, 61, 83]   # S=41 × r
    assert round_to_odd(1.0) == 1 and round_to_odd(0.3) == 1 and round_to_odd(33.99) == 33
    for bad in (0, -3, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            round_to_odd(bad)
    assert [scale_factor(s) for s in SIZES] == [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    assert scale_factor((1056, 704), (528, 352)) == 2.0
    for bad in ((640, 480), (528, 340), (0, 0), (100, -5)):          # 4:3・3:2 から 1 % 超のずれ・非正
        with pytest.raises(ValueError):
            scale_factor(bad)


def test_scaled_settings_scales_kernel_search_and_optionally_edge_th():
    base = DispSettings(kernel_size=17, search_range_x=41, edge_th=30.0)
    scaled = [scaled_settings(base, s) for s in SIZES]
    assert [s.kernel_size for s in scaled] == [9, 13, 17, 21, 25, 35]
    assert [s.search_range_x for s in scaled] == [21, 31, 41, 51, 61, 83]
    assert all(s.edge_th == 30.0 for s in scaled)                      # 既定では T 固定
    assert [tuple(s.image_size) for s in scaled] == SIZES
    assert scaled_settings(base, config.IMAGE_SIZE) == base            # 基準解像度では不変
    big = scaled_settings(base, (1056, 704))
    assert big.key() == "K35_S83_T30_W1056xH704" and big.disp_max == 41.0 and big.reach_max == 40.5
    assert scaled_settings(base, (1056, 704), scale_edge_th=True).edge_th == 15.0
    assert scaled_settings(base, (264, 176), scale_edge_th=True).edge_th == 60.0
    with pytest.raises(ValueError):
        scaled_settings(base, (640, 480))


# ---------- 画像 → 視差 ----------

def _raw() -> np.ndarray:
    return RNG.integers(0, 256, size=(config.WEBCAM_NATIVE_SIZE[1], config.WEBCAM_NATIVE_SIZE[0], 3), dtype=np.uint8)


def test_preprocess_to_matches_camera_preprocess_at_default_size():
    raw = _raw()
    assert np.array_equal(preprocess_to(raw, config.IMAGE_SIZE), preprocess(raw))
    small = preprocess_to(raw, (264, 176))
    assert small.shape == (176, 264, 3)
    with pytest.raises(ValueError):
        preprocess_to(raw[:, :1000], config.IMAGE_SIZE)


def test_calib_image_guard():
    ok = RNG.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    assert calib_image_to_proc(ok, config.IMAGE_SIZE).shape == (H, W, 3)
    full = RNG.integers(0, 256, size=(720, 1080, 3), dtype=np.uint8)     # 3:2 のフル解像度は許可
    assert calib_image_to_proc(full, config.IMAGE_SIZE).shape == (H, W, 3)
    with pytest.raises(ValueError):
        calib_image_to_proc(_raw(), config.IMAGE_SIZE)                   # 16:9 の raw 誤投入は拒否


def test_disparity_of_proc_reproduces_process_frame_after_gates():
    """生視差に gate1＋推論帯（config.INFER_BAND_LOW/HIGH・2026-09-18）を掛けて参照式に通せば
    process_frame とビット一致する（同じ手順の証明。実験の評価器 apply_gates と本番の定義が同じことの根拠）。"""
    proc = preprocess(_raw())
    params = DepthParams(v_pos=426165.15, u_pos=-61.04, a_pos=-26.43, w_pos=0.0,
                         v_neg=411311.86, u_neg=-17.39, a_neg=-22.76, w_neg=0.0)
    disp = disparity_of_proc(proc, DispSettings())
    assert disp.shape == (H, W) and disp.dtype == np.float32
    band = (config.INFER_BAND_LOW, config.INFER_BAND_HIGH)
    depth = depth_of(apply_gates(disp, disp_max=config.DISP_MAX, band=band), params)
    ref = process_frame(proc, params)
    assert np.array_equal(np.isnan(depth), np.isnan(ref))
    assert np.array_equal(depth[~np.isnan(depth)], ref[~np.isnan(ref)])
    # 推論帯なし（gate1 のみ）とは一致しない＝推論帯が実際に効いていることの確認
    depth_g1 = depth_of(apply_gates(disp, disp_max=config.DISP_MAX), params)
    assert np.count_nonzero(~np.isnan(depth_g1)) > np.count_nonzero(~np.isnan(ref))
    # 到達上限: gate1 (|d| > DISP_MAX) は 1 画素も落とさない
    assert np.nanmax(np.abs(disp)) <= DispSettings().reach_max
    with pytest.raises(ValueError):
        disparity_of_proc(proc[:, :, 0], DispSettings())


def test_edge_strength_shape_and_dtype():
    e = edge_strength_of_proc(preprocess(_raw()))
    assert e.shape == (H, W) and e.dtype == np.uint8


# ---------- キャッシュ ----------

def test_capture_and_calib_cache_roundtrip(tmp_path):
    cap = tmp_path / "captures" / "dist_2000"
    cap.mkdir(parents=True)
    for i in range(4):
        cv2.imwrite(str(cap / f"{i:06d}.png"), _raw())
    frames = list_frames(cap, skip_first=1, max_frames=2)
    assert [p.stem for p in frames] == ["000001", "000002"]
    assert [p.stem for p in list_frames(cap, step=2)] == ["000000", "000002"]
    with pytest.raises(ValueError):
        list_frames(cap, skip_first=-1)
    with pytest.raises(FileNotFoundError):
        list_frames(tmp_path / "nope")

    s = DispSettings()
    outs = ensure_capture_disparity(s, "dist_2000", frames, root=tmp_path / "out")
    assert [p.name for p in outs] == ["000001.npy", "000002.npy"]
    assert outs[0].parent == cache_root(s, tmp_path / "out") / "dist_2000"
    mtimes = [p.stat().st_mtime_ns for p in outs]
    assert ensure_capture_disparity(s, "dist_2000", frames, root=tmp_path / "out") == outs
    assert [p.stat().st_mtime_ns for p in outs] == mtimes                # 2 回目はスキップ
    stack = load_stack(outs)
    assert stack.shape == (2, H, W) and stack.dtype == np.float32

    calib = tmp_path / "calib_imgs"
    calib.mkdir()
    for d in (1400, 1600, 1800):
        cv2.imwrite(str(calib / f"{d}.png"), RNG.integers(0, 256, size=(H, W, 3), dtype=np.uint8))
    (calib / "notes.txt").write_text("x", encoding="utf-8")            # 距離名でないファイルは無視
    c = ensure_calib_disparity(s, calib, root=tmp_path / "out")
    assert sorted(c) == [1400, 1600, 1800] and all(p.exists() for p in c.values())
    with pytest.raises(FileNotFoundError):
        ensure_calib_disparity(s, tmp_path / "empty_dir_missing", root=tmp_path / "out")


def test_calib_raw_cache_matches_processed_png_at_default_size(tmp_path):
    """capture_calib --save-raw は raw と preprocess(raw) の両方を可逆 PNG で保存する。既定解像度では raw 経路の視差が
    処理済み PNG 経路（ensure_calib_disparity）とビット一致し、他解像度では preprocess_to で作り直せること。"""
    raw_dir = tmp_path / "calib_raw"
    proc_dir = tmp_path / "calib_imgs"
    raw_dir.mkdir()
    proc_dir.mkdir()
    raws = {d: _raw() for d in (1400, 1600, 1800)}
    for d, raw in raws.items():
        cv2.imwrite(str(raw_dir / f"{d}.png"), raw)
        cv2.imwrite(str(proc_dir / f"{d}.png"), preprocess(raw))          # capture_calib と同じ保存
    (raw_dir / ".gitkeep").write_text("", encoding="utf-8")                # 距離名でないファイルは無視
    s = DispSettings()
    a = ensure_calib_disparity_from_raw(s, raw_dir, root=tmp_path / "out")
    b = ensure_calib_disparity(s, proc_dir, root=tmp_path / "out")
    assert sorted(a) == [1400, 1600, 1800] and a[1400].parent == cache_root(s, tmp_path / "out") / "calib_raw"
    for d in a:
        np.testing.assert_array_equal(np.load(str(a[d])), np.load(str(b[d])))   # NaN の位置も含めて一致
    mtime = a[1400].stat().st_mtime_ns
    sub = ensure_calib_disparity_from_raw(s, raw_dir, distances=[1600, 1400], root=tmp_path / "out")
    assert list(sub) == [1400, 1600] and a[1400].stat().st_mtime_ns == mtime     # サブセット・既存はスキップ
    with pytest.raises(FileNotFoundError):
        ensure_calib_disparity_from_raw(s, raw_dir, distances=[1600, 2000], root=tmp_path / "out")   # raw の無い距離
    with pytest.raises(ValueError):
        ensure_calib_disparity_from_raw(s, raw_dir, distances=[], root=tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        ensure_calib_disparity_from_raw(s, tmp_path / "nope", root=tmp_path / "out")
    small = scaled_settings(DispSettings(), (264, 176))
    c = ensure_calib_disparity_from_raw(small, raw_dir, distances=[1400], root=tmp_path / "out")
    assert np.load(str(c[1400])).shape == (176, 264)
    bad = tmp_path / "bad"
    bad.mkdir()
    cv2.imwrite(str(bad / "1400.png"), preprocess(raws[1400]))            # 528×352 は raw ではない
    with pytest.raises(ValueError):
        ensure_calib_disparity_from_raw(s, bad, root=tmp_path / "out_bad")   # 別キャッシュ（既存 npy の再利用を避ける）


# ---------- キャッシュの入力識別（2026-10-30 監査 REUSE-001） ----------

def test_cache_root_uses_identity_checked_namespace(tmp_path):
    """識別つきの置き場は cache/disp_v2/。入力を識別しない旧 cache/disp/ は読まない。"""
    s = DispSettings()
    assert cache_root(s, tmp_path) == tmp_path / "cache" / "disp_v2" / s.key()
    assert cache_root(s).parent.name == "disp_v2"


def test_calib_cache_rejects_same_name_from_another_campaign(tmp_path):
    """別キャンペーンの同名 2000.png に、先に作った視差を返さない（以前は無警告で返していた）。"""
    s = DispSettings()
    camp_a, camp_b = tmp_path / "campaign_a", tmp_path / "campaign_b"
    for d, seed in ((camp_a, 1), (camp_b, 2)):
        d.mkdir()
        cv2.imwrite(str(d / "2000.png"), np.random.default_rng(seed).integers(0, 256, size=(H, W, 3), dtype=np.uint8))
    root = tmp_path / "out"
    first = ensure_calib_disparity(s, camp_a, root=root)[2000]
    with pytest.raises(ValueError, match="一致しません"):
        ensure_calib_disparity(s, camp_b, root=root)
    assert ensure_calib_disparity(s, camp_a, root=root)[2000] == first      # 同じ入力は再利用
    again = ensure_calib_disparity(s, camp_b, root=root, overwrite=True)[2000]
    expected = disparity_of_proc(calib_image_to_proc(cv2.imread(str(camp_b / "2000.png")), s.image_size), s)
    np.testing.assert_array_equal(np.load(str(again)), expected)          # overwrite を明示すれば計算し直す


def test_capture_cache_detects_replaced_frame_and_missing_identity(tmp_path):
    """同じ dir で中身を差し替えた画像と、識別の無い npy（旧形式の写し等）は使わない。"""
    cap = tmp_path / "captures" / "dist_2000"
    cap.mkdir(parents=True)
    cv2.imwrite(str(cap / "000000.png"), _raw())
    s = DispSettings()
    root = tmp_path / "out"
    frames = list_frames(cap)
    (dst,) = ensure_capture_disparity(s, "dist_2000", frames, root=root)
    cv2.imwrite(str(cap / "000000.png"), _raw())                          # 同じ名前のまま中身だけ差し替え
    with pytest.raises(ValueError, match="一致しません"):
        ensure_capture_disparity(s, "dist_2000", frames, root=root)
    dst.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="識別情報がありません"):
        ensure_capture_disparity(s, "dist_2000", frames, root=root)


# ---------- 校正 ----------

TRUE = DepthParams(v_pos=426000.0, u_pos=-61.0, a_pos=-26.4, w_pos=0.0,
                   v_neg=411000.0, u_neg=-17.4, a_neg=-22.8, w_neg=0.0)
DEPTHS = list(range(1400, 3201, 200))


def _synthetic_disparity(depth: float, *, noise_px: float = 0.05, cols_per_row: int = 40,
                         rng=None) -> np.ndarray:
    """真の参照式から、行ごとに視差を割り当てた合成視差マップ（上部 pos・下部 neg・切片の間は NaN）。"""
    rng = rng or np.random.default_rng(depth)
    d = np.full((H, W), np.nan, dtype=np.float32)
    y0_pos = TRUE.v_pos / depth + TRUE.u_pos
    y0_neg = TRUE.v_neg / depth + TRUE.u_neg
    cols = rng.choice(W, size=cols_per_row, replace=False)
    for y in range(H):
        if y < y0_pos:
            val = (y - y0_pos) / TRUE.a_pos          # > 0
        elif y > y0_neg:
            val = (y - y0_neg) / TRUE.a_neg          # < 0
        else:
            continue
        d[y, cols] = val + rng.normal(0.0, noise_px, size=cols_per_row)
    return d


def test_calibrate_recovers_true_coefficients_and_reports_diagnostics():
    disp_by_depth = {z: _synthetic_disparity(z) for z in DEPTHS}
    res = calibrate_from_disparity(disp_by_depth, low=0.45, high=7.5, ransac_th=8.0)
    assert isinstance(res, CalibResult)
    p = res.params
    assert p.v_pos == pytest.approx(TRUE.v_pos, rel=2e-3) and p.u_pos == pytest.approx(TRUE.u_pos, abs=1.0)
    assert p.v_neg == pytest.approx(TRUE.v_neg, rel=2e-3) and p.u_neg == pytest.approx(TRUE.u_neg, abs=1.0)
    assert p.a_pos == pytest.approx(TRUE.a_pos, abs=0.2) and p.a_neg == pytest.approx(TRUE.a_neg, abs=0.2)
    assert p.w_pos == 0.0 and p.w_neg == 0.0
    assert [r["depth"] for r in res.rows] == DEPTHS
    assert set(res.rows[0]) == {"depth", "zero_disp_pos", "zero_disp_neg", "a_pos", "a_neg",
                                "inlier_pos", "inlier_neg", "n_pos", "n_neg"}
    assert res.n_used == {"pos": 10, "neg": 10} and res.skipped == []
    assert res.fit_rms_px["pos"] < 0.5 and res.fit_rms_px["neg"] < 0.5
    assert res.depth_rmse_mm["combined"] < 10.0
    assert set(res.depth_err_mm["pos"]) == set(DEPTHS)
    # 帯を狭めても係数は大きく変わらない（合成データは帯の内外で同じ直線）
    res2 = calibrate_from_disparity(disp_by_depth, low=1.0, high=5.0, ransac_th=None)
    assert res2.params.v_pos == pytest.approx(p.v_pos, rel=5e-3)


def test_calibrate_skips_sparse_depths_and_validates_inputs():
    disp_by_depth = {z: _synthetic_disparity(z) for z in DEPTHS}
    disp_by_depth[2000] = np.full((H, W), np.nan, dtype=np.float32)      # 点なし → スキップ
    res = calibrate_from_disparity(disp_by_depth, low=0.45, high=7.5)
    assert res.skipped == [2000] and res.n_used == {"pos": 9, "neg": 9}
    with pytest.raises(ValueError):
        calibrate_from_disparity(disp_by_depth, low=7.5, high=0.45)
    with pytest.raises(ValueError):
        calibrate_from_disparity({}, low=0.45, high=7.5)
    with pytest.raises(RuntimeError):                                     # 全距離スキップ
        calibrate_from_disparity({z: np.full((H, W), np.nan, dtype=np.float32) for z in DEPTHS}, low=0.45, high=7.5)
    with pytest.raises(RuntimeError):                                     # 有効 2 距離しかない
        calibrate_from_disparity({z: disp_by_depth[z] for z in (1400, 1600)}, low=0.45, high=7.5)


# ---------- 評価 ----------

def test_apply_gates_and_evaluate_session():
    d = np.full((H, W), np.nan, dtype=np.float32)
    d[100, :10] = 2.0          # 帯内・pos
    d[150, :10] = 0.2          # 帯の下側
    d[200, :10] = -9.0         # 帯の上側
    d[210, :10] = -20.0        # gate1 で落ちる（合成なので到達上限超えも置ける）
    g = apply_gates(d, disp_max=15.0)
    assert np.isnan(g[210, 0]) and g[100, 0] == 2.0 and g[150, 0] == 0.2 and g[200, 0] == -9.0
    g2 = apply_gates(d, disp_max=15.0, band=(0.45, 7.5))
    assert g2[100, 0] == 2.0 and np.isnan(g2[150, 0]) and np.isnan(g2[200, 0])
    assert d[210, 0] == -20.0 and d[150, 0] == np.float32(0.2)          # 元配列は不変
    with pytest.raises(ValueError):
        apply_gates(d, disp_max=15.0, band=(7.5, 0.45))

    stack = np.stack([d, d])
    m = evaluate_session(stack, 2000.0, TRUE, disp_max=15.0)
    z = depth_of(apply_gates(d, disp_max=15.0), TRUE)
    valid = z[(z > config.DEPTH_MIN) & (z < config.DEPTH_MAX)].astype(np.float64)
    assert m["frames_used"] == 2 and m["n_valid_total"] == 2 * valid.size
    assert m["mean_err_mm"] == pytest.approx(float(valid.mean() - 2000.0))
    assert m["mae_mm"] == pytest.approx(float(np.abs(valid - 2000.0).mean()))
    assert m["valid_rate"] == pytest.approx(valid.size / (H * W))
    assert m["frame_mean_est_std_mm"] == 0.0                              # 同じフレーム 2 枚
    m_band = evaluate_session(stack, 2000.0, TRUE, disp_max=15.0, band=(0.45, 7.5))
    assert m_band["n_valid_total"] == 2 * 10                              # 帯内の 10 px だけ
    m_roi = evaluate_session(stack, 2000.0, TRUE, roi=(0, 0, W, 120), disp_max=15.0)
    assert m_roi["n_valid_total"] == 2 * 10
    with pytest.raises(ValueError):
        evaluate_session(stack, 2000.0, TRUE, roi=(0, 0, W + 1, H))
    with pytest.raises(ValueError):
        evaluate_session(d, 2000.0, TRUE)
    empty = evaluate_session(np.full((1, H, W), np.nan, dtype=np.float32), 2000.0, TRUE)
    assert empty["n_valid_total"] == 0 and math.isnan(empty["mae_mm"])


def test_summarize_holdout():
    mb = {
        "a": {"n_valid_total": 10, "mae_mm": 10.0, "mean_err_mm": -3.0, "rmse_mm": 12.0, "valid_rate": 0.4},
        "b": {"n_valid_total": 10, "mae_mm": 30.0, "mean_err_mm": 5.0, "rmse_mm": 40.0, "valid_rate": 0.2},
        "c": {"n_valid_total": 0, "mae_mm": float("nan"), "mean_err_mm": float("nan"), "rmse_mm": float("nan"),
              "valid_rate": 0.0},
    }
    s = summarize_holdout(mb, ["a", "b", "c", "missing"])
    assert s["n_sessions"] == 2 and s["mae_mean_mm"] == 20.0 and s["abs_bias_max_mm"] == 5.0
    assert s["rmse_mean_mm"] == 26.0 and s["valid_rate_mean"] == pytest.approx(0.3)
    assert summarize_holdout(mb, ["c"])["n_sessions"] == 0
