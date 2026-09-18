"""checks/evaluate_disparity_band_fraction.py の HW 非依存テスト（合成 PNG・既知の視差・tmp_path・monkeypatch のみ）。"""
import csv
import math

import cv2
import numpy as np
import pytest

import config
import checks.evaluate_disparity_band_fraction as mod
from checks.evaluate_disparity_band_fraction import main, summarize_session
from core.depth import compute_depth
from core.params import DepthParams

H, W = config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]   # (352, 528)
FULL = (0, 0, W, H)
# 2026-09-15 live 校正の桁に近い係数（全桁一致は目的ではない）
PARAMS = DepthParams(v_pos=426165.15, u_pos=-61.04, a_pos=-26.43, w_pos=0.0,
                     v_neg=411311.86, u_neg=-17.39, a_neg=-22.76, w_neg=0.0)


def _disp_frame() -> np.ndarray:
    """既知の視差マップ: 全 NaN に below_band・in_band・above_band・帯の境界値・d=0 を置く。"""
    d = np.full((H, W), np.nan, dtype=np.float32)
    d[150, 10:20] = 0.2      # below_band・pos・10 px
    d[160, 10:20] = -0.3     # below_band・neg・10 px
    d[155, 10:15] = 0.45     # 境界 |d| == band_low → below_band（校正は > low しか採らない）・5 px
    d[50, 10:40] = 3.0       # in_band・pos・30 px
    d[300, 10:40] = -3.0     # in_band・neg・30 px
    d[100, 10:15] = 10.0     # above_band・pos・5 px（gate1 の DISP_MAX=20 は通る。推論帯 HIGH=5 では落ちる → z_live で NaN）
    d[105, 10:15] = 7.5      # 境界 |d| == band_high → above_band（校正は < high しか採らない）・5 px
    d[170, 10:15] = 0.0      # d == 0 → compute_depth が NaN にする（有効画素に入らない）
    return d


N_BELOW, N_IN, N_ABOVE = 25, 60, 10
N_VALID = N_BELOW + N_IN + N_ABOVE


def _depth_of(d: np.ndarray) -> np.ndarray:
    return compute_depth(d, PARAMS.v_pos, PARAMS.u_pos, PARAMS.a_pos, PARAMS.w_pos,
                         PARAMS.v_neg, PARAMS.u_neg, PARAMS.a_neg, PARAMS.w_neg, config.EPS_DEN)


def _depth_live_of(d: np.ndarray) -> np.ndarray:
    """本番（process_frame）と同じ推論帯 |d| <= LOW または |d| >= HIGH → NaN を掛けた z（照合用・2026-09-18）。"""
    d2 = d.copy()
    ad = np.abs(d2)
    d2[(ad <= config.INFER_BAND_LOW) | (ad >= config.INFER_BAND_HIGH)] = np.nan
    return _depth_of(d2)


def _fake_compute(raw, params):
    d = _disp_frame()
    return d, _depth_of(d), _depth_live_of(d)


def _write_pngs(cap_dir, n: int, size=(8, 8)) -> None:
    cap_dir.mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(str(cap_dir / f"{i:06d}.png"), np.zeros((size[1], size[0], 3), dtype=np.uint8))


def _write_npys(out_dir, n: int, depth: np.ndarray) -> None:
    out_dir.mkdir(parents=True)
    for i in range(n):
        np.save(str(out_dir / f"{i:06d}.npy"), depth)


def test_bands_partition_valid_pixels_and_match_known_disparity(tmp_path):
    cap = tmp_path / "cap" / "dist_2000"
    _write_pngs(cap, 3)
    r = summarize_session(cap, None, 2000, PARAMS, roi=FULL, compute=_fake_compute)
    d = _disp_frame()
    z = _depth_of(d)
    valid = (z > config.DEPTH_MIN) & (z < config.DEPTH_MAX)
    assert int(valid.sum()) == N_VALID                   # d=0 の 5 px は無効
    assert r["frames_used"] == 3 and r["frames_total"] == 3
    assert r["n_valid_total"] == N_VALID * 3
    assert r["n_below_band"] == N_BELOW * 3
    assert r["n_in_band"] == N_IN * 3
    assert r["n_above_band"] == N_ABOVE * 3
    assert r["n_below_band"] + r["n_in_band"] + r["n_above_band"] == r["n_valid_total"]   # 3 区分で過不足なし
    assert r["below_band_fraction"] == pytest.approx(N_BELOW / N_VALID)
    assert r["below_band_fraction_min"] == pytest.approx(N_BELOW / N_VALID)
    assert r["below_band_fraction_max"] == pytest.approx(N_BELOW / N_VALID)
    assert r["in_band_fraction"] == pytest.approx(N_IN / N_VALID)
    assert r["above_band_fraction"] == pytest.approx(N_ABOVE / N_VALID)
    # below_band 画素の行: 150×10・155×5・160×10 → median 155・p10 150・p90 160
    assert r["below_band_row_median"] == 155.0 and r["below_band_row_p10"] == 150.0 and r["below_band_row_p90"] == 160.0
    ad = np.abs(d)
    for name, mask in (("below_band", valid & (ad <= 0.45)),
                       ("in_band", valid & (ad > 0.45) & (ad < 7.5)),
                       ("above_band", valid & (ad >= 7.5))):
        e = z[mask].astype(np.float64) - 2000.0
        assert r[f"mean_err_{name}_mm"] == pytest.approx(float(e.mean()))
        assert r[f"mae_{name}_mm"] == pytest.approx(float(np.abs(e).mean()))
    assert r["replay_match"] == ""                        # out_dir=None なら照合しない
    assert r["band_low_px"] == config.DISP_PLOT_LOW and r["band_high_px"] == config.DISP_PLOT_HEIGHT


def test_roi_skip_and_max_frames(tmp_path):
    cap = tmp_path / "cap" / "dist_2000"
    _write_pngs(cap, 5)
    # ROI を行 140–165 に絞ると below_band の 25 px だけが残る（in_band・above_band は ROI 外）
    r = summarize_session(cap, None, 2000, PARAMS, roi=(0, 140, W, 165), skip_first=2, max_frames=2,
                          compute=_fake_compute)
    assert r["frames_used"] == 2 and r["frames_total"] == 5
    assert r["n_valid_total"] == N_BELOW * 2 and r["n_below_band"] == N_BELOW * 2
    assert r["below_band_fraction"] == pytest.approx(1.0)
    assert r["n_in_band"] == 0 and r["in_band_fraction"] == 0.0
    assert r["n_above_band"] == 0 and r["above_band_fraction"] == 0.0
    assert r["below_band_row_median"] == 155.0            # 行は ROI の y0 を足した絶対座標
    assert math.isnan(r["mean_err_in_band_mm"]) and math.isnan(r["mae_above_band_mm"])


def test_replay_match_true_false_and_missing(tmp_path):
    cap = tmp_path / "cap" / "dist_2000"
    _write_pngs(cap, 2)
    good = _depth_live_of(_disp_frame())                  # 保存 npy は本番手順（推論帯適用後）の値
    out = tmp_path / "out" / "dist_2000"
    _write_npys(out, 2, good)
    r = summarize_session(cap, out, 2000, PARAMS, roi=FULL, compute=_fake_compute)
    assert r["replay_match"] is True
    # 区分の集計は推論帯を掛ける前の (d, z) で行う: 推論帯で落ちる below_band・above_band も数えられている
    assert r["n_below_band"] == N_BELOW * 2 and r["n_above_band"] == N_ABOVE * 2
    bad = good.copy()
    bad[50, 10] += 1.0                                    # 有限値 1 画素の不一致（行 50 は帯内 d=3.0）
    np.save(str(out / "000001.npy"), bad)
    assert summarize_session(cap, out, 2000, PARAMS, roi=FULL, compute=_fake_compute)["replay_match"] is False
    (out / "000001.npy").unlink()                         # npy 欠落も False
    assert summarize_session(cap, out, 2000, PARAMS, roi=FULL, compute=_fake_compute)["replay_match"] is False


def test_replay_match_false_for_npy_without_inference_band(tmp_path):
    """推論帯を掛けていない z（2026-09-18 以前の旧設定セッションの保存 npy に相当）とは照合不一致になる。"""
    cap = tmp_path / "cap" / "dist_2000"
    _write_pngs(cap, 1)
    out = tmp_path / "out" / "dist_2000"
    _write_npys(out, 1, _depth_of(_disp_frame()))        # gate1 のみの z
    assert summarize_session(cap, out, 2000, PARAMS, roi=FULL, compute=_fake_compute)["replay_match"] is False


def test_disparity_and_depth_live_matches_process_frame():
    """既定の compute（本物の ZNCC）で、z_live が realtime.pipeline.process_frame とビット一致し、
    z（gate1 のみ）は推論帯ぶん有効画素が多いことを確認する。"""
    from realtime.camera import preprocess
    from realtime.pipeline import process_frame

    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(config.WEBCAM_NATIVE_SIZE[1], config.WEBCAM_NATIVE_SIZE[0], 3), dtype=np.uint8)
    d, z, z_live = mod.disparity_and_depth(raw, PARAMS)
    ref = process_frame(preprocess(raw), PARAMS)
    assert np.array_equal(np.isnan(z_live), np.isnan(ref))
    assert np.array_equal(z_live[~np.isnan(z_live)], ref[~np.isnan(ref)])
    assert np.count_nonzero(~np.isnan(z)) > np.count_nonzero(~np.isnan(z_live))
    assert d.shape == z.shape == z_live.shape == (H, W)


def test_no_valid_pixels_gives_nan_stats(tmp_path):
    cap = tmp_path / "cap" / "dist_2000"
    _write_pngs(cap, 1)

    def all_nan(raw, params):
        d = np.full((H, W), np.nan, dtype=np.float32)
        return d, _depth_of(d), _depth_of(d)

    r = summarize_session(cap, None, 2000, PARAMS, roi=FULL, compute=all_nan)
    assert r["n_valid_total"] == 0 and r["n_below_band"] == 0
    assert all(math.isnan(r[k]) for k in ("below_band_fraction", "in_band_fraction", "above_band_fraction",
                                          "below_band_row_median", "mean_err_below_band_mm", "mae_in_band_mm"))


def test_invalid_inputs_raise(tmp_path):
    cap = tmp_path / "cap" / "dist_2000"
    _write_pngs(cap, 2)
    with pytest.raises(ValueError):
        summarize_session(cap, None, 2000, PARAMS, roi=FULL, skip_first=-1, compute=_fake_compute)
    with pytest.raises(ValueError):
        summarize_session(cap, None, 2000, PARAMS, roi=FULL, max_frames=0, compute=_fake_compute)
    with pytest.raises(ValueError):
        summarize_session(cap, None, 2000, PARAMS, roi=FULL, skip_first=5, compute=_fake_compute)
    with pytest.raises(ValueError):
        summarize_session(cap, None, 2000, PARAMS, roi=FULL, band_low=7.5, band_high=0.45, compute=_fake_compute)
    with pytest.raises(FileNotFoundError):
        summarize_session(tmp_path / "cap" / "empty", None, 2000, PARAMS, roi=FULL, compute=_fake_compute)


# ---------- CLI（preprocess は本物・ZNCC だけ既知の視差に差し替える） ----------

def _write_params_csv(path) -> None:
    rows = [("v_pos", PARAMS.v_pos), ("u_pos", PARAMS.u_pos), ("w_pos", 0.0), ("a_pos", PARAMS.a_pos),
            ("v_neg", PARAMS.v_neg), ("u_neg", PARAMS.u_neg), ("w_neg", 0.0), ("a_neg", PARAMS.a_neg)]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "value"])
        w.writerows(rows)


def test_cli_writes_csv_and_exit_codes(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "calcDisparityByZNCC", lambda red, blue, k, s, t: _disp_frame())
    cap_root, out_root = tmp_path / "captures", tmp_path / "outputs"
    _write_pngs(cap_root / "dist_2000", 3, size=config.WEBCAM_NATIVE_SIZE)   # preprocess を通す実寸 PNG
    _write_npys(out_root / "dist_2000", 3, _depth_live_of(_disp_frame()))   # 保存 npy は推論帯適用後
    params_csv = tmp_path / "fit_result.csv"
    _write_params_csv(params_csv)
    out_csv = tmp_path / "bands.csv"
    rc = main(["--sessions", "dist_2000", "--skip-first", "1", "--params", str(params_csv),
               "--captures-dir", str(cap_root), "--outputs-dir", str(out_root), "--out", str(out_csv)])
    assert rc == 0
    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["session"] for r in rows] == ["dist_2000"]
    assert list(rows[0].keys()) == mod.COLUMNS
    assert rows[0]["frames_used"] == "2" and rows[0]["n_valid_total"] == str(N_VALID * 2)
    assert float(rows[0]["below_band_fraction"]) == pytest.approx(N_BELOW / N_VALID)
    assert rows[0]["replay_match"] == "True"
    assert rows[0]["params_path"] == str(params_csv)

    # 保存 npy と食い違えば exit 1（結果は書くが警告）
    np.save(str(out_root / "dist_2000" / "000002.npy"), np.full((H, W), np.nan, dtype=np.float32))
    rc = main(["--sessions", "dist_2000", "--params", str(params_csv),
               "--captures-dir", str(cap_root), "--outputs-dir", str(out_root), "--out", str(out_csv)])
    assert rc == 1
    # --no-replay-check なら照合しないので 0 に戻る
    rc = main(["--sessions", "dist_2000", "--params", str(params_csv), "--no-replay-check",
               "--captures-dir", str(cap_root), "--outputs-dir", str(out_root), "--out", str(out_csv)])
    assert rc == 0
    # captures が無い・正解距離が読めないセッション名は exit 2
    assert main(["--sessions", "dist_9999", "--params", str(params_csv),
                 "--captures-dir", str(cap_root), "--outputs-dir", str(out_root)]) == 2
    assert main(["--sessions", "nodigits", "--captures-dir", str(cap_root)]) == 2
