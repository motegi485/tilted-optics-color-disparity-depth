"""tools/analyze_calib_sensitivity.py の HW 非依存テスト（合成視差・tmp_path・monkeypatch）。"""
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

import config
import tools.analyze_calib_sensitivity as mod
from core.params import DepthParams
from tools.analysis_common import CalibResult, calibrate_from_disparity
from tools.analyze_calib_sensitivity import calib_summary, main, parse_th, residual_vs_disparity

W, H = config.IMAGE_SIZE
TRUE = DepthParams(v_pos=426000.0, u_pos=-61.0, a_pos=-26.4, w_pos=0.0,
                   v_neg=411000.0, u_neg=-17.4, a_neg=-22.8, w_neg=0.0)
CALIB = list(range(1400, 3201, 200))


def _synthetic_disparity(depth: float, *, noise_px: float = 0.05, cols_per_row: int = 40, seed=None) -> np.ndarray:
    rng = np.random.default_rng(int(depth) if seed is None else seed)
    d = np.full((H, W), np.nan, dtype=np.float32)
    y0_pos = TRUE.v_pos / depth + TRUE.u_pos
    y0_neg = TRUE.v_neg / depth + TRUE.u_neg
    cols = rng.choice(W, size=cols_per_row, replace=False)
    for y in range(H):
        if y < y0_pos:
            val = (y - y0_pos) / TRUE.a_pos
        elif y > y0_neg:
            val = (y - y0_neg) / TRUE.a_neg
        else:
            continue
        d[y, cols] = val + rng.normal(0.0, noise_px, size=cols_per_row)
    return d


def test_parse_th():
    assert parse_th("8") == 8.0 and parse_th("none") is None and parse_th("MAD") is None
    with pytest.raises(ValueError):
        parse_th("0")


def test_calib_summary_gap():
    res = calibrate_from_disparity({z: _synthetic_disparity(z) for z in CALIB}, low=0.45, high=7.5)
    s = calib_summary(res)
    p = res.params
    assert s["y0_gap_2300_px"] == pytest.approx((p.v_neg / 2300 + p.u_neg) - (p.v_pos / 2300 + p.u_pos))
    assert s["n_used_pos"] == 10 and s["n_skipped"] == 0 and 0 < s["inlier_pos_mean"] <= 1


def test_residual_vs_disparity_on_straight_lines():
    rows = residual_vs_disparity({z: _synthetic_disparity(z) for z in CALIB}, low=0.45, high=7.5, ransac_th=8.0,
                                 ransac_seed=0, bin_px=1.0, reach_max=14.5, inlier_th=8.0)
    assert {r["sign"] for r in rows} == {"pos", "neg"}
    pos = [r for r in rows if r["sign"] == "pos" and r["n_px"] > 0]
    assert all(abs(r["resid_median_row"]) < 1.0 for r in pos)                                   # 直線どおり
    # |d| < 1 のビンは、ノイズで符号が反転した画素が反対枝の直線で評価され外れ値になる（実データでも同じ性質）
    assert pos[0]["inlier_rate_th"] > 0.95 and all(r["inlier_rate_th"] > 0.99 for r in pos[1:])
    assert pos[0]["bin_lo"] == 0.0 and pos[0]["n_px"] > 0                                        # 帯の下側も集計される


def test_cli_end_to_end(tmp_path, monkeypatch):
    cache = tmp_path / "cache"

    def fake_calib(settings, calib_dir, *, root=None, overwrite=False):
        d = cache / "calib"
        d.mkdir(parents=True, exist_ok=True)
        out = {}
        for z in CALIB:
            p = d / f"{z}.npy"
            if not p.exists():
                np.save(str(p), _synthetic_disparity(z))
            out[z] = p
        return out

    def fake_capture(settings, session, frames, *, root=None, overwrite=False):
        gt = int(session.split("_")[1])
        d = cache / session
        d.mkdir(parents=True, exist_ok=True)
        outs = []
        for i, _ in enumerate(frames):
            p = d / f"{i:06d}.npy"
            if not p.exists():
                np.save(str(p), _synthetic_disparity(gt, seed=gt * 10 + i))
            outs.append(p)
        return outs

    def fake_list_frames(cap_dir, *, skip_first=0, max_frames=None, step=1):
        return [Path(f"{i:06d}.png") for i in range(2)]

    monkeypatch.setattr(mod, "ensure_calib_disparity", fake_calib)
    monkeypatch.setattr(mod, "ensure_capture_disparity", fake_capture)
    monkeypatch.setattr(mod, "list_frames", fake_list_frames)
    out = tmp_path / "out"
    rc = main(["--sessions", "dist_1300", "dist_1500", "dist_2100", "dist_3100", "--out-dir", str(out), "--no-plots",
               "--lows", "0.45", "1.0", "--highs", "5", "7.5", "--ths", "8", "none", "--frames-per-session", "2",
               "--residual-bin", "1.0", "--calib-dir", str(tmp_path)])
    assert rc == 0
    with (out / "grid.csv").open(encoding="utf-8", newline="") as f:
        grid = list(csv.DictReader(f))
    assert len(grid) == 4 and list(grid[0].keys()) == mod.GRID_COLUMNS
    assert all(r["calib_ok"] == "True" for r in grid)
    for r in grid:
        assert float(r["v_pos"]) == pytest.approx(TRUE.v_pos, rel=5e-3)
        assert float(r["mae_mean_mm_nogate"]) < 40.0 and float(r["valid_rate_mean_bandgate"]) <= float(r["valid_rate_mean_nogate"])
        assert float(r["extrap_mae_mean_mm_nogate"]) < 60.0
    with (out / "th_sweep.csv").open(encoding="utf-8", newline="") as f:
        ths = list(csv.DictReader(f))
    assert [r["th"] for r in ths] == ["8", "none"] and all(r["calib_ok"] == "True" for r in ths)
    with (out / "inference_band.csv").open(encoding="utf-8", newline="") as f:
        inf = list(csv.DictReader(f))
    assert len(inf) == 4 and list(inf[0].keys()) == mod.INF_COLUMNS
    with (out / "residual_vs_disparity.csv").open(encoding="utf-8", newline="") as f:
        res = list(csv.DictReader(f))
    assert list(res[0].keys()) == mod.RES_COLUMNS and {r["sign"] for r in res} == {"pos", "neg"}
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["holdout"] == ["dist_1500", "dist_2100", "dist_3100"] and summary["extrapolation"] == ["dist_1300"]
    assert summary["current"]["low"] == 0.45 and summary["current"]["high"] == 7.5
    assert summary["best_nogate"]["low"] in (0.45, 1.0) and summary["best_inference_band"] is not None
    # 引数不備
    assert main(["--sessions", "dist_1500", "--th", "0", "--calib-dir", str(tmp_path)]) == 2
