"""tools/analyze_disparity_settings.py の HW 非依存テスト（合成視差・tmp_path・monkeypatch）。"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import config
import tools.analysis_common as common
import tools.analyze_disparity_settings as mod
from core.params import DepthParams
from tools.analyze_disparity_settings import COLUMNS, bench_setting, main, stability_summary

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


def test_stability_summary_known_sigma():
    rng = np.random.default_rng(0)
    stack = (2.0 + rng.normal(0, 0.1, size=(50, 20, 30))).astype(np.float32)
    stack[:, 0, :] = np.nan
    s = stability_summary({"a": stack, "b": stack.copy()})
    assert s["px_used_stability"] == 2 * 19 * 30
    assert 0.08 < s["sigma_median_px"] < 0.12 and s["flip_rate_mean"] == 0.0
    empty = stability_summary({"a": np.full((3, 4, 4), np.nan, dtype=np.float32)})
    assert empty["px_used_stability"] == 0 and np.isnan(empty["sigma_median_px"])


def test_bench_setting(monkeypatch):
    monkeypatch.setattr(common, "calcDisparityByZNCC", lambda red, blue, k, s, t: np.full((H, W), np.nan, np.float32))
    proc = np.random.default_rng(0).integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    b = bench_setting(proc, common.DispSettings(), 3)
    assert b["n_bench"] == 3 and b["edge_px_bench"] > 0 and b["mean_ms"] <= b["p95_ms"] + 1e-9
    b2 = bench_setting(proc, common.DispSettings(edge_th=200.0), 2)
    assert b2["edge_px_bench"] < b["edge_px_bench"]                      # 閾値を上げるとエッジ画素が減る


def test_cli_end_to_end(tmp_path, monkeypatch):
    cache = tmp_path / "cache"

    def fake_calib(settings, calib_dir, *, root=None, overwrite=False):
        d = cache / settings.key() / "calib"
        d.mkdir(parents=True, exist_ok=True)
        out = {}
        for z in CALIB:
            p = d / f"{z}.npy"
            if not p.exists():
                np.save(str(p), _synthetic_disparity(z, noise_px=0.02 * settings.kernel_size / 17))
            out[z] = p
        return out

    def fake_capture(settings, session, frames, *, root=None, overwrite=False):
        gt = int(session.split("_")[1])
        d = cache / settings.key() / session
        d.mkdir(parents=True, exist_ok=True)
        outs = []
        for i, _ in enumerate(frames):
            p = d / f"{i:06d}.npy"
            if not p.exists():
                np.save(str(p), _synthetic_disparity(gt, seed=gt * 10 + i))
            outs.append(p)
        return outs

    def fake_list_frames(cap_dir, *, skip_first=0, max_frames=None, step=1):
        return [Path(f"{i:06d}.png") for i in range(max_frames or 2)]

    monkeypatch.setattr(mod, "ensure_calib_disparity", fake_calib)
    monkeypatch.setattr(mod, "ensure_capture_disparity", fake_capture)
    monkeypatch.setattr(mod, "list_frames", fake_list_frames)
    out = tmp_path / "out"
    rc = main(["--sessions", "dist_1300", "dist_1500", "dist_2100", "dist_3100", "--out-dir", str(out), "--no-plots",
               "--no-bench", "--kernel-sizes", "9", "17", "--edge-ths", "30", "60", "--frames-per-session", "2",
               "--calib-dir", str(tmp_path)])
    assert rc == 0
    with (out / "settings.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == COLUMNS and len(rows) == 4
    assert [(r["param"], float(r["value"])) for r in rows] == [("kernel_size", 9.0), ("kernel_size", 17.0),
                                                              ("edge_th", 30.0), ("edge_th", 60.0)]
    assert rows[2]["kernel_size"] == str(config.KERNEL_SIZE) and float(rows[2]["edge_th"]) == 30.0
    for r in rows:
        assert r["calib_ok"] == "True" and float(r["v_pos"]) == pytest.approx(TRUE.v_pos, rel=5e-3)
        assert float(r["mae_mean_mm_nogate"]) < 40.0 and float(r["px_used_stability"]) > 0
        assert r["mean_ms"] == ""                                            # --no-bench
    with (out / "per_session.csv").open(encoding="utf-8", newline="") as f:
        per = list(csv.DictReader(f))
    assert list(per[0].keys()) == mod.PER_SESSION_COLUMNS
    assert len(per) == 4 * 4 * 2                                             # 設定 4 × セッション 4 × ゲート 2
    assert {r["gate"] for r in per} == {"nogate", "bandgate"} and all(float(r["mae_mm"]) < 60.0 for r in per)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["band"] == [config.DISP_PLOT_LOW, config.DISP_PLOT_HEIGHT] and summary["bench_frame"] == ""
    assert main(["--sessions", "dist_1500", "--kernel-sizes", "8", "--calib-dir", str(tmp_path)]) == 2   # 偶数 K
