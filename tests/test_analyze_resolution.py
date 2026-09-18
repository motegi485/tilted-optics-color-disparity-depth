"""tools/analyze_resolution.py の HW 非依存テスト（合成視差・tmp_path・monkeypatch）。"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import config
import tools.analyze_resolution as mod
from core.params import DepthParams
from tools.analyze_resolution import (
    BENCH_COLUMNS,
    CALIB_ROW_COLUMNS,
    COLUMNS,
    PER_SESSION_COLUMNS,
    assign_group,
    main,
    parse_size,
)

W0 = config.IMAGE_SIZE[0]
TRUE = DepthParams(v_pos=426000.0, u_pos=-61.0, a_pos=-26.4, w_pos=0.0,
                   v_neg=411000.0, u_neg=-17.4, a_neg=-22.8, w_neg=0.0)
CALIB_ALL = list(range(1000, 4001, 200))     # B3 の 16 距離
CORE = list(range(1400, 3201, 200))          # 2026-09-15 と同じ 10 距離


def _synthetic_disparity(depth: float, size: tuple[int, int], *, noise_px: float = 0.03, cols_per_row: int = 40,
                         seed=None) -> np.ndarray:
    """合焦行 y0 が r 倍・傾き a [行/px] は不変（視差 [px] も行も r 倍）という解像度スケールの関係で合成する。"""
    w, h = size
    r = w / W0
    rng = np.random.default_rng(int(depth) if seed is None else seed)
    d = np.full((h, w), np.nan, dtype=np.float32)
    y0_pos = (TRUE.v_pos / depth + TRUE.u_pos) * r
    y0_neg = (TRUE.v_neg / depth + TRUE.u_neg) * r
    cols = rng.choice(w, size=cols_per_row, replace=False)
    for y in range(h):
        if y < y0_pos:
            val = (y - y0_pos) / TRUE.a_pos
        elif y > y0_neg:
            val = (y - y0_neg) / TRUE.a_neg
        else:
            continue
        d[y, cols] = val + rng.normal(0.0, noise_px * r, size=cols_per_row)
    return d


def test_parse_size_and_assign_group():
    assert parse_size("1056x704") == (1056, 704) and parse_size("264X176") == (264, 176)
    for bad in ("1056", "1056x", "640x480", "axb", "528x352x3"):
        with pytest.raises(ValueError):
            parse_size(bad)
    assert assign_group(1500, CORE) == "interior" and assign_group(1400, CORE) == "at_calib"
    assert assign_group(1300, CORE) == "extrap" and assign_group(3300, CORE) == "extrap"
    assert assign_group(4000, CALIB_ALL) == "at_calib" and assign_group(3900, CALIB_ALL) == "interior"
    with pytest.raises(ValueError):
        assign_group(1500, [])


def _install_fakes(monkeypatch, cache: Path, calib_depths: list[int], *, fake_frames: bool = True):
    """視差キャッシュ関数を、解像度に応じた合成視差を返す fake に差し替える（disparity 計算は呼ばない）。
    fake_frames=False なら list_frames は実物のまま（bench が実 PNG を読むテスト用）。"""

    def fake_calib_raw(settings, raw_dir, *, distances=None, root=None, overwrite=False):
        depths = sorted(set(distances)) if distances is not None else calib_depths
        missing = [z for z in depths if z not in calib_depths]
        if missing:
            raise FileNotFoundError(f"missing {missing}")
        d = cache / settings.key() / "calib_raw"
        d.mkdir(parents=True, exist_ok=True)
        out = {}
        for z in depths:
            p = d / f"{z}.npy"
            if not p.exists():
                np.save(str(p), _synthetic_disparity(z, settings.image_size))
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
                np.save(str(p), _synthetic_disparity(gt, settings.image_size, seed=gt * 10 + i))
            outs.append(p)
        return outs

    def fake_list_frames(cap_dir, *, skip_first=0, max_frames=None, step=1):
        return [Path(f"{i:06d}.png") for i in range(max_frames or 2)]

    monkeypatch.setattr(mod, "ensure_calib_disparity_from_raw", fake_calib_raw)
    monkeypatch.setattr(mod, "ensure_capture_disparity", fake_capture)
    if fake_frames:
        monkeypatch.setattr(mod, "list_frames", fake_list_frames)


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_cli_end_to_end_scales_settings_and_recovers_scaled_coefficients(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, tmp_path / "cache", CALIB_ALL)
    raw_dir = tmp_path / "calib_raw"
    raw_dir.mkdir()
    out = tmp_path / "out"
    sessions = ["dist_1300", "dist_1500", "dist_2100", "dist_3100", "dist_3300"]
    rc = main(["--sessions", *sessions, "--calib-raw-dir", str(raw_dir), "--calib-distances", *map(str, CORE),
               "--image-sizes", "264x176", "528x352", "1056x704", "--base-kernel", "17", "--base-search-range", "41",
               "--frames-per-session", "2", "--out-dir", str(out), "--no-plots", "--no-bench"])
    assert rc == 0
    rows = _read(out / "settings.csv")
    assert list(rows[0].keys()) == COLUMNS and [r["width"] for r in rows] == ["264", "528", "1056"]   # 小さい順
    assert [(int(r["kernel_size"]), int(r["search_range_x"])) for r in rows] == [(9, 21), (17, 41), (35, 83)]
    assert [float(r["scale"]) for r in rows] == [0.5, 1.0, 2.0] and all(float(r["edge_th"]) == config.EDGE_TH for r in rows)
    for r in rows:
        s = float(r["scale"])
        assert r["calib_ok"] == "True" and r["calib_distances"] == " ".join(map(str, CORE)) and r["skipped"] == ""
        assert float(r["band_low"]) == pytest.approx(config.DISP_PLOT_LOW * s)
        assert float(r["band_high"]) == pytest.approx(config.DISP_PLOT_HEIGHT * s)
        assert float(r["ransac_th"]) == pytest.approx(config.RANSAC_RESIDUAL_TH * s)
        assert float(r["v_pos"]) == pytest.approx(TRUE.v_pos * s, rel=5e-3)     # 切片 [行] は r 倍
        assert float(r["u_neg"]) == pytest.approx(TRUE.u_neg * s, abs=2.0)
        assert float(r["a_pos"]) == pytest.approx(TRUE.a_pos, rel=2e-2)          # 傾き [行/px] は不変
        assert (r["n_interior"], r["n_at_calib"], r["n_extrap"]) == ("3", "0", "2")
        assert float(r["interior_mae_mean_mm_nogate"]) < 40.0 and float(r["extrap_mae_mean_mm_nogate"]) < 80.0
        assert float(r["px_used_stability"]) > 0 and r["mean_ms"] == "" and r["bench_session"] == ""
        assert float(r["sigma_median_base_px"]) == pytest.approx(float(r["sigma_median_px"]) / s)
    calib_rows = _read(out / "calib_rows.csv")
    assert list(calib_rows[0].keys()) == CALIB_ROW_COLUMNS and len(calib_rows) == 3 * len(CORE)
    assert [int(r["depth"]) for r in calib_rows[:len(CORE)]] == CORE
    per = _read(out / "per_session.csv")
    assert list(per[0].keys()) == PER_SESSION_COLUMNS and len(per) == 3 * len(sessions) * 2
    assert {r["gate"] for r in per} == {"nogate", "bandgate"}
    assert {(r["session"], r["group"]) for r in per} == {("dist_1300", "extrap"), ("dist_1500", "interior"),
                                                       ("dist_2100", "interior"), ("dist_3100", "interior"),
                                                       ("dist_3300", "extrap")}
    assert _read(out / "bench.csv") == []
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["image_sizes"] == ["264x176", "528x352", "1056x704"] and summary["calib_distances"] == CORE
    assert summary["base"] == {"kernel_size": 17, "search_range_x": 41, "edge_th": 30.0, "image_size": [528, 352]}
    assert summary["bench_frames"] == [] and summary["scale_edge_th"] is False


def test_cli_all_distances_skips_sparse_near_depths_and_groups_at_calib(tmp_path, monkeypatch):
    """16 距離すべてで校正すると 1000・1100 mm は neg 枝の点が無くスキップされ（calibrate と同じ規則）、
    校正範囲は使った距離で決まる。同じ距離の評価セッションは at_calib 群になる。"""
    _install_fakes(monkeypatch, tmp_path / "cache", CALIB_ALL)
    raw_dir = tmp_path / "calib_raw"
    raw_dir.mkdir()
    out = tmp_path / "out"
    rc = main(["--sessions", "dist_1000", "dist_1200", "dist_1500", "dist_3900", "dist_4000",
               "--calib-raw-dir", str(raw_dir), "--image-sizes", "528x352", "--frames-per-session", "2",
               "--scale-edge-th", "--out-dir", str(out), "--no-plots", "--no-bench"])
    assert rc == 0
    row = _read(out / "settings.csv")[0]
    assert row["calib_ok"] == "True" and row["calib_distances"] == " ".join(map(str, CALIB_ALL))
    assert row["skipped"] == "1000" and row["n_skipped"] == "1"                     # y0_neg=394 > 351 → neg 枝が空
    assert (row["n_interior"], row["n_at_calib"], row["n_extrap"]) == ("2", "2", "1")
    per = {(r["session"], r["gate"]): r for r in _read(out / "per_session.csv")}
    assert per[("dist_1000", "nogate")]["group"] == "extrap"                       # 1000 は校正から外れたので外挿
    assert per[("dist_1200", "nogate")]["group"] == "at_calib" and per[("dist_4000", "nogate")]["group"] == "at_calib"
    assert per[("dist_3900", "nogate")]["group"] == "interior"
    depths = [int(r["depth"]) for r in _read(out / "calib_rows.csv")]
    assert depths == [z for z in CALIB_ALL if z != 1000]
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["scale_edge_th"] is True


def test_cli_bench_reads_raw_frame_and_scales_it(tmp_path, monkeypatch):
    """bench は評価 raw の 1 フレームを各解像度に preprocess_to してから計時する（実 ZNCC・小解像度で高速）。"""
    _install_fakes(monkeypatch, tmp_path / "cache", CORE, fake_frames=False)   # list_frames は実物
    cap = tmp_path / "captures" / "dist_1500"
    cap.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for i in range(3):
        cv2.imwrite(str(cap / f"{i:06d}.png"),
                    rng.integers(0, 256, size=(config.WEBCAM_NATIVE_SIZE[1], config.WEBCAM_NATIVE_SIZE[0], 3), dtype=np.uint8))
    raw_dir = tmp_path / "calib_raw"
    raw_dir.mkdir()
    out = tmp_path / "out"
    rc = main(["--sessions", "dist_1500", "--calib-raw-dir", str(raw_dir), "--captures-dir", str(tmp_path / "captures"),
               "--image-sizes", "264x176", "--frames-per-session", "2", "--bench-sessions", "dist_1500", "--bench-n", "2",
               "--skip-first", "1", "--out-dir", str(out)])                # 図も生成する
    assert rc == 0
    bench = _read(out / "bench.csv")
    assert list(bench[0].keys()) == BENCH_COLUMNS and len(bench) == 1
    assert bench[0]["session"] == "dist_1500" and bench[0]["frame"] == "000001.png" and bench[0]["n_bench"] == "2"
    row = _read(out / "settings.csv")[0]
    assert row["bench_session"] == "dist_1500" and float(row["p95_ms"]) >= float(row["mean_ms"]) - 1e-9
    assert (out / "resolution_tradeoff.png").is_file()


def test_cli_rejects_bad_inputs(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, tmp_path / "cache", CORE)
    raw_dir = tmp_path / "calib_raw"
    raw_dir.mkdir()
    base = ["--sessions", "dist_1500", "--calib-raw-dir", str(raw_dir), "--no-bench", "--no-plots",
            "--out-dir", str(tmp_path / "out")]
    assert main(base + ["--image-sizes", "640x480"]) == 2                 # 3:2 でない
    assert main(base + ["--image-sizes", "528"]) == 2                     # 形式不正
    assert main(base + ["--base-kernel", "16"]) == 2                      # 偶数 K
    assert main(base + ["--band-low", "8", "--band-high", "7.5"]) == 2    # low >= high
    assert main(base + ["--calib-distances"]) == 2                        # 空
    assert main(base + ["--calib-distances", "1400", "5000"]) == 2        # raw の無い距離
    assert main(["--sessions", "dist_1500", "--calib-raw-dir", str(tmp_path / "nope"), "--no-bench"]) == 2
