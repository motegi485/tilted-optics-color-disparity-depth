"""checks/evaluate_distance_accuracy.py の HW 非依存テスト（合成 npy・tmp_path・monkeypatch のみ）。"""
import csv
import math

import numpy as np
import pytest

import config
from checks.evaluate_distance_accuracy import (
    load_manifest, main, parse_gt_from_session_name, parse_roi, parse_runner_log,
    summarize_errors, summarize_session,
)


def _write_session(out_root, name: str, frames: list[np.ndarray]) -> None:
    d = out_root / name
    d.mkdir(parents=True)
    for i, fr in enumerate(frames):
        np.save(str(d / f"{i:06d}.npy"), fr.astype(np.float32))
    (d / "meta.json").write_text('{"params_path": "params/fit_result.csv"}', encoding="utf-8")


def _frame(h: int, w: int, values: dict[tuple[int, int], float]) -> np.ndarray:
    """全 NaN の (h, w) 深度マップに、(row, col) -> 値 を書き込む。"""
    fr = np.full((h, w), np.nan, dtype=np.float32)
    for (r, c), v in values.items():
        fr[r, c] = v
    return fr


# ---------- 単体: 正解距離の解釈・ROI・誤差統計 ----------

def test_parse_gt_from_session_name():
    assert parse_gt_from_session_name("dist_1500") == 1500
    assert parse_gt_from_session_name("eval_2026-09-15_3100") == 3100
    with pytest.raises(ValueError):
        parse_gt_from_session_name("nodigits")


def test_parse_gt_rejects_out_of_range():
    """runner 既定のセッション名（日時）を渡すと末尾の時刻 101530 が距離に化けるので、
    gate2 レンジ外（DEPTH_MAX=50000 超）として拒否する。"""
    with pytest.raises(ValueError):
        parse_gt_from_session_name("20260915_101530")
    with pytest.raises(ValueError):
        parse_gt_from_session_name("dist_0")


def test_parse_roi_default_and_validation():
    assert parse_roi(None, 528, 352) == (0, 0, 528, 352)
    assert parse_roi("10,20,100,200", 528, 352) == (10, 20, 100, 200)
    for bad in ("1,2,3", "a,b,c,d", "100,20,10,200", "0,0,529,352", "0,0,528,353"):
        with pytest.raises(ValueError):
            parse_roi(bad, 528, 352)


def test_summarize_errors_values():
    e = np.array([-10.0, 0.0, 10.0, 20.0])
    s = summarize_errors(e)
    assert s["mean_err_mm"] == pytest.approx(5.0)
    assert s["mae_mm"] == pytest.approx(10.0)
    assert s["std_err_mm"] == pytest.approx(math.sqrt(((15 ** 2) + (5 ** 2) + (5 ** 2) + (15 ** 2)) / 4))
    assert s["rmse_mm"] == pytest.approx(math.sqrt((100 + 0 + 100 + 400) / 4))
    assert s["median_err_mm"] == pytest.approx(5.0)
    with pytest.raises(ValueError):
        summarize_errors(np.array([]))


# ---------- セッション集計 ----------

def test_summarize_session_pooled_and_per_frame(tmp_path):
    h, w = 4, 6
    # フレーム0: 有効3画素 (2000, 2010, 1990)、フレーム1: 有効1画素 (2100)、
    # フレーム2: gate2 外 (0 と 60000) と NaN のみ → 有効0
    f0 = _frame(h, w, {(0, 0): 2000.0, (1, 1): 2010.0, (2, 2): 1990.0})
    f1 = _frame(h, w, {(3, 5): 2100.0})
    f2 = _frame(h, w, {(0, 1): 0.0, (1, 2): 60000.0})
    _write_session(tmp_path, "dist_2000", [f0, f1, f2])

    r = summarize_session(tmp_path / "dist_2000", 2000, roi=(0, 0, w, h))

    assert r["frames_used"] == 3 and r["frames_total"] == 3
    assert r["n_valid_total"] == 4
    assert r["n_roi_px_per_frame"] == h * w
    assert r["valid_rate"] == pytest.approx(4 / (h * w * 3))
    assert r["n_valid_min"] == 0 and r["n_valid_max"] == 3
    errors = np.array([0.0, 10.0, -10.0, 100.0])
    assert r["mean_est_mm"] == pytest.approx(2000 + errors.mean())
    assert r["mean_err_mm"] == pytest.approx(errors.mean())
    assert r["mae_mm"] == pytest.approx(np.abs(errors).mean())
    assert r["std_err_mm"] == pytest.approx(errors.std())
    assert r["rmse_mm"] == pytest.approx(np.sqrt((errors ** 2).mean()))
    # フレーム平均: f0=2000, f1=2100（f2 は有効0で除外）
    assert r["frame_mean_est_avg_mm"] == pytest.approx(2050.0)
    assert r["frame_mean_est_std_mm"] == pytest.approx(np.std([2000.0, 2100.0], ddof=1))


def test_summarize_session_roi_and_skip(tmp_path):
    h, w = 4, 6
    f0 = _frame(h, w, {(0, 0): 1500.0, (3, 5): 1600.0})
    f1 = _frame(h, w, {(0, 0): 1500.0, (3, 5): 1700.0})
    _write_session(tmp_path, "dist_1500", [f0, f1])

    # ROI を右下 1 画素に限定し、先頭フレームをスキップ → 1700 の1画素だけ
    r = summarize_session(tmp_path / "dist_1500", 1500, roi=(5, 3, 6, 4), skip_first=1)
    assert r["frames_used"] == 1 and r["frames_total"] == 2
    assert r["n_valid_total"] == 1
    assert r["mean_err_mm"] == pytest.approx(200.0)
    assert math.isnan(r["frame_mean_est_std_mm"])       # 有効フレーム1枚では定義できない

    with pytest.raises(ValueError):
        summarize_session(tmp_path / "dist_1500", 1500, roi=(0, 0, w, h), skip_first=2)   # 集計0件
    with pytest.raises(ValueError):
        summarize_session(tmp_path / "dist_1500", 1500, roi=(0, 0, w + 1, h))            # ROI 超過


def test_summarize_session_all_invalid_returns_nan_stats(tmp_path):
    h, w = 3, 3
    _write_session(tmp_path, "dist_1500", [np.full((h, w), np.nan)])
    r = summarize_session(tmp_path / "dist_1500", 1500, roi=(0, 0, w, h))
    assert r["n_valid_total"] == 0
    assert math.isnan(r["mae_mm"]) and math.isnan(r["mean_est_mm"])


def test_summarize_session_without_npy_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        summarize_session(tmp_path / "empty", 1500, roi=(0, 0, 2, 2))


# ---------- 実行ログの E2E 推定 ----------

def test_parse_runner_log(tmp_path):
    log = tmp_path / "dist_1500.txt"
    log.write_text(
        "2026-09-15 10:00:00,000 realtime.runner INFO runner 起動\n"
        "2026-09-15 10:00:01,000 realtime.runner INFO processed=0 (frame id=5): 12.0 ms (~83.33 fps), valid=100\n"
        "junk line without pattern\n"
        "uv : 2026-09-15 10:00:03,000 realtime.runner INFO processed=60 (frame id=71): 12.0 ms (~83.33 fps), valid=100\n",
        encoding="utf-8")
    s = parse_runner_log(log)
    assert s is not None
    assert s["log_span_s"] == pytest.approx(2.0)
    assert s["e2e_fps"] == pytest.approx(30.0)
    assert s["camera_fps_est"] == pytest.approx(33.0)
    assert s["drop_ratio"] == pytest.approx(1.0 - 60 / 66)


def test_parse_runner_log_needs_two_lines(tmp_path):
    log = tmp_path / "one.txt"
    log.write_text("2026-09-15 10:00:01,000 realtime.runner INFO processed=0 (frame id=5): x\n", encoding="utf-8")
    assert parse_runner_log(log) is None
    assert parse_runner_log(tmp_path / "missing.txt") is None


# ---------- manifest ----------

def test_load_manifest(tmp_path):
    m = tmp_path / "m.csv"
    m.write_text("session,gt_mm,note\nrunA,1500,board A\nrunB,3100,\n", encoding="utf-8")
    assert load_manifest(m) == [("runA", 1500, "board A"), ("runB", 3100, "")]
    bad = tmp_path / "bad.csv"
    bad.write_text("session,gt_mm\nrunA,abc\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(bad)
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "none.csv")


# ---------- main（exit code と CSV） ----------

def test_main_writes_csv_and_returns_0(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path)
    h, w = 4, 6
    _write_session(tmp_path, "dist_2000", [_frame(h, w, {(0, 0): 2010.0}), _frame(h, w, {(1, 1): 1990.0})])
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "dist_2000.txt").write_text(
        "2026-09-15 10:00:00,000 realtime.runner INFO processed=0 (frame id=0): x\n"
        "2026-09-15 10:00:02,000 realtime.runner INFO processed=60 (frame id=60): x\n", encoding="utf-8")
    out = tmp_path / "result.csv"

    # ROI は合成フレームの形状に合わせる（config.IMAGE_SIZE ではなく明示指定）
    rc = main(["--sessions", "dist_2000", "--roi", f"0,0,{w},{h}", "--log-dir", str(tmp_path / "logs"),
               "--out", str(out)])

    assert rc == 0
    with out.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    r = rows[0]
    assert r["session"] == "dist_2000" and r["gt_mm"] == "2000"
    assert int(r["n_valid_total"]) == 2
    assert float(r["mae_mm"]) == pytest.approx(10.0)
    assert float(r["mean_err_mm"]) == pytest.approx(0.0)
    assert float(r["e2e_fps"]) == pytest.approx(30.0)
    assert float(r["drop_ratio"]) == pytest.approx(0.0)
    assert r["params_path"] == "params/fit_result.csv"


def test_main_returns_1_when_session_has_no_valid_pixels(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path)
    _write_session(tmp_path, "dist_1500", [np.full((2, 2), np.nan)])
    assert main(["--sessions", "dist_1500", "--roi", "0,0,2,2", "--out", str(tmp_path / "r.csv")]) == 1


def test_main_returns_2_on_bad_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path)
    assert main(["--sessions", "dist_1500", "--out", str(tmp_path / "r.csv")]) == 2     # セッション不在
    assert main(["--sessions", "noname", "--out", str(tmp_path / "r.csv")]) == 2        # 正解距離を解釈できない
    assert main(["--sessions", "dist_1500", "--roi", "1,2,3", "--out", str(tmp_path / "r.csv")]) == 2  # ROI 不正
    assert main(["--manifest", str(tmp_path / "none.csv"), "--out", str(tmp_path / "r.csv")]) == 2   # manifest 不在
