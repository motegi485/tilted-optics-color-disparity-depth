"""checks/verify_clip_replay.py の HW 非依存テスト（合成 numpy 配列・tmp_path のみ）。"""
import json

import cv2
import numpy as np

import config
from checks.verify_clip_replay import _compare, main
from core.params import load_params
from realtime.camera import preprocess
from realtime.pipeline import process_frame


def test_compare_dtype_mismatch_returns_false():
    """live が float64、recomputed が float32 で数値が同一でも dtype 不一致で FAIL 扱い
    （canonical な depth は float32 のため）。"""
    live = np.array([[1.0, 2.0]], dtype=np.float64)
    recomputed = live.astype(np.float32)

    ok, msg = _compare(live, recomputed)

    assert ok is False
    assert "dtype" in msg


def test_compare_exact_match_returns_true():
    live = np.array([[1.0, np.nan, 3.0]], dtype=np.float32)
    recomputed = live.copy()

    ok, _msg = _compare(live, recomputed)

    assert ok is True


def test_compare_both_all_nan_returns_false():
    """両方全 NaN は空集合同士の比較で match してしまう穴を無条件 FAIL で塞ぐガード。"""
    live = np.full((2, 2), np.nan, dtype=np.float32)
    recomputed = np.full((2, 2), np.nan, dtype=np.float32)

    ok, _msg = _compare(live, recomputed)

    assert ok is False


# ===== main: PNG と npy の対応（片方だけ欠けたフレームを見逃さない） =====

_PARAMS = {"v_pos": 432599.33, "u_pos": -63.93, "w_pos": 0.0, "a_pos": -26.57,
           "v_neg": 423627.24, "u_neg": -21.37, "w_neg": 0.0, "a_neg": -22.56}


def _make_session(tmp_path, monkeypatch, png_idx, npy_idx):
    """連番を指定して合成セッションを作る。npy は本物の preprocess → process_frame の出力（差し替えなし）。"""
    monkeypatch.setattr(config, "CAPTURES_DIR", tmp_path / "captures")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    cap, out = tmp_path / "captures" / "s", tmp_path / "outputs" / "s"
    cap.mkdir(parents=True)
    out.mkdir(parents=True)
    params_csv = tmp_path / "fit_result.csv"
    params_csv.write_text("param,value\n" + "".join(f"{k},{v}\n" for k, v in _PARAMS.items()), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({"params_path": str(params_csv)}), encoding="utf-8")
    w, h = config.WEBCAM_NATIVE_SIZE
    tex = cv2.GaussianBlur(np.random.default_rng(0).integers(0, 256, size=(h, w), dtype=np.uint8), (7, 7), 0)
    raw = np.dstack([np.roll(tex, 2, axis=1), tex, tex])        # 青を赤から水平にずらし、推論帯に入る視差を作る
    depth = process_frame(preprocess(raw), load_params(params_csv))
    assert np.count_nonzero(~np.isnan(depth)) > 0
    for i in png_idx:
        cv2.imwrite(str(cap / f"{i:06d}.png"), raw)
    for i in npy_idx:
        np.save(str(out / f"{i:06d}.npy"), depth)


def test_main_passes_when_png_and_npy_pair_up(tmp_path, monkeypatch):
    _make_session(tmp_path, monkeypatch, png_idx=[0], npy_idx=[0])

    assert main(["--session", "s"]) == 0


def test_main_fails_on_npy_without_raw_png(tmp_path, monkeypatch, capsys):
    """raw PNG の無い npy を母数から漏らさない（PNG だけを列挙すると残りの 1/1 で PASS してしまう）。"""
    _make_session(tmp_path, monkeypatch, png_idx=[1], npy_idx=[0, 1])

    assert main(["--session", "s"]) == 1
    out = capsys.readouterr().out
    assert "000000: 対応する raw png がありません" in out
    assert "1/2 フレーム一致" in out


# ===== main: 録画時の条件との食い違いの警告（判定は変えない） =====

def test_main_warns_when_recorded_config_differs(tmp_path, monkeypatch, capsys):
    """meta.json の config が現行と違えば [WARN] で知らせる（別設定で録画したセッションの不一致を回帰と誤読しない）。"""
    _make_session(tmp_path, monkeypatch, png_idx=[0], npy_idx=[0])
    meta_path = tmp_path / "outputs" / "s" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["config"] = {"KERNEL_SIZE": 17}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    assert main(["--session", "s"]) == 0          # 判定は従来どおり（警告だけ）
    out = capsys.readouterr().out
    assert "[WARN]" in out
    assert f"config.KERNEL_SIZE: 録画時=17 現行={config.KERNEL_SIZE}" in out


def test_main_no_warning_when_recorded_conditions_match(tmp_path, monkeypatch, capsys):
    _make_session(tmp_path, monkeypatch, png_idx=[0], npy_idx=[0])
    meta_path = tmp_path / "outputs" / "s" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in (
        (k, getattr(config, k)) for k in ("IMAGE_SIZE", "KERNEL_SIZE", "SEARCH_RANGE_X", "EDGE_TH",
                                          "DISP_MAX", "INFER_BAND_LOW", "INFER_BAND_HIGH"))}
    meta["params"] = dict(_PARAMS)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    assert main(["--session", "s"]) == 0
    assert "[WARN]" not in capsys.readouterr().out
