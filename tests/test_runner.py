"""realtime/runner.py の HW 非依存テスト（合成 numpy 配列・tmp_path のみ）。"""
import numpy as np
import pytest

import config
from realtime.runner import to_colormap, _ensure_session_dirs


# ===== to_colormap =====

def test_to_colormap_nan_becomes_black_and_input_unchanged():
    depth = np.array([[np.nan, 100.0], [5000.0, 25000.0]], dtype=np.float32)
    depth_copy = depth.copy()

    color = to_colormap(depth, vmin=0.0, vmax=50000.0)

    assert color.shape == (2, 2, 3)
    assert color.dtype == np.uint8
    assert tuple(int(c) for c in color[0, 0]) == (0, 0, 0)   # NaN -> 黒
    np.testing.assert_array_equal(depth, depth_copy)          # 元配列は不変(NaN も == で一致扱い)


def test_to_colormap_finite_values_not_black():
    depth = np.array([[25000.0]], dtype=np.float32)
    color = to_colormap(depth, vmin=0.0, vmax=50000.0)
    assert tuple(int(c) for c in color[0, 0]) != (0, 0, 0)


def test_to_colormap_vmax_le_vmin_raises():
    """D-2: vmax <= vmin は正規化が破綻するため raise。"""
    depth = np.array([[1.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        to_colormap(depth, vmin=100.0, vmax=100.0)
    with pytest.raises(ValueError):
        to_colormap(depth, vmin=100.0, vmax=50.0)


# ===== _ensure_session_dirs (B-5) =====

def test_ensure_session_dirs_creates_new_session(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CAPTURES_DIR", tmp_path / "captures")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")

    captures, outputs = _ensure_session_dirs("new_session")

    assert captures.is_dir()
    assert outputs.is_dir()


def test_ensure_session_dirs_existing_nonempty_raises_and_preserves_content(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CAPTURES_DIR", tmp_path / "captures")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    existing = tmp_path / "captures" / "dup_session"
    existing.mkdir(parents=True)
    marker = existing / "000000.png"
    marker.write_bytes(b"dummy")

    with pytest.raises(RuntimeError):
        _ensure_session_dirs("dup_session")

    assert marker.read_bytes() == b"dummy"     # 中身が1バイトも変化していない
    assert list(existing.iterdir()) == [marker]


def test_ensure_session_dirs_existing_empty_dir_is_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CAPTURES_DIR", tmp_path / "captures")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    (tmp_path / "captures" / "empty_session").mkdir(parents=True)
    (tmp_path / "outputs" / "empty_session").mkdir(parents=True)

    captures, outputs = _ensure_session_dirs("empty_session")

    assert captures.is_dir()
    assert outputs.is_dir()
