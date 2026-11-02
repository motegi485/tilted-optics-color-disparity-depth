"""checks/evaluate.py::main の HW 非依存テスト（合成 JPG・tmp_path・monkeypatch のみ。
実アンカー画像 2300.jpg には依存しない）。
"""
import cv2
import numpy as np

import config
from checks.evaluate import main


def _make_pattern(width: int, height: int, edge_col: int, seed: int = 42) -> np.ndarray:
    """core.disparity が検出できる縦エッジ1本を持つ列パターン（tests/test_core.py と同方式）。"""
    rng = np.random.default_rng(seed)
    base = 100 + rng.integers(-3, 4, size=width)
    base = base.astype(np.int32)
    base[edge_col:] += 150
    pattern = np.clip(base, 0, 255).astype(np.uint8)
    return np.broadcast_to(pattern, (height, width)).copy()


def _write_unregistered_jpg(path) -> None:
    """EXPECTED に無い数値名で、有効視差が出る程度の縦エッジを持つ合成 JPG を書く。"""
    width, height = config.IMAGE_SIZE
    shift = 3
    edge_col = width // 2
    red = _make_pattern(width, height, edge_col)
    blue = _make_pattern(width, height, edge_col - shift)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = blue   # BGR順のB
    img[:, :, 2] = red    # BGR順のR
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def test_main_returns_2_when_no_images(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TEST_DIR", tmp_path)
    assert main() == 2


def test_main_returns_2_when_only_unregistered_anchor_present(tmp_path, monkeypatch):
    """EXPECTED に無い数値名 JPG しか無いと、照合0件のまま空振り成功にせず 2 を返す。"""
    monkeypatch.setattr(config, "TEST_DIR", tmp_path)
    _write_unregistered_jpg(tmp_path / "9999.jpg")
    assert main() == 2


def test_main_finds_uppercase_jpg_extension(tmp_path, monkeypatch, capsys):
    """拡張子が大文字（実アンカーは 2300.JPG）でも、大小を区別するファイルシステムで検出できる（2026-11-02 修正）。"""
    monkeypatch.setattr(config, "TEST_DIR", tmp_path)
    _write_unregistered_jpg(tmp_path / "9999.JPG")
    assert main() == 2                       # 照合 0 件の 2（画像なしの 2 ではない）
    assert "9999.JPG" in capsys.readouterr().out
