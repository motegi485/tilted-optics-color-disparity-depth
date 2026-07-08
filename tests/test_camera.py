"""realtime/camera.py の HW 非依存テスト（合成 numpy 配列・tmp_path・モック cap のみ）。"""
import numpy as np
import pytest
import cv2

import config
from realtime.camera import (
    FrameSource,
    StillImageSource,
    VideoFileSource,
    WebcamSource,
    preprocess,
    _verify_native,
)


NATIVE_W, NATIVE_H = config.WEBCAM_NATIVE_SIZE


def _raw(fill=0) -> np.ndarray:
    return np.full((NATIVE_H, NATIVE_W, 3), fill, dtype=np.uint8)


# ===== preprocess =====

def test_preprocess_output_shape_and_dtype():
    out = preprocess(_raw())
    assert out.shape == (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0], 3)
    assert out.dtype == np.uint8


def test_preprocess_wrong_size_raises():
    bad = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        preprocess(bad)


def test_preprocess_crop_wider_than_raw_raises(monkeypatch):
    """D-2: WEBCAM_CROP_W が raw 幅を超える設定だと負インデックス crop になるため raise。"""
    monkeypatch.setattr(config, "WEBCAM_CROP_W", NATIVE_W + 100)
    with pytest.raises(ValueError):
        preprocess(_raw())


def test_preprocess_crop_excludes_outside_columns():
    """中央クロップの範囲外(左右)を変えても出力が変わらないこと(x0=(1280-1080)//2=100 の検証)。"""
    base = _raw(fill=0)
    baseline = preprocess(base)

    left_outside = base.copy()
    left_outside[:, :100] = 255            # クロップ範囲外(左)
    assert np.array_equal(preprocess(left_outside), baseline)

    right_outside = base.copy()
    right_outside[:, 1180:] = 255          # クロップ範囲外(右)
    assert np.array_equal(preprocess(right_outside), baseline)


def test_preprocess_crop_includes_boundary_column():
    """クロップ範囲内(x=100、左端)を変えると出力が変わること。"""
    base = _raw(fill=0)
    baseline = preprocess(base)

    inside = base.copy()
    inside[:, 100:101] = 255               # クロップ範囲内の最左列
    assert not np.array_equal(preprocess(inside), baseline)


# ===== _verify_native =====

def test_verify_native_ok():
    _verify_native(_raw(), "test")   # raise しなければ OK


def test_verify_native_wrong_size_raises():
    bad = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        _verify_native(bad, "test")


# ===== FrameSource: context manager 契約 =====

class _DummySource(FrameSource):
    def __init__(self):
        self.opened = False
        self.released = False

    def open(self):
        self.opened = True

    def read(self):
        return True, _raw()

    def release(self):
        self.released = True


def test_framesource_context_manager_normal_exit():
    src = _DummySource()
    with src as s:
        assert s.opened is True
        assert src.released is False
    assert src.released is True


def test_framesource_context_manager_releases_on_exception():
    src = _DummySource()
    with pytest.raises(ValueError):
        with src:
            raise ValueError("boom")
    assert src.released is True


# ===== StillImageSource =====

def test_stillimagesource_returns_copy_not_shared_reference(tmp_path):
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), _raw(fill=10))

    src = StillImageSource(path)
    src.open()
    _, frame1 = src.read()
    frame1[:] = 0                          # 返却されたコピーを破壊しても...
    _, frame2 = src.read()
    assert frame2.mean() != 0              # ...内部保持データは無事であること


def test_stillimagesource_missing_file_raises(tmp_path):
    src = StillImageSource(tmp_path / "missing.png")
    with pytest.raises(FileNotFoundError):
        src.open()


def test_stillimagesource_wrong_size_raises(tmp_path):
    path = tmp_path / "wrong.png"
    cv2.imwrite(str(path), np.zeros((480, 640, 3), dtype=np.uint8))
    src = StillImageSource(path)
    with pytest.raises(ValueError):
        src.open()


# ===== VideoFileSource =====

def test_videofilesource_missing_raises(tmp_path):
    src = VideoFileSource(tmp_path / "missing.mp4")
    with pytest.raises(FileNotFoundError):
        src.open()


def test_videofilesource_double_open_raises(tmp_path):
    """D-2: 多重 open はハンドルリーク防止のため raise。"""
    src = VideoFileSource(tmp_path / "whatever.mp4")
    src.cap = object()   # 既に open 済みを装う
    with pytest.raises(RuntimeError):
        src.open()


# ===== WebcamSource =====

def test_webcamsource_read_before_open_raises():
    src = WebcamSource()
    with pytest.raises(RuntimeError):
        src.read()


def test_webcamsource_release_is_idempotent():
    src = WebcamSource()
    src.release()   # open していない状態で release しても例外を出さない
    src.release()


def test_webcamsource_double_open_raises():
    """D-2: 多重 open はハンドルリーク防止のため raise。"""
    src = WebcamSource()
    src.cap = object()   # 既に open 済みを装う
    with pytest.raises(RuntimeError):
        src.open()


class _StaticCap:
    """毎回まったく同一のフレームを返す(no-feed プレースホルダ相当)。"""

    def __init__(self, shape=(10, 10, 3)):
        self._frame = np.full(shape, 77, dtype=np.uint8)

    def read(self):
        return True, self._frame.copy()


class _NoiseCap:
    """毎回異なる乱数フレームを返す(実フィード相当)。"""

    def __init__(self, shape=(10, 10, 3)):
        self._rng = np.random.default_rng(0)
        self._shape = shape

    def read(self):
        return True, self._rng.integers(0, 256, size=self._shape, dtype=np.uint8)


def test_wait_until_live_static_feed_times_out():
    src = WebcamSource()
    src.cap = _StaticCap()
    with pytest.raises(RuntimeError):
        src._wait_until_live(timeout_s=0.05)


def test_wait_until_live_noise_feed_returns_frame():
    src = WebcamSource()
    src.cap = _NoiseCap()
    frame = src._wait_until_live(timeout_s=2.0)
    assert frame is not None
