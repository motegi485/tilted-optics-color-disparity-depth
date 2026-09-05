"""realtime/runner.py の HW 非依存テスト（合成 numpy 配列・tmp_path のみ）。"""
import json

import numpy as np
import pytest

import config
from core.params import DepthParams
from realtime.camera import StillImageSource
from realtime.runner import (
    _ensure_session_dirs,
    _LatestFrameSlot,
    _open_video_writer,
    _session_video_paths,
    _SessionVideoRecorder,
    _write_session_meta,
    draw_colorbar,
    to_colormap,
)


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


# ===== draw_colorbar =====

def test_draw_colorbar_vmax_le_vmin_raises():
    img = np.zeros((352, 528, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        draw_colorbar(img, vmin=100.0, vmax=100.0)
    with pytest.raises(ValueError):
        draw_colorbar(img, vmin=100.0, vmax=50.0)


def test_draw_colorbar_num_ticks_below_2_raises():
    img = np.zeros((352, 528, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        draw_colorbar(img, vmin=0.0, vmax=50000.0, num_ticks=1)


def test_draw_colorbar_too_small_image_raises():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        draw_colorbar(img, vmin=0.0, vmax=50000.0)


def test_draw_colorbar_preserves_shape_and_dtype_and_draws_in_place():
    img = np.zeros((352, 528, 3), dtype=np.uint8)
    before_id = id(img)

    draw_colorbar(img, vmin=0.0, vmax=50000.0)

    assert img.shape == (352, 528, 3)
    assert img.dtype == np.uint8
    assert id(img) == before_id
    assert img.any()                                   # 何かしら描画されている


def test_draw_colorbar_near_red_at_bottom_far_blue_at_top():
    """近距離(赤)をバー下端、遠距離(青)をバー上端に置く仕様（実装時の確認事項でユーザー指定）。"""
    img = np.zeros((352, 528, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    x_in_bar = w - 12                                   # バー内側の列（margin_right=6, bar_w=16 前提）

    draw_colorbar(img, vmin=0.0, vmax=50000.0)

    top_color = img[20, x_in_bar]                       # 上端寄り
    bottom_color = img[h - 20, x_in_bar]                 # 下端寄り
    # BGR 順。赤は R(index2) が高く B(index0) が低い／青はその逆（to_colormap と同じ反転JET規約）。
    assert int(bottom_color[2]) > int(bottom_color[0])   # 下端: 赤寄り(近距離)
    assert int(top_color[0]) > int(top_color[2])         # 上端: 青寄り(遠距離)


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


@pytest.mark.parametrize("bad_session", ["..", "a/b", "a\\b", ""])
def test_ensure_session_dirs_rejects_unsafe_name(bad_session, tmp_path, monkeypatch):
    """`..`・区切り文字混入・空文字は単一ディレクトリ名でないため raise（2026-09 レビュー）。"""
    monkeypatch.setattr(config, "CAPTURES_DIR", tmp_path / "captures")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")

    with pytest.raises(ValueError):
        _ensure_session_dirs(bad_session)


def test_ensure_session_dirs_rejects_absolute_path(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CAPTURES_DIR", tmp_path / "captures")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")

    with pytest.raises(ValueError):
        _ensure_session_dirs(str(tmp_path))


def test_ensure_session_dirs_normal_name_still_succeeds(tmp_path, monkeypatch):
    """単一ディレクトリ名ガード追加後も、通常名では従来どおり成功すること。"""
    monkeypatch.setattr(config, "CAPTURES_DIR", tmp_path / "captures")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")

    captures, outputs = _ensure_session_dirs("ok_session")

    assert captures.is_dir()
    assert outputs.is_dir()


def test_ensure_session_dirs_existing_empty_dir_is_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CAPTURES_DIR", tmp_path / "captures")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    (tmp_path / "captures" / "empty_session").mkdir(parents=True)
    (tmp_path / "outputs" / "empty_session").mkdir(parents=True)

    captures, outputs = _ensure_session_dirs("empty_session")

    assert captures.is_dir()
    assert outputs.is_dir()


# ===== セッション動画（§7.3）: _LatestFrameSlot =====

def test_latest_frame_slot_starts_empty_and_returns_updated_reference():
    slot = _LatestFrameSlot()
    assert slot.latest() is None

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    slot.update(frame)

    assert slot.latest() is frame


# ===== セッション動画: _session_video_paths =====

def test_session_video_paths_uses_config_names(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSION_VIDEO_RAW_NAME", "custom_raw.mp4")
    monkeypatch.setattr(config, "SESSION_VIDEO_DEPTH_NAME", "custom_depth.mp4")
    captures_dir = tmp_path / "captures" / "s"
    outputs_dir = tmp_path / "outputs" / "s"

    raw_path, depth_path = _session_video_paths(captures_dir, outputs_dir)

    assert raw_path == captures_dir / "custom_raw.mp4"
    assert depth_path == outputs_dir / "custom_depth.mp4"


# ===== セッション動画: フェイク VideoWriter によるテストダブル =====

class _FakeVideoWriter:
    """cv2.VideoWriter の最小フェイク。実コーデック・実ファイル I/O 無しでテストする。"""

    def __init__(self, opened: bool = True, write_ok: bool = True):
        self._opened = opened
        self.write_ok = write_ok
        self.written: list[np.ndarray] = []
        self.released = 0

    def isOpened(self) -> bool:
        return self._opened

    def write(self, frame: np.ndarray) -> bool:
        if self.write_ok:
            self.written.append(frame)
        return self.write_ok

    def release(self) -> None:
        self.released += 1


def _fake_writer_factory(writers):
    """あらかじめ用意した fake writer 群を、呼び出し順に1つずつ返す factory。"""
    it = iter(writers)

    def factory(_path, _fourcc, _fps, _frame_size):
        return next(it)

    return factory


class _StubGrabber:
    """_FrameGrabber の latest()/stopped だけを模した最小スタブ。"""

    def __init__(self, raw: np.ndarray | None, frame_id: int = 0, stopped: bool = False):
        self._raw = raw
        self._frame_id = frame_id
        self.stopped = stopped

    def latest(self):
        return self._raw, self._frame_id


def _make_recorder(tmp_path, *, raw_writer=None, depth_writer=None, fps=30.0,
                    frame_size=config.IMAGE_SIZE) -> _SessionVideoRecorder:
    raw_writer = raw_writer if raw_writer is not None else _FakeVideoWriter()
    depth_writer = depth_writer if depth_writer is not None else _FakeVideoWriter()
    return _SessionVideoRecorder(
        tmp_path / "video_raw.mp4", tmp_path / "video_depth.mp4", _LatestFrameSlot(),
        fps=fps, fourcc="mp4v", frame_size=frame_size,
        writer_factory=_fake_writer_factory([raw_writer, depth_writer]))


# ===== セッション動画: _open_video_writer =====

def test_open_video_writer_raises_and_releases_when_not_opened(tmp_path):
    fake = _FakeVideoWriter(opened=False)

    with pytest.raises(RuntimeError):
        _open_video_writer(tmp_path / "x.mp4", fps=30.0, fourcc="mp4v", frame_size=(528, 352),
                            writer_factory=lambda *a, **k: fake)

    assert fake.released == 1


# ===== セッション動画: _SessionVideoRecorder =====

def test_session_video_recorder_init_rejects_nonpositive_fps(tmp_path):
    with pytest.raises(ValueError):
        _make_recorder(tmp_path, fps=0.0)


def test_session_video_recorder_init_releases_first_writer_if_second_fails(tmp_path):
    good = _FakeVideoWriter(opened=True)
    bad = _FakeVideoWriter(opened=False)

    with pytest.raises(RuntimeError):
        _make_recorder(tmp_path, raw_writer=good, depth_writer=bad)

    assert good.released == 1


def test_start_rejects_grabber_without_required_interface(tmp_path):
    rec = _make_recorder(tmp_path)
    with pytest.raises(ValueError):
        rec.start(object())


def test_write_one_tick_skips_both_when_no_raw_frame(tmp_path):
    raw_w, depth_w = _FakeVideoWriter(), _FakeVideoWriter()
    rec = _make_recorder(tmp_path, raw_writer=raw_w, depth_writer=depth_w)
    rec._grabber = _StubGrabber(raw=None)

    rec._write_one_tick()

    assert raw_w.written == []
    assert depth_w.written == []


def test_write_one_tick_writes_preprocessed_raw_and_black_when_slot_empty(tmp_path):
    raw_w, depth_w = _FakeVideoWriter(), _FakeVideoWriter()
    rec = _make_recorder(tmp_path, raw_writer=raw_w, depth_writer=depth_w)
    raw = np.zeros((*reversed(config.WEBCAM_NATIVE_SIZE), 3), dtype=np.uint8)
    rec._grabber = _StubGrabber(raw=raw)

    rec._write_one_tick()

    assert len(raw_w.written) == 1
    assert raw_w.written[0].shape == (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0], 3)  # preprocess 後の形状
    assert len(depth_w.written) == 1
    np.testing.assert_array_equal(
        depth_w.written[0], np.zeros((config.IMAGE_SIZE[1], config.IMAGE_SIZE[0], 3), dtype=np.uint8))


def test_write_one_tick_sample_and_holds_latest_display_frame(tmp_path):
    raw_w, depth_w = _FakeVideoWriter(), _FakeVideoWriter()
    slot = _LatestFrameSlot()
    rec = _SessionVideoRecorder(
        tmp_path / "video_raw.mp4", tmp_path / "video_depth.mp4", slot,
        fps=30.0, fourcc="mp4v", frame_size=config.IMAGE_SIZE,
        writer_factory=_fake_writer_factory([raw_w, depth_w]))
    raw = np.zeros((*reversed(config.WEBCAM_NATIVE_SIZE), 3), dtype=np.uint8)
    rec._grabber = _StubGrabber(raw=raw)
    color = np.full((config.IMAGE_SIZE[1], config.IMAGE_SIZE[0], 3), 7, dtype=np.uint8)
    slot.update(color)

    rec._write_one_tick()
    rec._write_one_tick()

    assert len(depth_w.written) == 2
    assert depth_w.written[0] is color        # サンプル&ホールド: 新しい絵が来るまで同一参照を繰り返す
    assert depth_w.written[1] is color


def test_write_one_tick_raises_when_writer_write_fails(tmp_path):
    raw_w = _FakeVideoWriter(write_ok=False)
    rec = _make_recorder(tmp_path, raw_writer=raw_w)
    raw = np.zeros((*reversed(config.WEBCAM_NATIVE_SIZE), 3), dtype=np.uint8)
    rec._grabber = _StubGrabber(raw=raw)

    with pytest.raises(RuntimeError):
        rec._write_one_tick()


def test_raise_if_error_and_log_error_if_any(tmp_path):
    rec = _make_recorder(tmp_path)
    rec._error = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        rec.raise_if_error()

    rec.log_error_if_any()   # raise せず記録のみ（例外にならないことを確認）


def test_ensure_released_is_idempotent(tmp_path):
    raw_w, depth_w = _FakeVideoWriter(), _FakeVideoWriter()
    rec = _make_recorder(tmp_path, raw_writer=raw_w, depth_writer=depth_w)

    rec.ensure_released()
    rec.ensure_released()

    assert raw_w.released == 1
    assert depth_w.released == 1


def test_start_and_stop_and_join_releases_writers(tmp_path):
    raw_w, depth_w = _FakeVideoWriter(), _FakeVideoWriter()
    rec = _make_recorder(tmp_path, raw_writer=raw_w, depth_writer=depth_w, fps=1000.0)
    grabber = _StubGrabber(raw=None, stopped=False)

    rec.start(grabber)
    rec.stop_and_join(timeout=1.0)

    assert raw_w.released == 1
    assert depth_w.released == 1


# ===== セッション動画: _write_session_meta の session_video キー =====

def _dummy_params() -> DepthParams:
    return DepthParams(v_pos=1.0, u_pos=1.0, a_pos=1.0, w_pos=0.0,
                       v_neg=1.0, u_neg=1.0, a_neg=1.0, w_neg=0.0)


def test_write_session_meta_records_session_video_disabled_by_default(tmp_path):
    _write_session_meta(tmp_path, _dummy_params(), None, StillImageSource(tmp_path / "x.png"))

    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["session_video"] == {"enabled": False}


def test_write_session_meta_records_session_video_enabled_details(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSION_VIDEO_FPS", 15.0)
    monkeypatch.setattr(config, "SESSION_VIDEO_FOURCC", "mp4v")
    monkeypatch.setattr(config, "SESSION_VIDEO_RAW_NAME", "video_raw.mp4")
    monkeypatch.setattr(config, "SESSION_VIDEO_DEPTH_NAME", "video_depth.mp4")

    _write_session_meta(tmp_path, _dummy_params(), None, StillImageSource(tmp_path / "x.png"),
                        record_video=True)

    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["session_video"] == {
        "enabled": True, "fps": 15.0, "fourcc": "mp4v",
        "raw": "video_raw.mp4", "depth": "video_depth.mp4",
    }
