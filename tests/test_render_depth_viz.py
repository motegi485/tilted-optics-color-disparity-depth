"""tools/render_depth_viz.py の HW 非依存テスト（合成 npy・tmp_path のみ）。"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.render_depth_viz import main, range_tag, render_session


def _write_session(root: Path, name: str, depths: list[np.ndarray]) -> Path:
    """runner の保存レイアウト（連番 npy＋meta.json）を tmp_path に再現する。"""
    d = root / name
    d.mkdir(parents=True)
    for i, arr in enumerate(depths):
        np.save(str(d / f"{i:06d}.npy"), arr)
    (d / "meta.json").write_text("{}", encoding="utf-8")
    return d


def _plane(z: float, shape=(352, 528)) -> np.ndarray:
    return np.full(shape, z, dtype=np.float32)


def _files(d: Path) -> dict:
    return {p.name: p.stat().st_mtime_ns for p in d.iterdir() if p.is_file()}


def test_range_tag_drops_trailing_zeros():
    assert range_tag(1000.0, 4000.0) == "1000-4000"
    assert range_tag(0, 50000) == "0-50000"


def test_render_writes_png_per_npy_and_does_not_touch_source(tmp_path):
    d = _write_session(tmp_path, "s", [_plane(1300.0), _plane(3300.0)])
    before = _files(d)
    res = render_session(d, 1000.0, 4000.0)
    assert res["written"] == 2 and res["skipped"] == 0
    assert res["out_dir"] == d / "viz_1000-4000"
    assert sorted(p.name for p in res["out_dir"].iterdir()) == ["000000_viz.png", "000001_viz.png"]
    assert _files(d) == before                      # 元の npy／meta.json は触らない
    # 1000–4000 mm なら 1300 mm は近い側（赤）・3300 mm は遠い側（青寄り）に分かれる。BGR 順。
    near = cv2.imread(str(res["out_dir"] / "000000_viz.png"))
    far = cv2.imread(str(res["out_dir"] / "000001_viz.png"))
    assert near[10, 10, 2] > near[10, 10, 0]        # R > B
    assert far[10, 10, 0] > far[10, 10, 2]          # B > R
    # 同じ 2 枚を 0–50000 mm で描くと両方とも赤に潰れる（本ツールを作った動機）
    res2 = render_session(d, 0.0, 50000.0)
    assert res2["out_dir"] == d / "viz_0-50000"
    near2 = cv2.imread(str(res2["out_dir"] / "000000_viz.png"))
    far2 = cv2.imread(str(res2["out_dir"] / "000001_viz.png"))
    assert near2[10, 10, 2] > near2[10, 10, 0] and far2[10, 10, 2] > far2[10, 10, 0]


def test_nan_black_index_filter_skip_and_overwrite(tmp_path):
    arr = _plane(2000.0)
    arr[0, 0] = np.nan
    d = _write_session(tmp_path, "s", [arr, arr, arr])
    res = render_session(d, 1000.0, 4000.0, indices=[2], colorbar=False)
    assert res["written"] == 1 and res["skipped"] == 0
    assert not (res["out_dir"] / "000000_viz.png").exists()
    img = cv2.imread(str(res["out_dir"] / "000002_viz.png"))
    assert tuple(img[0, 0]) == (0, 0, 0)            # NaN は黒（to_colormap と同じ）
    assert img[10, 10, 2] > 0                       # 有効画素は黒ではない
    res = render_session(d, 1000.0, 4000.0, indices=[2], colorbar=False)
    assert res["written"] == 0 and res["skipped"] == 1
    res = render_session(d, 1000.0, 4000.0, indices=[2], colorbar=False, overwrite=True)
    assert res["written"] == 1 and res["skipped"] == 0


def test_invalid_inputs_raise(tmp_path):
    d = _write_session(tmp_path, "s", [_plane(2000.0)])
    with pytest.raises(ValueError):
        render_session(d, 4000.0, 1000.0)
    with pytest.raises(FileNotFoundError):
        render_session(tmp_path / "nope", 1000.0, 4000.0)
    with pytest.raises(FileNotFoundError):
        render_session(d, 1000.0, 4000.0, indices=[5])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        render_session(empty, 1000.0, 4000.0)


def test_cli_multi_session_out_dir_and_exit_codes(tmp_path):
    _write_session(tmp_path, "a", [_plane(1500.0)])
    _write_session(tmp_path, "b", [_plane(2500.0)])
    rc = main(["--session", "a", "b", "--vmin", "1000", "--vmax", "4000",
               "--outputs-dir", str(tmp_path), "--out-dir", str(tmp_path / "o")])
    assert rc == 0
    assert (tmp_path / "o" / "a" / "000000_viz.png").exists()
    assert (tmp_path / "o" / "b" / "000000_viz.png").exists()
    rc = main(["--session", "a", "--vmin", "1000", "--vmax", "4000", "--outputs-dir", str(tmp_path)])
    assert rc == 0 and (tmp_path / "a" / "viz_1000-4000" / "000000_viz.png").exists()
    assert main(["--session", "a", "--vmin", "4000", "--vmax", "1000", "--outputs-dir", str(tmp_path)]) == 2
    assert main(["--session", "zzz", "--outputs-dir", str(tmp_path)]) == 2
