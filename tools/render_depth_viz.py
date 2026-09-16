#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/render_depth_viz.py — 保存済みの深度 npy から、表示レンジを変えて viz PNG を再生成する。

背景（2026-09-16）:
  runner が保存する ``outputs/<session>/<idx>_viz.png`` は、撮影時の ``config.DEPTH_MIN``/``DEPTH_MAX``
  （表示レンジ＝gate2 の表示側適用）で固定スケール化した見た目である。表示レンジは ``to_colormap`` の
  vmin/vmax にだけ効き、保存される深度 npy には影響しない（``realtime/pipeline.py``: gate2 は適用しない／
  ``docs/module-realtime.md``「表示・保存」）。2026-09-15 の距離別精度キャンペーンは ``DEPTH_MAX=50000``
  のまま撮影したため 1.3–3.3 m の板が全部「赤」に潰れて見えるが、npy は無事なので再撮影せず本ツールで
  再描画する。``meta.json`` は表示レンジを記録しない（``docs/known-issues.md``）ため、出力フォルダ名に
  レンジを含めて事後追跡できるようにする。

色の割り当ては runner と同じ ``to_colormap``／``draw_colorbar`` を呼ぶ（反転 JET・近いほど赤・遠いほど青・
NaN は黒）。``config.py`` の値は書き換えない（``DEPTH_MAX`` は ``checks.evaluate`` の回帰アンカーと
``checks.evaluate_distance_accuracy`` の gate2 にも使われるため）。**`config.py` に依存する**
（既定の ``OUTPUTS_DIR``・``DEPTH_MIN``/``DEPTH_MAX``）。

実行例（リポジトリ root・PowerShell）:
    uv run python -m tools.render_depth_viz --session dist_2100 --vmin 1000 --vmax 4000
    uv run python -m tools.render_depth_viz --session dist_1300 dist_3300 --vmin 1000 --vmax 4000 --index 0 10
出力: ``outputs/<session>/viz_<vmin>-<vmax>/<idx>_viz.png``（runner が保存した ``*_viz.png`` は上書きしない）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import config
from realtime.runner import draw_colorbar, to_colormap

NPY_GLOB = "[0-9][0-9][0-9][0-9][0-9][0-9].npy"   # runner の連番 npy（6 桁）。meta.json 等は拾わない


def range_tag(vmin: float, vmax: float) -> str:
    """出力フォルダ名用のレンジ表記（1000.0, 4000.0 → ``1000-4000``）。"""
    return f"{vmin:g}-{vmax:g}"


def render_session(session_dir: Path, vmin: float, vmax: float, *, indices: list[int] | None = None,
                   colorbar: bool = True, out_dir: Path | None = None, overwrite: bool = False) -> dict:
    """1 セッション分の連番 npy を読み、``[vmin, vmax]`` 固定スケールの viz PNG を書き出す。

    session_dir: ``outputs/<session>``（連番 npy のあるフォルダ）。元の npy・``*_viz.png`` は触らない。
    indices: 再描画するフレーム index（None なら全フレーム）。存在しない index は raise。
    out_dir: 既定は ``session_dir / f"viz_{range_tag(vmin, vmax)}"``。
    戻り値: ``{"out_dir", "written", "skipped"}``。既存の出力は ``overwrite=False`` ならスキップする。
    """
    if vmax <= vmin:
        raise ValueError(f"render_session: vmax({vmax}) は vmin({vmin}) より大きい必要があります")
    if not session_dir.is_dir():
        raise FileNotFoundError(f"render_session: セッションの outputs が見つかりません: {session_dir}")
    npy_paths = sorted(session_dir.glob(NPY_GLOB))
    if indices is not None:
        wanted = {f"{i:06d}" for i in indices}
        missing = sorted(wanted - {p.stem for p in npy_paths})
        if missing:
            raise FileNotFoundError(f"render_session: 指定 index の npy が無い: {missing}（{session_dir}）")
        npy_paths = [p for p in npy_paths if p.stem in wanted]
    if not npy_paths:
        raise FileNotFoundError(f"render_session: 連番 npy が 1 つも無い: {session_dir}")

    dst_dir = out_dir if out_dir is not None else session_dir / f"viz_{range_tag(vmin, vmax)}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for p in npy_paths:
        dst = dst_dir / f"{p.stem}_viz.png"
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        depth = np.load(str(p))
        if depth.ndim != 2:
            raise ValueError(f"render_session: 深度 npy は 2 次元を想定。実際は shape={depth.shape}（{p}）")
        color = to_colormap(depth.astype(np.float32, copy=False), vmin, vmax)
        if colorbar:
            draw_colorbar(color, vmin, vmax)
        if not cv2.imwrite(str(dst), color):
            raise OSError(f"render_session: 書き込みに失敗: {dst}")
        written += 1
    return {"out_dir": dst_dir, "written": written, "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="保存済み深度 npy から表示レンジを変えて viz PNG を再生成する")
    ap.add_argument("--session", nargs="+", required=True, help="outputs/ 配下のセッション名（複数可）")
    ap.add_argument("--vmin", type=float, default=float(config.DEPTH_MIN),
                    help=f"表示レンジ下限 [mm]（既定 config.DEPTH_MIN={config.DEPTH_MIN}）")
    ap.add_argument("--vmax", type=float, default=float(config.DEPTH_MAX),
                    help=f"表示レンジ上限 [mm]（既定 config.DEPTH_MAX={config.DEPTH_MAX}）")
    ap.add_argument("--index", type=int, nargs="+", default=None,
                    help="再描画するフレーム index（省略時は全フレーム）")
    ap.add_argument("--no-colorbar", action="store_true", help="カラーバーを描かない")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="出力先（既定 outputs/<session>/viz_<vmin>-<vmax>/。複数セッション時は配下に <session>/ を切る）")
    ap.add_argument("--overwrite", action="store_true", help="既存の出力 PNG を上書きする（既定はスキップ）")
    ap.add_argument("--outputs-dir", type=Path, default=config.OUTPUTS_DIR,
                    help="セッションフォルダの親（既定 config.OUTPUTS_DIR。テスト用）")
    args = ap.parse_args(argv)

    if args.vmax <= args.vmin:
        print(f"[ERROR] --vmax({args.vmax:g}) は --vmin({args.vmin:g}) より大きい必要があります", file=sys.stderr)
        return 2
    for session in args.session:
        out_dir = None
        if args.out_dir is not None:
            out_dir = args.out_dir / session if len(args.session) > 1 else args.out_dir
        try:
            res = render_session(args.outputs_dir / session, args.vmin, args.vmax,
                                 indices=args.index, colorbar=not args.no_colorbar,
                                 out_dir=out_dir, overwrite=args.overwrite)
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return 2
        print(f"{session}: range=[{args.vmin:g}, {args.vmax:g}] mm  written={res['written']}  "
              f"skipped={res['skipped']}  -> {res['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
