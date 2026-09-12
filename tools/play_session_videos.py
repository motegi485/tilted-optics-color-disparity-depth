#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tilted-optics-color-disparity-depth: runner.py --record-video で生成した
video_raw.mp4 / video_depth.mp4 を、フレーム同期で「同時再生」する閲覧ツール。

なぜ専用ツールが要るか:
  Windows 標準プレイヤーは 1 プロセス 1 本再生で、2 本を同時に開いても開始点を
  揃えられず、各々が独立クロックで走るため再生位置がドリフトする。本ツールは両動画を
  「同一フレーム番号でロックステップ」に描画するので、開始点が完全に揃い以後もずれない
  （＝生映像と深度マップを並べて計算遅延を見る、という録画目的に一致・docs/module-realtime.md）。

同期の根拠:
  2 本は設計上フレーム数・fps が一致する（runner の 1 tick = 両方書く/両方スキップの
  不変条件＋depth のサンプル&ホールド）。保険として min(frame_count) までを対象にし、
  差異があれば警告する。

依存: opencv-python, numpy のみ（uv 環境の既存依存で動く）。

実行例（PowerShell / uv・Windows 11）:
    uv run python tools/play_session_videos.py                         # 最新セッションを自動選択
    uv run python tools/play_session_videos.py --session 20260710_101530
    uv run python tools/play_session_videos.py --raw path\to\video_raw.mp4 --depth path\to\video_depth.mp4
    uv run python tools/play_session_videos.py --session 20260710_101530 --export  # 合成を1本のmp4に書き出して終了

操作:
    SPACE     再生 / 一時停止
    . / ,     1 フレーム進む / 戻る（自動で一時停止）
    [ / ]     再生速度を下げる / 上げる（0.25x〜4x）
    r         先頭に戻す
    l         ループ ON/OFF
    トラックバー  シーク
    q / ESC   終了

書き出し（--export）:
    再生ではなく、同時再生している合成（raw|depth をラベル付きで横連結。操作ヒント行は付けない）を
    そのまま 1 本の mp4 に書き出す。GUI は開かず、全フレームを native fps（等倍＝実時間）で決定論的に
    書き出して終了するため、再生速度・一時停止・シークの影響を受けず毎回同一結果になる（ヘッドレス可）。
    出力は既定で outputs/{session}/video_combined.mp4（--raw/--depth 指定時は depth と同じディレクトリ）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# config の既存レイアウトに合わせる（root/data/captures, root/outputs）。config は import しない
# ——本ツールは repo 外に置いても単体で動くようにするため、パス規約だけを内側に持つ。
CAPTURES_SUBDIR = ("data", "captures")
OUTPUTS_SUBDIR = ("outputs",)
RAW_NAME = "video_raw.mp4"
DEPTH_NAME = "video_depth.mp4"
COMBINED_NAME = "video_combined.mp4"  # --export の既定出力名（depth と同じディレクトリに置く）
# 記録側（config.SESSION_VIDEO_FOURCC）と同じコーデック。OpenCV 同梱 FFmpeg で追加 DLL 不要。
# 本ツールは config を import しない方針（repo 外でも単体で動くため）なので規約値だけを持つ。
EXPORT_FOURCC = "mp4v"
# 記録側（config.SESSION_VIDEO_FPS）と同じ規約値。fps が異なる動画（他所由来など）を
# 無警告で等倍(native fps)再生扱いしないための照合用（EXPORT_FOURCC と同じ理由で config は import しない）。
EXPECTED_FPS = 30.0
FPS_MISMATCH_TOL = 0.5  # 許容差（fps）。コンテナ側の丸め誤差程度は許容する

WINDOW = "tilted-optics-color-disparity-depth : raw | depth  (frame-locked)"
SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]


def _default_root() -> Path:
    # tools/ 直下に置かれる前提。見つからなければ cwd にフォールバック。
    here = Path(__file__).resolve().parent
    for cand in (here.parent, here, Path.cwd()):
        if (cand / OUTPUTS_SUBDIR[0]).is_dir():
            return cand
    return here.parent


def _find_latest_session(root: Path) -> str | None:
    # outputs/*/video_depth.mp4 のうち最終更新が最も新しいものをセッションとみなす。
    outputs = root.joinpath(*OUTPUTS_SUBDIR)
    if not outputs.is_dir():
        return None
    candidates = [p for p in outputs.glob(f"*/{DEPTH_NAME}") if p.is_file()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.parent.name


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.raw and args.depth:
        return Path(args.raw), Path(args.depth)
    root = Path(args.root) if args.root else _default_root()
    session = args.session or _find_latest_session(root)
    if session is None:
        raise SystemExit(
            f"セッションが見つかりません（{root.joinpath(*OUTPUTS_SUBDIR)} 配下に "
            f"*/{DEPTH_NAME} が無い）。--session か --raw/--depth を指定してください。"
        )
    raw = root.joinpath(*CAPTURES_SUBDIR, session, RAW_NAME)
    depth = root.joinpath(*OUTPUTS_SUBDIR, session, DEPTH_NAME)
    return raw, depth


class SyncPair:
    """2 本の VideoCapture をフレーム番号でロックステップに読むラッパ。

    連番読み出し（i == pos+1）は grab/retrieve で高速に、シーク（トラックバー・逆送り）は
    set(POS_FRAMES) で行う。mp4v はキーフレーム間引きのため set() が厳密フレーム精度に
    ならない場合があるが、閲覧・遅延確認の用途では実害はない。
    """

    def __init__(self, raw_path: Path, depth_path: Path):
        for p in (raw_path, depth_path):
            if not p.is_file():
                raise SystemExit(f"動画が見つかりません: {p}")
        self.cap_raw = cv2.VideoCapture(str(raw_path))
        self.cap_depth = cv2.VideoCapture(str(depth_path))
        for cap, p in ((self.cap_raw, raw_path), (self.cap_depth, depth_path)):
            if not cap.isOpened():
                raise SystemExit(f"動画を開けません: {p}")

        n_raw = int(self.cap_raw.get(cv2.CAP_PROP_FRAME_COUNT))
        n_depth = int(self.cap_depth.get(cv2.CAP_PROP_FRAME_COUNT))
        self.n = min(n_raw, n_depth)
        if n_raw != n_depth:
            print(f"[warn] フレーム数が不一致: raw={n_raw}, depth={n_depth} → {self.n} まで再生",
                  file=sys.stderr)
        if self.n <= 0:
            raise SystemExit("有効なフレームがありません（0 フレーム）。")

        fps = self.cap_raw.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1e-3 else 30.0
        if fps and fps > 1e-3 and abs(fps - EXPECTED_FPS) > FPS_MISMATCH_TOL:
            print(f"[warn] raw の fps が想定({EXPECTED_FPS:.1f})と異なります: {fps:.2f} "
                  f"→ この fps のまま等倍(native)再生します", file=sys.stderr)

        self._pos = -1  # 直近に retrieve 済みのフレーム番号（-1 = 未読）

    def read_at(self, idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        idx = max(0, min(idx, self.n - 1))
        if idx != self._pos + 1:
            self.cap_raw.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.cap_depth.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok_r, fr = self.cap_raw.read()
        ok_d, fd = self.cap_depth.read()
        self._pos = idx
        return (fr if ok_r else None), (fd if ok_d else None)

    def release(self) -> None:
        self.cap_raw.release()
        self.cap_depth.release()


def _label(img: np.ndarray, text: str) -> np.ndarray:
    # パネル左上に、背景に依らず読めるよう黒縁＋白文字で見出しを描く（元画像は破壊しない）。
    out = img.copy()
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _compose_panels(raw: np.ndarray | None, depth: np.ndarray | None) -> np.ndarray:
    # raw|depth をラベル付きで横連結した行を返す（操作ヒント行は付けない）。
    # 対話再生（_compose）と --export の共通の土台。欠損フレームは黒で埋め、高さを揃える。
    def _norm(img: np.ndarray | None, ref_shape: tuple[int, int]) -> np.ndarray:
        if img is None:
            return np.zeros((ref_shape[0], ref_shape[1], 3), dtype=np.uint8)
        return img

    ref = None
    for cand in (raw, depth):
        if cand is not None:
            ref = cand.shape[:2]
            break
    if ref is None:
        ref = (352, 528)

    left = _label(_norm(raw, ref), "RAW (pre-processed FOV)")
    right = _label(_norm(depth, ref), "DEPTH (HUD)")
    if left.shape[0] != right.shape[0]:
        h = min(left.shape[0], right.shape[0])
        left, right = left[:h], right[:h]

    divider = np.full((left.shape[0], 3, 3), 60, dtype=np.uint8)
    return np.hstack([left, divider, right])


def _compose(raw: np.ndarray | None, depth: np.ndarray | None,
             idx: int, n: int, fps: float, speed: float, playing: bool) -> np.ndarray:
    # ラベル付き横連結の下に、操作ヒントを含む状態行を足す（対話再生用）。
    row = _compose_panels(raw, depth)
    status = (f"frame {idx + 1}/{n}   {fps:.1f}fps x{speed:g}   "
              f"[{'PLAY' if playing else 'PAUSE'}]   SPACE:play/pause  .,:step  []:speed  r:reset  l:loop  q:quit")
    bar = np.full((28, row.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, status, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([row, bar])


def _even_pad(img: np.ndarray) -> np.ndarray:
    # mp4v（MPEG-4）は偶数幅/高さを前提とするエンコーダがあるため、奇数辺は右/下に 1px 黒で詰める。
    # raw|depth 合成は 528+3+528=1059 のように奇数幅になり得る（仕切り 3px）ため export 前に必ず通す。
    h, w = img.shape[:2]
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return img


def _export_combined(pair: SyncPair, out_path: Path) -> None:
    """全フレームを raw|depth のラベル付き合成（操作ヒント行なし）として 1 本の mp4 に書き出す。

    native fps・等倍で決定論的に書き出す（再生速度/一時停止/シークに依存しない）。GUI は開かない。
    フレームサイズは先頭フレームの合成から決め、以後サイズが食い違ったフレームは詰めて揃える
    （通常は raw/depth が固定サイズなので発生しないが、片側欠損等の保険）。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fps = pair.fps
    n = pair.n
    fourcc = cv2.VideoWriter_fourcc(*EXPORT_FOURCC)

    writer: cv2.VideoWriter | None = None
    frame_size: tuple[int, int] | None = None
    written = 0
    try:
        for idx in range(n):
            raw_f, depth_f = pair.read_at(idx)
            frame = _even_pad(_compose_panels(raw_f, depth_f))
            h, w = frame.shape[:2]
            if writer is None:
                frame_size = (w, h)
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)
                if not writer.isOpened():
                    writer.release()
                    raise SystemExit(
                        f"出力動画の VideoWriter を開けません: {out_path} "
                        f"(fourcc={EXPORT_FOURCC!r}, fps={fps}, size={frame_size})。"
                        "コーデックが利用可能か確認してください。"
                    )
            elif (w, h) != frame_size:
                frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
            # VideoWriter.write は void を返す OpenCV ビルドがあるため、明示的 False のみ失敗とみなす
            # （runner._SessionVideoRecorder と同じ規約）。
            if writer.write(frame) is False:
                raise SystemExit(f"フレーム書き込みに失敗しました（idx={idx}）: {out_path}")
            written += 1
            if idx == 0 or (idx + 1) % 30 == 0 or idx == n - 1:
                print(f"\rexport: {idx + 1}/{n} ({(idx + 1) / n * 100.0:5.1f}%)",
                      end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)  # 進捗行の改行
    finally:
        if writer is not None:
            writer.release()

    if frame_size is not None:
        print(f"combined 動画を書き出しました: {out_path}  "
              f"({written} フレーム, {fps:.1f}fps, {frame_size[0]}x{frame_size[1]})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="raw/depth セッション動画のフレーム同期同時再生")
    parser.add_argument("--session", default=None, help="セッション名（既定: 最新セッションを自動選択）")
    parser.add_argument("--root", default=None, help="リポジトリルート（既定: ツール位置から推定）")
    parser.add_argument("--raw", default=None, help="video_raw.mp4 の明示パス（--depth と併用）")
    parser.add_argument("--depth", default=None, help="video_depth.mp4 の明示パス（--raw と併用）")
    parser.add_argument("--loop", action="store_true", help="終端で先頭に戻して継続")
    parser.add_argument("--speed", type=float, default=1.0, help="初期再生速度（既定 1.0＝実時間）")
    parser.add_argument("--export", action="store_true",
                        help="再生せず、raw|depth の合成（ラベル付き・操作ヒント行なし）を 1 本の mp4 に "
                             "native fps で書き出して終了。出力は outputs/{session}/video_combined.mp4"
                             "（--raw/--depth 指定時は depth と同じディレクトリ）")
    args = parser.parse_args(argv)

    raw_path, depth_path = _resolve_paths(args)
    print(f"raw  : {raw_path}")
    print(f"depth: {depth_path}")

    pair = SyncPair(raw_path, depth_path)

    if args.export:
        # depth と同じディレクトリに置く＝セッション経路なら outputs/{session}/video_combined.mp4。
        out_path = depth_path.parent / COMBINED_NAME
        print(f"out  : {out_path}")
        try:
            _export_combined(pair, out_path)
        finally:
            pair.release()
        return

    fps = pair.fps
    n = pair.n
    loop = bool(args.loop)

    # 最も近い既定速度に丸める。
    speed_idx = min(range(len(SPEEDS)), key=lambda k: abs(SPEEDS[k] - args.speed))
    speed = SPEEDS[speed_idx]

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    # トラックバーのプログラム更新と手動ドラッグを区別するためのフラグ。
    seeking = {"programmatic": False, "target": None}

    def on_trackbar(v: int) -> None:
        if seeking["programmatic"]:
            return
        seeking["target"] = v  # メインループでフレーム反映（描画スレッド衝突回避）

    cv2.createTrackbar("frame", WINDOW, 0, max(0, n - 1), on_trackbar)

    idx = 0
    playing = True
    need_render = True
    # next_due を「今」にすると、ループ最初の pacing 判定が描画前に即座に true になり
    # idx が 0→1 に進んでしまって frame 0 が一度も描画されない。1 周期分先に置くことで
    # 最初のループでは frame 0 の描画のみ行い、idx はまだ進めないようにする。
    next_due = time.perf_counter() + 1.0 / (fps * speed)

    while True:
        # --- シーク要求の取り込み（トラックバーのドラッグ） ---
        if seeking["target"] is not None:
            idx = max(0, min(seeking["target"], n - 1))
            seeking["target"] = None
            playing = False
            need_render = True
            next_due = time.perf_counter()

        # --- 再生の進行（実時間ペーシング） ---
        now = time.perf_counter()
        if playing and now >= next_due:
            if idx + 1 < n:
                idx += 1
                need_render = True
            elif loop:
                idx = 0
                need_render = True
            else:
                playing = False  # 終端で停止（最終フレームを保持表示）
            interval = 1.0 / (fps * speed)
            next_due += interval
            if next_due < time.perf_counter():  # 描画が interval を超過したら詰めずにリセット
                next_due = time.perf_counter() + interval

        # --- 描画 ---
        if need_render:
            raw_f, depth_f = pair.read_at(idx)
            canvas = _compose(raw_f, depth_f, idx, n, fps, speed, playing)
            cv2.imshow(WINDOW, canvas)
            seeking["programmatic"] = True
            cv2.setTrackbarPos("frame", WINDOW, idx)
            seeking["programmatic"] = False
            need_render = False

        # --- キー処理（1ms 待ちで応答性を確保） ---
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q / ESC
            break
        elif key == ord(" "):
            playing = not playing
            next_due = time.perf_counter()
        elif key == ord("."):
            playing = False
            idx = min(idx + 1, n - 1)
            need_render = True
        elif key == ord(","):
            playing = False
            idx = max(idx - 1, 0)
            need_render = True
        elif key == ord("]"):
            speed_idx = min(speed_idx + 1, len(SPEEDS) - 1)
            speed = SPEEDS[speed_idx]
            need_render = True
            next_due = time.perf_counter()
        elif key == ord("["):
            speed_idx = max(speed_idx - 1, 0)
            speed = SPEEDS[speed_idx]
            need_render = True
            next_due = time.perf_counter()
        elif key == ord("r"):
            idx = 0
            need_render = True
            next_due = time.perf_counter()
        elif key == ord("l"):
            loop = not loop
            need_render = True

        # ウィンドウが閉じられたら終了（× ボタン対応）。
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    pair.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
