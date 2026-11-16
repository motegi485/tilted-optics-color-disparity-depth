"""realtime/runner.py — オンライン実行のループ本体（取得 → 計算 → 表示 → 録画）。

実行: リポジトリ root から `uv run python -m realtime.runner --source webcam`
      （camera.py と同じく config を絶対 import するため、パッケージとして起動する）。

(1) データフロー
    source.read() -> raw(1280x720) -> preprocess -> proc(352,528,3)
      -> pipeline.process_frame(proc, params) -> depth_map（NaN 伝播）
      -> 表示（距離範囲ゲートを vmin/vmax で適用・カラーマップ＋距離凡例のカラーバー）＋FPS
      -> 録画（消費フレームのみ: raw -> captures/, depth -> outputs/）

(2) スレッドモデル
    取得スレッドが source.read() を回し、最新フレームを Lock 保護の単一スロットへ「上書き」する
    （キューに溜めない）。計算ループ（メインスレッド）はスロットから最新フレームのみを消費し、
    間に合わなかった中間フレームは自然に破棄される（遅延が蓄積しない）。同期は Lock のみで足りる
    （新フレーム待ちの Condition は不要）。計算がカメラより速いと「新フレームがまだ無い」状態が
    定常になるが、その場合は frame_id の一致で空回りを検出し 1ms sleep して待つ（計算ループ参照）。
    計算が遅い場合も最新のみを消費するだけなので、いずれの速度関係でも Lock のみで正しく動く。
    cv2 ウィンドウのイベント処理（waitKey）は1フレームの計算が終わるまで走らないため、
    1フレームの計算が長い入力では表示の更新がその分遅れる。

(3) 保存と表示の関係
    表示の距離レンジ（config.DEPTH_MIN〜DEPTH_MAX）は表示側で vmin/vmax として使う。process_frame が
    返す depth マップは書き換えない（書き換えると checks/evaluate.py の depth マップとビット一致しなくなる）。
    録画する depth は NaN を保持した float32 をそのまま np.save する（保存データの唯一の正）。
    表示用カラーマップは使い捨てコピー上で NaN を潰すだけで、元 depth には触れない。
    カラーバー（距離→色の凡例・draw_colorbar）は to_colormap 直後にこのコピーへ焼き込むため、
    保存する viz.png にも含まれる。HUD の FPS 等はこの後さらに焼き込むため、ライブ表示と
    セッション動画にのみ残る（viz.png はクリーンなカラーマップ＋カラーバーで統一）。
    ライブ表示だけは valid-hold（hold_valid）で直前フレームの有効値を重ねるが、保存物には使わない。

本 runner はライブ取得（中間フレームの破棄あり）が責務で、保存した raw を1フレームずつ
再生して結果を照合する用途は扱わない（その検証は checks/verify_clip_replay.py）。

`--record-video`（既定 OFF）: 上記とは別に、セッション全長の閲覧用動画（raw/depth）を
専用スレッドで記録できる。連番 PNG/npy には影響しない付加機能。

realtime/ は core ではない（config import・I/O・cv2・threading を持ってよい）。
計算の中核は持たず pipeline.process_frame（→ core 純関数）を呼ぶだけ。
入力検証は raise（assert は -O で無効化されるため使わない）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import config
from core.params import DepthParams, load_params
from realtime.camera import (
    FrameSource,
    StillImageSource,
    VideoFileSource,
    WebcamSource,
    preprocess,
)
from realtime.pipeline import process_frame

logger = logging.getLogger(__name__)


def to_colormap(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """depth(float32, NaN 伝播) -> 表示用 BGR uint8。元 depth は一切書き換えない。

    [vmin, vmax]=[DEPTH_MIN, DEPTH_MAX] を 0..255 へ線形マッピングして固定スケールで
    カラーマップ化する（毎フレーム自動スケールしないことで、動画として見たとき色変化＝被写体変化と
    読める）。JET は小さい値ほど青・大きい値ほど赤になる並びのため、反転（255-u8）してから
    適用し「近いほど赤・遠いほど青」にする。
    無効な画素は黒で塗る。無効とは NaN（測定不能）と、範囲外（z <= vmin または z >= vmax）のこと。
    範囲外を端の色にクランプすると近接の赤などに見えてしまうため、checks/ の距離範囲ゲート
    （0 < z < DEPTH_MAX）と同じく無効として扱う。これは見た目だけの処理で、保存する depth(npy) は
    変えない。
    """
    if vmax <= vmin:
        raise ValueError(f"to_colormap: vmax({vmax}) は vmin({vmin}) より大きい必要があります")
    valid = (depth > vmin) & (depth < vmax)           # NaN との比較は False なので NaN も無効になる
    scaled = (np.nan_to_num(depth, nan=vmin) - vmin) / (vmax - vmin) * 255.0
    u8 = np.clip(scaled, 0, 255).astype(np.uint8)     # NaN を uint8 へ直接 astype すると不定値化するため必ず潰してから
    color = cv2.applyColorMap(255 - u8, cv2.COLORMAP_JET)  # 反転: 近い(小さいz)ほど赤、遠い(大きいz)ほど青
    # 無効は黒。bool マスクの 3ch 代入（color[invalid]=0）より、マスク付き bitwise_and の方が
    # 約 10 倍速い（有効画素は全体の数 % で、大半が黒になるため）。
    return cv2.bitwise_and(color, color, mask=valid.view(np.uint8))


def draw_colorbar(img: np.ndarray, vmin: float, vmax: float, *, num_ticks: int = 6) -> None:
    """深度マップ右端に距離→色の対応を示す縦カラーバーを重ね描きする（in-place）。

    to_colormap と同じ反転 JET マッピング（近い(小さいz)ほど赤・遠い(大きいz)ほど青）をバー内で
    再現し、近距離(赤)をバー下端・遠距離(青)をバー上端に置く（値が下から上へ増加する、通常の
    グラフ縦軸と同じ向き）。目盛りラベルは m 単位（vmin/vmax は mm で渡し、表示時に 1000 で割る）。
    黒の太線+白の細線の二重描画で、_draw_hud と同様に背景色に依らず読めるようにする。
    """
    if vmax <= vmin:
        raise ValueError(f"draw_colorbar: vmax({vmax}) は vmin({vmin}) より大きい必要があります")
    if num_ticks < 2:
        raise ValueError(f"draw_colorbar: num_ticks は2以上が必要です（実際: {num_ticks}）")
    h, w = img.shape[:2]
    bar_w, margin_right, margin_tb = 16, 6, 14
    x2 = w - margin_right
    x1 = x2 - bar_w
    y1 = margin_tb
    y2 = h - margin_tb
    bar_h = y2 - y1
    if x1 < 0 or bar_h < num_ticks:
        raise ValueError(f"draw_colorbar: 画像が小さすぎます（shape={img.shape}）")

    frac = np.arange(bar_h) / (bar_h - 1)               # 0(上端)→1(下端)
    depth_col = vmax - frac * (vmax - vmin)              # 上端=vmax(遠)→下端=vmin(近)
    u8 = np.clip((depth_col - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    gradient = cv2.applyColorMap(255 - u8, cv2.COLORMAP_JET).reshape(bar_h, 1, 3)  # to_colormap と同じ反転
    img[y1:y2, x1:x2] = np.repeat(gradient, bar_w, axis=1)

    cv2.rectangle(img, (x1 - 1, y1 - 1), (x2, y2), (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 1, cv2.LINE_AA)

    for i in range(num_ticks):
        t = i / (num_ticks - 1)                          # 0(上端)→1(下端)
        y = int(round(y1 + t * (bar_h - 1)))
        val = vmax - t * (vmax - vmin)
        label = f"{val / 1000.0:g}"                      # 目盛りは m 単位（0, 1, 2, ...）
        cv2.line(img, (x1 - 5, y), (x1 - 1, y), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(img, (x1 - 5, y), (x1 - 1, y), (255, 255, 255), 1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        tx, ty = x1 - 8 - tw, y + th // 2
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    unit_pos = (x1 - 2, y1 - 4)                          # バー上端の上に単位 "m" を添える
    cv2.putText(img, "m", unit_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "m", unit_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


class _FrameGrabber:
    """取得スレッド: source.read() を回し最新 raw を単一スロットへ上書きする。

    キューに溜めないことで遅延を一定に保つ。frame_id（単調増加）で計算側が「新しいフレームか／
    同じ最新を二度処理していないか」を判定できるようにする。EOF・read 失敗・スレッド内例外では
    stop を立てて終了し、例外は main 側で再 raise できるよう保持する（黙って握り潰さない）。
    """

    def __init__(self, source: FrameSource):
        self._source = source
        self._lock = threading.Lock()                 # 単一スロットの保護（新フレーム待ちには使わない）
        self._stop = threading.Event()
        self._frame: np.ndarray | None = None
        self._frame_id = -1                           # まだ1枚も無い
        self._error: BaseException | None = None
        # still/video はほぼ瞬時に読めるため、スロットル無しだと計算スレッドとの帯域競合になりうる。
        # webcam は実フレームレートで自然に律速されるため対象外（コンストラクタで1回だけ判定）。
        self._throttle = not isinstance(source, WebcamSource)
        self._thread = threading.Thread(target=self._loop, name="frame-grabber", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                ok, raw = self._source.read()
                if not ok or raw is None:             # EOF（file source）や読み取り失敗
                    break
                with self._lock:
                    self._frame = raw                 # 参照差し替え。古い配列を計算側が握っていても不変なので copy 不要
                    self._frame_id += 1
                if self._throttle:
                    time.sleep(0.001)                 # still/video 用の軽い間引き
        except BaseException as e:                     # スレッド内例外は飲み込まず main へ運ぶ
            self._error = e
        finally:
            self._stop.set()                          # どの終了経路でも計算側へ停止を伝える

    def latest(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            return self._frame, self._frame_id

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def stop_and_join(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout)                    # daemon なので join が遅れても interpreter 終了は妨げない
        if self._thread.is_alive():
            # timeout 時点でまだ cap.read() 実行中の可能性がある。この状態で source.release() を
            # 呼ぶと OpenCV ネイティブ層での競合になりうるため、追加で猶予を与える。
            logger.warning("frame-grabber が %.1fs 以内に終了しませんでした。追加で最大5.0s 待機します", timeout)
            self._thread.join(5.0)
            if self._thread.is_alive():
                logger.warning("frame-grabber がまだ終了していません（cap.read() 実行中の可能性）。処理を続行します")

    def raise_if_error(self) -> None:
        if self._error is not None:
            raise self._error

    def log_error_if_any(self) -> None:
        """raise_if_error の非 raise 版。計算ループ側の例外を優先したい finally 節で使う。"""
        if self._error is not None:
            logger.error("frame-grabber 内で例外が発生していました（記録のみ）: %r", self._error)


class _LatestFrameSlot:
    """録画スレッドへ HUD 付き表示画像を1枚受け渡す単一スロット（セッション動画用）。

    _FrameGrabber と同型の「参照を上書き」パターンだが、ロックは _FrameGrabber とは別インスタンス
    にして両者を結合させない。メインループが公開する color は毎イテレーション to_colormap で
    新規生成され、公開（_draw_hud 後）以降メインループ側では二度と変更しないため、copy せず
    参照共有でよい（_FrameGrabber._loop の raw と同じ根拠）。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._frame


def _open_video_writer(
    path: Path,
    *,
    fps: float,
    fourcc: str,
    frame_size: tuple[int, int],
    writer_factory=cv2.VideoWriter,
) -> cv2.VideoWriter:
    """VideoWriter を開く。失敗時は release してから raise（壊れた動画を黙って残さない）。"""
    writer = writer_factory(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, frame_size)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"セッション動画の VideoWriter を開けません: {path} "
            f"(fourcc={fourcc!r}, fps={fps}, size={frame_size})。コーデックが利用可能か確認してください"
        )
    return writer


class _SessionVideoRecorder:
    """セッション全長の閲覧用動画（raw/depth）を録画する専用スレッド。

    grabber・メインループとは別スレッドで動かし、計算の計測区間（dt）に録画のコードを含めない: raw は grabber.latest()
    （既存の複数リーダー安全なロック保護アクセサ）を、depth は display_slot（_LatestFrameSlot）を、
    それぞれ自スレッドの一定周期で読むだけで、書き込み側をブロックしない。

    depth 側は「新しい計算結果が出るまで直前の絵を繰り返す」サンプル&ホールドで、raw と同じ
    tick 数だけ書く（不変条件: 1 tick につき両方書くか両方書かないかの二択。2動画のフレーム数を
    常に一致させ、並べて再生したときのタイムラインを揃える＝計算遅延の可視化という目的の要）。
    """

    def __init__(
        self,
        raw_path: Path,
        depth_path: Path,
        display_slot: _LatestFrameSlot,
        *,
        fps: float,
        fourcc: str,
        frame_size: tuple[int, int],
        writer_factory=cv2.VideoWriter,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"_SessionVideoRecorder: fps は正である必要があります（実際: {fps}）")
        self._display_slot = display_slot
        self._interval = 1.0 / fps
        self._black = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

        self._raw_writer = _open_video_writer(
            raw_path, fps=fps, fourcc=fourcc, frame_size=frame_size, writer_factory=writer_factory)
        try:
            self._depth_writer = _open_video_writer(
                depth_path, fps=fps, fourcc=fourcc, frame_size=frame_size, writer_factory=writer_factory)
        except BaseException:
            self._raw_writer.release()                 # 2本目失敗時に1本目のハンドルをリークさせない
            raise

        self._grabber = None
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._released = False
        self.frames_written = 0
        self.ticks_skipped = 0
        self._thread = threading.Thread(target=self._loop, name="session-video-recorder", daemon=True)

    def start(self, grabber) -> None:
        if not hasattr(grabber, "latest") or not hasattr(grabber, "stopped"):
            raise ValueError("_SessionVideoRecorder.start: grabber は latest()/stopped を持つ必要があります")
        self._grabber = grabber
        self._thread.start()

    def _write_one_tick(self) -> None:
        raw, _fid = self._grabber.latest()
        if raw is None:                                # grabber 起動直後のごく短い窓（不変条件: 両方スキップ）
            return
        proc = preprocess(raw)                          # メインループと同一の共有前処理（安価。ZNCC は通らない）
        # cv2.VideoWriter.write は C++ 側が void のため、この OpenCV ビルド（4.13）は None を返す
        # （bool を返す cv2.imwrite とは異なる）。None は「失敗」ではなく「情報無し」なので、
        # 明示的に False が返った場合のみ失敗として扱う（テストの fake writer は bool を返す）。
        if self._raw_writer.write(proc) is False:
            raise RuntimeError("session-video-recorder: raw VideoWriter.write が失敗しました")
        disp = self._display_slot.latest()
        frame = disp if disp is not None else self._black  # サンプル&ホールドの実体（無ければ黒＝既存の無効色規約）
        if self._depth_writer.write(frame) is False:
            raise RuntimeError("session-video-recorder: depth VideoWriter.write が失敗しました")
        self.frames_written += 1

    def _loop(self) -> None:
        try:
            next_due = time.perf_counter()
            while not self._stop.is_set() and not self._grabber.stopped:
                now = time.perf_counter()
                if now < next_due:
                    if self._stop.wait(next_due - now):  # 停止要求に即応答（sleep でなく Event.wait）
                        break
                    continue
                self._write_one_tick()
                next_due += self._interval
                if next_due < time.perf_counter():       # エンコードが interval を超過：重複書きで追いつこうとしない
                    self.ticks_skipped += 1
                    next_due = time.perf_counter() + self._interval
        except BaseException as e:                        # grabber と同型: 飲み込まず保持して main へ運ぶ
            self._error = e
        finally:
            self._stop.set()

    def stop_and_join(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.warning("session-video-recorder が %.1fs 以内に終了しませんでした。追加で最大5.0s 待機します", timeout)
            self._thread.join(5.0)
            if self._thread.is_alive():
                logger.warning("session-video-recorder がまだ終了していません。処理を続行します")
        self.ensure_released()

    def ensure_released(self) -> None:
        """release() の冪等な公開版（構築後・スレッド起動前に例外が飛ぶ経路の保険）。"""
        if self._released:
            return
        self._released = True
        for name, writer in (("raw", self._raw_writer), ("depth", self._depth_writer)):
            try:
                writer.release()
            except Exception as e:
                logger.error("session-video-recorder: %s VideoWriter の release に失敗しました: %r", name, e)

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def raise_if_error(self) -> None:
        if self._error is not None:
            raise self._error

    def log_error_if_any(self) -> None:
        """raise_if_error の非 raise 版。計算ループ側の例外を優先したい finally 節で使う。"""
        if self._error is not None:
            logger.error("session-video-recorder 内で例外が発生していました（記録のみ）: %r", self._error)


def _ensure_session_dirs(session: str) -> tuple[Path, Path]:
    # 録画を実行（セッション）ごとに分離する。captures/{session}, outputs/{session} に保存する
    # 実行を跨いだ連番（000000..）の上書き・混在を論理的に防ぐ（クリップ再生での再現確認は
    # 1セッション内のクリーンな連番が前提）。既存かつ非空のディレクトリを黙って再利用すると
    # 連番が 000000 から上書きされ新旧フレームが混在した「キメラセッション」になり、
    # verify_clip_replay の自己整合チェックでは検出できないため、
    # 録画開始前（フレーム取得前）に raise で止める。空ディレクトリは許可する。
    # session は無検証で Path 結合すると `..` や絶対パスで captures/outputs の外へ書けてしまう。
    # 単一のディレクトリ名だけを許可する。
    # 注: Windows の pathlib では Path("..").name == ".." となり name 比較だけでは弾けないため、
    # カレント/親ディレクトリ参照は明示的に除外する。
    # 区切り文字も明示的に拒否する（POSIX では "\\" が通常の文字なので Path.name 比較だけでは
    # "a\\b" が通ってしまい、同じセッション名が Windows と Linux で別の扱いになる）。
    if (not session or session in {".", ".."} or "/" in session or "\\" in session
            or Path(session).name != session):
        raise ValueError(
            f"セッション名が不正です: {session!r}。パス区切り・親ディレクトリ参照(..)・"
            f"絶対パスを含まない単一のディレクトリ名を指定してください")

    captures = Path(config.CAPTURES_DIR) / session
    outputs = Path(config.OUTPUTS_DIR) / session
    for d in (captures, outputs):
        if d.is_dir() and any(d.iterdir()):
            raise RuntimeError(
                f"セッション {session} は既存です。別名を指定するか、既存ディレクトリを退避してください"
            )
    captures.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    return captures, outputs


def _session_video_paths(captures_dir: Path, outputs_dir: Path) -> tuple[Path, Path]:
    """セッション動画の raw/depth 出力先。captures/outputs の既存分離を踏襲する。"""
    return captures_dir / config.SESSION_VIDEO_RAW_NAME, outputs_dir / config.SESSION_VIDEO_DEPTH_NAME


def _write_session_meta(outputs_dir: Path, params: DepthParams, params_path: str | None,
                        source: FrameSource, *, record_video: bool = False) -> None:
    """録画セッションの使用 params・config・起動条件を outputs/{session}/meta.json に記録する。

    「録画時と同じ params・config を使う」という制約が人間の記憶頼みだと、--params の既定が
    変わったときに過去セッションとの不一致事故が起きやすい。verify_clip_replay.py がこれを
    自動参照できるようにする。

    session_video は record_video が False でも常に記録する（{"enabled": false}）。後からセッションを
    見たとき「動画が無いのは仕様か、この項目の追加前の古いセッションか」を meta.json だけで判別できる
    ようにするため。
    """
    if record_video:
        session_video = {
            "enabled": True,
            "fps": config.SESSION_VIDEO_FPS,
            "fourcc": config.SESSION_VIDEO_FOURCC,
            "raw": config.SESSION_VIDEO_RAW_NAME,
            "depth": config.SESSION_VIDEO_DEPTH_NAME,
        }
    else:
        session_video = {"enabled": False}
    meta = {
        "params_path": params_path,
        "params": {
            "v_pos": params.v_pos, "u_pos": params.u_pos, "a_pos": params.a_pos, "w_pos": params.w_pos,
            "v_neg": params.v_neg, "u_neg": params.u_neg, "a_neg": params.a_neg, "w_neg": params.w_neg,
        },
        "config": {
            "IMAGE_SIZE": list(config.IMAGE_SIZE),
            "KERNEL_SIZE": config.KERNEL_SIZE,
            "SEARCH_RANGE_X": config.SEARCH_RANGE_X,
            "EDGE_TH": config.EDGE_TH,
            "DISP_MAX": config.DISP_MAX,
            "INFER_BAND_LOW": config.INFER_BAND_LOW,     # 推論帯。この項目の追加前のセッションの meta.json には無い
            "INFER_BAND_HIGH": config.INFER_BAND_HIGH,
        },
        # 表示レンジ（距離範囲ゲートの表示側適用・viz PNG／セッション動画の色の割り当て）。保存 npy には影響しない。
        # この項目の追加前のセッションの meta.json には無い。
        "display": {
            "DEPTH_MIN": config.DEPTH_MIN,
            "DEPTH_MAX": config.DEPTH_MAX,
        },
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_type": type(source).__name__,
        "session_video": session_video,
    }
    (outputs_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def hold_valid(depth: np.ndarray, history) -> np.ndarray:
    """ライブ表示用: 現在のフレームで無効（NaN）の画素を、直前のフレームの有効値で埋めた depth を返す。

    縦エッジ上の有効画素は毎フレーム 2 割ほど入れ替わり、そのまま表示すると点が明滅して見える。
    直前の数フレームで一度でも有効だった画素を残すと、明滅が減る（代わりに、動く物体の後ろに
    数フレーム分の残像が出る）。history は新しい順の depth の並び。入力は書き換えない。
    「有効」は NaN でないことだけで判定する（範囲外の値も有効として保持し、描画のときに to_colormap が黒にする）。
    この処理は表示専用で、保存する depth(npy)・viz・セッション動画には使わない。
    """
    out = depth.copy()
    for prev in history:
        fill = np.isnan(out) & ~np.isnan(prev)
        out[fill] = prev[fill]
    return out


def _put_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5) -> None:
    """背景色に依らず読めるよう、黒の太線＋白の細線で文字を描く（in-place）。"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def compose_live_view(proc_bgr: np.ndarray, depth_panel: np.ndarray, *,
                      hold_on: bool, show_overlay: bool) -> np.ndarray:
    """ライブ表示用に、左に入力画像・右に距離マップを並べた画像を作る。

    入力画像と距離マップを同時に見られるようにするための表示専用の画像で、保存はしない。
    入力画像・距離マップの下端に名前を入れ、show_overlay のときは操作キーと valid-hold の状態も書く。
    """
    left = proc_bgr.copy()
    right = depth_panel.copy()
    h = left.shape[0]
    _put_text(left, "Input", (8, h - 10))
    _put_text(right, "Distance map", (8, h - 10))
    if show_overlay:
        _put_text(left, f"[v] hold {'ON' if hold_on else 'OFF'}   [h] hide info   [q] quit", (8, 22), 0.5)
    return np.hstack([left, right])


def _draw_hud(img: np.ndarray, *, fps: float | None, ms: float, valid: int, frame_id: int) -> None:
    """表示画像へ cFPS / ms / 有効画素数 / frame_id をオーバレイ（ASCII のみ・in-place）。

    黒の太線 + 白の細線で、背景色に依らず読めるようにする。HUD は表示専用画像にのみ描き、
    保存する viz（クリーン）には描かない。

    表記が `cFPS`（compute FPS）なのは、この値が preprocess + process_frame 区間だけの
    所要時間 dt の逆数の EMA であり、入力待ち・表示・同期保存・GUI を含まないため。
    実効的な更新レートや録画スループット（E2E throughput）ではない。
    入力が ~30fps でも計算が ~12ms なら cFPS は ~80 と表示される。
    """
    fps_txt = f"{fps:.2f}" if fps is not None else "n/a"
    lines = [f"cFPS: {fps_txt}  ms/frame: {ms:.0f}", f"valid: {valid}  id: {frame_id}"]
    for i, text in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def run(
    source: FrameSource,
    params: DepthParams,
    *,
    record: bool = True,
    save_viz: bool = True,
    record_video: bool = False,
    colorbar: bool = True,
    session: str | None = None,
    window_name: str = "tilted-optics-color-disparity-depth",
    log_every: int = 30,
    params_path: str | None = None,
) -> None:
    """ライブ取得→処理→表示→（消費フレームのみ）録画のループ本体。

    source は context manager で開き、取得スレッドを起動してから計算ループへ入る。終了・例外時は
    「取得スレッド停止 → join → source release → ウィンドウ破棄」を必ず通す。`with` の __enter__ が
    raise した場合 __exit__ は呼ばれないが、source 側（camera.py）が掴んだハンドルを自前で release
    する規約なので、ここでは外側 finally でウィンドウ破棄のみ保証すればよい。
    録画中に raw PNG・viz PNG の書き込みが失敗したら RuntimeError で止める（npy は raw PNG の保存に
    成功したフレームにだけ書くので、raw の欠けた npy は残らない）。

    params_path: 録画セッションの meta.json に記録する起動時の --params 文字列（任意）。
    record_video: セッション全長の閲覧用動画（raw/depth）を録画するか。既定 False（オプトイン）。
        raw は前処理後 FOV の連続 ~SESSION_VIDEO_FPS、depth はクリーン版（valid-hold を掛けない距離マップ＋HUD）
        をサンプル&ホールドで記録する。専用スレッド（_SessionVideoRecorder）で行い、dt の計測区間
        （preprocess＋process_frame）には録画のコードを含めない。
    colorbar: 距離→色の対応を示す凡例（draw_colorbar）を焼き込むか。既定 True。to_colormap 直後・
        HUD 描画より前に color へ焼き込むため、HUD（FPS 等・ライブ表示とセッション動画のみ）とは
        異なり、保存する viz.png にも含まれる（viz.png は目視用で数値検証には使われないため、
        焼き込んでも再生照合や checks/evaluate.py の回帰には影響しない）。

    ライブ表示（保存・動画には影響しない）: 左に入力画像、右に距離マップを並べて出す（compose_live_view）。
    キー操作は v＝valid-hold（直前の数フレームの有効値で埋める。既定 ON）、h＝HUD と操作ヒントの
    表示切替、q／Esc＝終了。valid-hold と表示切替は run() 内の状態で、起動のたびに既定へ戻る。
    """
    if not isinstance(source, FrameSource):
        raise ValueError(f"run: source は FrameSource を想定。実際は {type(source)!r}")

    if record or record_video:
        session = session or time.strftime("%Y%m%d_%H%M%S")  # 既定は起動時刻。--session で固定可
        captures_dir, outputs_dir = _ensure_session_dirs(session)
        logger.info("録画セッション: captures=%s, outputs=%s", captures_dir, outputs_dir)
        _write_session_meta(outputs_dir, params, params_path, source, record_video=record_video)
    else:
        captures_dir, outputs_dir = None, None
    processed = 0                                     # 処理フレーム数。captures/outputs の連番 index と 1:1
    last_id = -1
    ema_fps: float | None = None                      # 処理レートの指数移動平均（表示用の平滑値）

    hold_on = True                                    # ライブ表示の valid-hold（v キーで切替）
    show_overlay = True                               # HUD と操作ヒントの表示（h キーで切替）
    history: deque = deque(maxlen=config.HOLD_FRAMES - 1)    # 直前の depth（新しい順・valid-hold 用）

    def render(d: np.ndarray) -> np.ndarray:          # depth -> 距離カラーマップ（カラーバー付き）
        c = to_colormap(d, config.DEPTH_MIN, config.DEPTH_MAX)
        if colorbar:
            draw_colorbar(c, config.DEPTH_MIN, config.DEPTH_MAX)
        return c

    display_slot: _LatestFrameSlot | None = None
    recorder: _SessionVideoRecorder | None = None
    if record_video:
        display_slot = _LatestFrameSlot()
        raw_video_path, depth_video_path = _session_video_paths(captures_dir, outputs_dir)
        # webcam の cold-start 待ち（with source の __enter__、最大 LIVE_WAIT_TIMEOUT_S）より前に
        # writer を開き、コーデック不在等をフェイルファストで検知する。
        recorder = _SessionVideoRecorder(
            raw_video_path, depth_video_path, display_slot,
            fps=config.SESSION_VIDEO_FPS, fourcc=config.SESSION_VIDEO_FOURCC, frame_size=config.IMAGE_SIZE)

    try:
        with source as src:                           # __enter__ で open
            # 大きな画面・プロジェクタで見られるよう、ウィンドウの大きさを変えられるようにする（縦横比は保つ）。
            # 画像は拡大表示されるだけで、計算・保存には影響しない。
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            grabber = _FrameGrabber(src)
            grabber.start()
            if recorder is not None:
                recorder.start(grabber)
            try:
                while True:
                    raw, fid = grabber.latest()
                    if raw is None or fid == last_id:  # 新しいフレームが無い（最新を二度処理しない）
                        if grabber.stopped:            # EOF/失敗で取得が終わったなら計算側も終了
                            break
                        time.sleep(0.001)              # 起動直後の空回り回避（定常では毎回新フレームがある）
                        continue
                    last_id = fid

                    t0 = time.perf_counter()
                    proc = preprocess(raw)             # raw -> (352,528,3)（runner が適用）
                    depth = process_frame(proc, params)  # -> depth_map（NaN 伝播・視差上限ゲート＋推論帯のみ適用済み・距離範囲ゲートは表示側）
                    dt = time.perf_counter() - t0      # 純粋な per-frame 計算時間（cFPS 表示の素データ）

                    # color: 保存する viz とセッション動画の元になるクリーン版（valid-hold を掛けない）。
                    # どちらも使わないとき（viz を保存しない・動画なし）は作らない。
                    need_clean = (record and save_viz) or display_slot is not None
                    color = render(depth) if need_clean else None
                    # live_color: ライブ表示用。hold_on のときだけ、直前の数フレームの有効値で埋める。
                    live_depth = hold_valid(depth, history) if (hold_on and history) else depth
                    if live_depth is depth and color is not None:
                        live_color = color.copy()       # HUD はライブ表示とセッション動画で別々に描く
                    else:
                        live_color = render(live_depth)
                    history.appendleft(depth)

                    if record:                         # 消費フレームのみ保存（raw->captures, depth->outputs）
                        idx = processed
                        # cv2.imwrite は書込み失敗（ディスク不足・権限・開けないパス等）を例外ではなく False で返す
                        # （OpenCV 4.13 で確認）。戻り値を捨てると raw の欠けた npy が残り、再生照合からも漏れるため、
                        # npy を書く前に止める（capture_calib.save_calib_png と同じ扱い）。
                        png_path = captures_dir / f"{idx:06d}.png"
                        if not cv2.imwrite(str(png_path), raw):   # 可逆PNG（既定圧縮。再現に使う正本の入力）
                            raise RuntimeError(
                                f"runner: raw PNG の書き込みに失敗しました: {png_path}（このフレームの depth npy は保存していません）")
                        np.save(str(outputs_dir / f"{idx:06d}.npy"), depth)      # NaN 保持の float32 を厳密保存
                        if save_viz:
                            viz_path = outputs_dir / f"{idx:06d}_viz.png"
                            if not cv2.imwrite(str(viz_path), color):      # HUD 無しのクリーン viz
                                raise RuntimeError(f"runner: viz PNG の書き込みに失敗しました: {viz_path}")

                    inst_fps = (1.0 / dt) if dt > 0 else float("inf")
                    ema_fps = inst_fps if ema_fps is None else 0.9 * ema_fps + 0.1 * inst_fps
                    n_valid = int(np.count_nonzero(~np.isnan(depth)))
                    if log_every > 0 and (processed % log_every == 0):
                        logger.info("processed=%d (frame id=%d): %.1f ms (~%.2f fps), valid=%d",
                                    processed, fid, dt * 1000.0, inst_fps, n_valid)
                    processed += 1                      # 連番 index と処理数（先頭=0。log は 0,log_every,... で発火）

                    if display_slot is not None:        # セッション動画の depth 側（サンプル&ホールドの供給元）
                        _draw_hud(color, fps=ema_fps, ms=dt * 1000.0, valid=n_valid, frame_id=fid)
                        display_slot.update(color)      # 動画は valid-hold・表示切替の影響を受けない
                    if show_overlay:
                        _draw_hud(live_color, fps=ema_fps, ms=dt * 1000.0, valid=n_valid, frame_id=fid)
                    cv2.imshow(window_name, compose_live_view(
                        proc, live_color, hold_on=hold_on, show_overlay=show_overlay))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):          # q or ESC で終了
                        break
                    if key == ord("v"):                # valid-hold の ON/OFF
                        hold_on = not hold_on
                    elif key == ord("h"):              # HUD と操作ヒントの表示/非表示
                        show_overlay = not show_overlay
            finally:
                if recorder is not None:
                    recorder.stop_and_join()           # grabber より先に止める（セッション終了後の書き込みを止める）
                grabber.stop_and_join()
                if sys.exc_info()[0] is not None:
                    # 計算ループ側の例外で finally に入った場合、grabber/recorder 側の例外で
                    # 元例外を置換しない（真因を保持する）。記録のみ行う。
                    grabber.log_error_if_any()
                    if recorder is not None:
                        recorder.log_error_if_any()
                else:
                    grabber.raise_if_error()           # 取得スレッドの例外を顕在化（無ければ no-op）
                    if recorder is not None:
                        recorder.raise_if_error()      # 動画書き込み失敗を静かに握り潰さない
    finally:
        if recorder is not None:
            recorder.ensure_released()                 # source.__enter__ が raise した経路の保険（冪等）
        cv2.destroyAllWindows()

    logger.info("runner 終了: 処理フレーム数=%d, 最終 fps~%s",
                processed, f"{ema_fps:.3f}" if ema_fps is not None else "n/a")
    if recorder is not None:
        logger.info("セッション動画: 書き込み=%d フレーム, 遅延ティック=%d",
                    recorder.frames_written, recorder.ticks_skipped)


def _build_source(args: argparse.Namespace) -> FrameSource:
    if args.source == "webcam":
        return WebcamSource()                          # index/backend は config から解決
    if args.source == "video":
        if not args.path:
            raise ValueError("--source video には --path が必要です")
        return VideoFileSource(args.path, loop=args.loop)
    if args.source == "still":
        if not args.path:
            raise ValueError("--source still には --path が必要です")
        return StillImageSource(args.path)
    raise ValueError(f"未知の source: {args.source!r}")


def main(argv: list[str] | None = None) -> None:
    # Windows の日本語コンソール(cp932)は非対応文字の print で UnicodeEncodeError を出すため
    # UTF-8 に固定する（tools/probe_webcam.py と同じ対処。ログに ~ 等が出ても計測を止めない）。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="tilted-optics-color-disparity-depth リアルタイム実行")
    parser.add_argument("--source", choices=("webcam", "video", "still"), default="webcam")
    parser.add_argument("--path", default=None, help="video/still 用の入力パス")
    parser.add_argument("--params", default=str(config.PARAMS_PATH),
                        help="fit_result.csv のパス（既定: params/fit_result.csv。回帰用の旧係数 params/baseline/ を使う場合は明示指定）")
    parser.add_argument("--loop", action="store_true", help="video source を EOF で巻き戻して継続")
    parser.add_argument("--no-record", action="store_true", help="録画を無効化")
    parser.add_argument("--no-viz", action="store_true", help="colormap PNG(viz) の保存を無効化")
    parser.add_argument("--no-colorbar", action="store_true",
                        help="距離→色の対応を示すカラーバー凡例の描画を無効化")
    parser.add_argument("--record-video", action="store_true",
                        help="セッション全長の閲覧用動画を録画する（captures/{session}/video_raw.mp4 と "
                             "outputs/{session}/video_depth.mp4）。raw は前処理後FOVの連続~30fps、"
                             "depth は直近の計算結果を保持表示（サンプル&ホールド）。既定: 無効")
    parser.add_argument("--session", default=None,
                        help="録画セッション名（既定: 起動時刻 YYYYmmdd_HHMMSS）。"
                             "captures/{session}, outputs/{session} に保存")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    params = load_params(args.params)                  # 起動時に1回だけ（ループ内で再計算しない）
    source = _build_source(args)
    run(source, params, record=not args.no_record, save_viz=not args.no_viz,
        record_video=args.record_video, colorbar=not args.no_colorbar,
        session=args.session, params_path=args.params)


if __name__ == "__main__":
    main()
