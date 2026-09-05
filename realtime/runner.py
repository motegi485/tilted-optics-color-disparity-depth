"""realtime/runner.py — オンライン実行のループ本体（要件書 §5.5・§7.2・§7.3）。

データフロー（§4.3 オンライン）を組み立てる:
    source.read() -> raw(1280x720) -> preprocess -> proc(352,528,3)
      -> pipeline.process_frame(proc, params) -> depth_map(NaN 伝播)
      -> 表示（gate2 を vmin/vmax で適用・カラーマップ＋距離凡例のカラーバー）＋FPS
      -> 録画（消費フレームのみ: raw->captures/, depth->outputs/）

実行: リポジトリ root から `uv run python -m realtime.runner --source webcam`
      （camera.py と同じく config を絶対 import するため、パッケージとして起動する）。

スレッドモデル（§7.2）: 取得スレッドが source.read() を回し、最新フレームを Lock 保護の
単一スロットへ「上書き」する（キューに溜めない）。計算ループ（メインスレッド）はスロットから
最新フレームのみを消費し、間に合わなかった中間フレームは自然に破棄される（遅延が蓄積しない）。
同期は Lock のみで足りる（新フレーム待ちの Condition は不要）。Phase C（Python 実装・
ZNCC ~14s/frame）では計算がカメラ(30fps)より遥かに遅く、計算が取得を追い越す状況は設計上
起きなかった。Phase D で Rust 化した現在は process_frame が ~12ms/frame（§9.1 のベンチ実測）
で、カメラ側の ~33ms より速いため「新フレームがまだ無い」側が定常になる。この場合は
frame_id 一致で空回りを検出し 1ms sleep して待つ（下の計算ループ参照）ので、いずれの
速度関係でも Lock のみで正しく動く。

gate2（Z レンジ 0<z<50000）はここ（表示側）で vmin/vmax として適用し、process_frame が
返す depth マップは書き換えない（書き換えると evaluate.py の depth マップとビット一致しなく
なる・§9.2）。録画する depth は NaN を保持した float32 をそのまま np.save する（唯一の真実）。
表示用カラーマップは使い捨てコピー上で NaN を潰すだけで、元 depth には触れない。
カラーバー（距離→色の凡例・draw_colorbar）は to_colormap 直後にこのコピーへ焼き込むため、
保存する viz.png にも含まれる（HUD の FPS 等はこの後さらに焼き込むため、ライブ表示と
セッション動画のみに残る・viz.png はクリーンなカラーマップ＋カラーバーで統一）。

realtime/ は core ではない（config import・I/O・cv2・threading を持ってよい・§4.1）。
計算の中核は持たず pipeline.process_frame（→ core 純関数）を呼ぶだけ。
入力検証は raise（assert は -O で無効化されるため使わない・リスク#10）。

注: cv2 ウィンドウのイベント処理（waitKey）は1フレームの計算が終わるまで走らない。
    Phase C（Python 実装・~14s/frame）ではウィンドウが数秒間固まり、最初の表示も起動から
    ~14s 後だった。Phase D の Rust 実装（~12ms/frame）ではこの体感は解消している。

§9.2 のクリップ再生「再現一致」検証（録った raw を1フレームずつ決定的に再生してライブ depth と
一致を確認）は、フレームを間引かない別ドライバの役割（連番PNG再生 source の camera.py 追加と
あわせて別途用意する）。本 runner はライブ取得（中間破棄あり）が責務。

`--record-video`（既定 OFF）: 上記とは別に、セッション全長の閲覧用動画（raw/depth・§7.3）を
専用スレッドで記録できる。canonical な連番 PNG/npy には一切影響しない付加機能。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
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

    gate2 の「表示側適用」の実体: [vmin, vmax]=[DEPTH_MIN, DEPTH_MAX] を 0..255 へ線形
    マッピングして固定スケールでカラーマップ化する（毎フレーム自動スケールしないことで、
    動画として見たとき色変化＝被写体変化と読める）。JET は小さい値ほど青・大きい値ほど赤
    になる並びのため、反転（255-u8）してから適用し「近いほど赤・遠いほど青」にする。
    NaN（測定不能）は近接(z≈0=赤)と弁別するため専用色（黒）で塗る。車載では警告/制動などの
    判断は元 depth(npy)を読む前提で、これはあくまで人間が眺めるための投影（情報を落とした
    見た目であり判断根拠にはしない）。
    """
    if vmax <= vmin:
        raise ValueError(f"to_colormap: vmax({vmax}) は vmin({vmin}) より大きい必要があります")
    invalid = np.isnan(depth)                         # 無効画素の位置を先に確保（潰す前に）
    scaled = (np.nan_to_num(depth, nan=vmin) - vmin) / (vmax - vmin) * 255.0
    u8 = np.clip(scaled, 0, 255).astype(np.uint8)     # NaN を uint8 へ直接 astype すると不定値化するため必ず潰してから
    color = cv2.applyColorMap(255 - u8, cv2.COLORMAP_JET)  # 反転: 近い(小さいz)ほど赤、遠い(大きいz)ほど青
    color[invalid] = (0, 0, 0)                        # 無効は黒（赤=近接 と一目で弁別）
    return color


def draw_colorbar(img: np.ndarray, vmin: float, vmax: float, *, num_ticks: int = 6) -> None:
    """深度マップ右端に距離→色の対応を示す縦カラーバーを重ね描きする（in-place）。

    to_colormap と同じ反転 JET マッピング（近い(小さいz)ほど赤・遠い(大きいz)ほど青）をバー内で
    再現し、近距離(赤)をバー下端・遠距離(青)をバー上端に置く（値が下から上へ増加する、通常の
    グラフ縦軸と同じ向き）。目盛りラベルは mm 表記（config.DEPTH_MIN/DEPTH_MAX の生値）。
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
        label = f"{val:.0f}"
        cv2.line(img, (x1 - 5, y), (x1 - 1, y), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(img, (x1 - 5, y), (x1 - 1, y), (255, 255, 255), 1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        tx, ty = x1 - 8 - tw, y + th // 2
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


class _FrameGrabber:
    """取得スレッド: source.read() を回し最新 raw を単一スロットへ上書きする（§7.2）。

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
            # 呼ぶと OpenCV ネイティブ層での競合になりうるため、追加で猶予を与える（2026-07 レビュー）。
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
    """録画スレッドへ HUD 付き表示画像を1枚受け渡す単一スロット（§7.3 セッション動画）。

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
    """セッション全長の閲覧用動画（raw/depth）を録画する専用スレッド（§7.3）。

    grabber・メインループの計算ホットパスとは完全に分離する: raw は grabber.latest()
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
    # 実行を跨いだ連番（000000..）の上書き・混在を論理的に防ぐ（§9.2 のクリーン再現は
    # 1セッション内のクリーンな連番が前提）。既存かつ非空のディレクトリを黙って再利用すると
    # 連番が 000000 から上書きされ新旧フレームが混在した「キメラセッション」になり、
    # verify_clip_replay の自己整合チェックでは検出できない（2026-07 レビューで確認）ため、
    # 録画開始前（フレーム取得前）に raise で止める。空ディレクトリは許可する。
    # session は無検証で Path 結合すると `..` や絶対パスで captures/outputs の外へ書けてしまう。
    # 単一のディレクトリ名だけを許可する（2026-09 レビュー）。
    # 注: Windows の pathlib では Path("..").name == ".." となり name 比較だけでは弾けないため、
    # カレント/親ディレクトリ参照は明示的に除外する（2026-09 レビューの回帰テストで検出）。
    if not session or session in {".", ".."} or Path(session).name != session:
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
    """セッション動画（§7.3）の raw/depth 出力先。captures/outputs の既存分離を踏襲する。"""
    return captures_dir / config.SESSION_VIDEO_RAW_NAME, outputs_dir / config.SESSION_VIDEO_DEPTH_NAME


def _write_session_meta(outputs_dir: Path, params: DepthParams, params_path: str | None,
                        source: FrameSource, *, record_video: bool = False) -> None:
    """録画セッションの使用 params・config・起動条件を outputs/{session}/meta.json に記録する（B-6）。

    「録画時と同じ params を使う」制約（§9.2）が人間の記憶頼みだと、A-2 の既定変更（live）以降は
    過去セッション（baseline 録画）との不一致事故が起きやすい。verify_clip_replay.py がこれを
    自動参照できるようにする。

    session_video は record_video が False でも常に記録する（{"enabled": false}）。後からセッションを
    見たとき「動画が無いのは仕様か、本機能実装前の古いセッションか」を meta.json だけで判別できる
    ようにするため（§7.3）。
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
        },
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_type": type(source).__name__,
        "session_video": session_video,
    }
    (outputs_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _draw_hud(img: np.ndarray, *, fps: float | None, ms: float, valid: int, frame_id: int) -> None:
    """表示画像へ cFPS / ms / 有効画素数 / frame_id をオーバレイ（ASCII のみ・in-place）。

    黒の太線 + 白の細線で、背景色に依らず読めるようにする。HUD は表示専用画像にのみ描き、
    保存する viz（クリーン）には描かない。

    表記が `cFPS`（compute FPS）なのは、この値が preprocess + process_frame 区間だけの
    所要時間 dt の逆数の EMA であり、入力待ち・表示・同期保存・GUI を含まないため
    （2026-09 レビュー）。実効的な更新レートや録画スループット（E2E throughput）ではない。
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
    window_name: str = "rt-tilted-dfd",
    log_every: int = 30,
    params_path: str | None = None,
) -> None:
    """ライブ取得→処理→表示→（消費フレームのみ）録画のループ本体。

    source は context manager で開き、取得スレッドを起動してから計算ループへ入る。終了・例外時は
    「取得スレッド停止 → join → source release → ウィンドウ破棄」を必ず通す。`with` の __enter__ が
    raise した場合 __exit__ は呼ばれないが、source 側（camera.py）が掴んだハンドルを自前で release
    する規約なので、ここでは外側 finally でウィンドウ破棄のみ保証すればよい。

    params_path: 録画セッションの meta.json（B-6）に記録する起動時の --params 文字列（任意）。
    record_video: セッション全長の閲覧用動画（raw/depth・§7.3）を録画するか。既定 False（オプトイン）。
        raw は前処理後 FOV の連続 ~SESSION_VIDEO_FPS、depth は表示そのままをサンプル&ホールドで記録し、
        専用スレッド（_SessionVideoRecorder）で行うため計算ループの dt 計測窓には一切干渉しない。
    colorbar: 距離→色の対応を示す凡例（draw_colorbar）を焼き込むか。既定 True。to_colormap 直後・
        HUD 描画より前に color へ焼き込むため、HUD（FPS 等・ライブ表示とセッション動画のみ）とは
        異なり、保存する viz.png にも含まれる（viz.png は human-viewing 専用で数値検証には使われない
        ため、焼き込んでも§9.2 の再現一致検証やevaluate.py の回帰には影響しない）。
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
                    proc = preprocess(raw)             # raw -> (352,528,3)（runner が適用・§4.3）
                    depth = process_frame(proc, params)  # -> depth_map（NaN 伝播・gate1 のみ適用済み）
                    dt = time.perf_counter() - t0      # 純粋な per-frame 計算時間（Phase D ベンチの素データ）

                    color = to_colormap(depth, config.DEPTH_MIN, config.DEPTH_MAX)  # クリーン版を1回だけ生成
                    if colorbar:
                        draw_colorbar(color, config.DEPTH_MIN, config.DEPTH_MAX)  # 保存する viz にも含める

                    if record:                         # 消費フレームのみ保存（raw->captures, depth->outputs）
                        idx = processed
                        cv2.imwrite(str(captures_dir / f"{idx:06d}.png"), raw)   # 可逆PNG（既定圧縮・§9.2 canonical）
                        np.save(str(outputs_dir / f"{idx:06d}.npy"), depth)      # NaN 保持の float32 を厳密保存
                        if save_viz:
                            cv2.imwrite(str(outputs_dir / f"{idx:06d}_viz.png"), color)  # HUD 無しのクリーン viz

                    inst_fps = (1.0 / dt) if dt > 0 else float("inf")
                    ema_fps = inst_fps if ema_fps is None else 0.9 * ema_fps + 0.1 * inst_fps
                    n_valid = int(np.count_nonzero(~np.isnan(depth)))
                    if log_every > 0 and (processed % log_every == 0):
                        logger.info("processed=%d (frame id=%d): %.1f ms (~%.2f fps), valid=%d",
                                    processed, fid, dt * 1000.0, inst_fps, n_valid)
                    processed += 1                      # 連番 index と処理数（先頭=0。log は 0,log_every,... で発火）

                    _draw_hud(color, fps=ema_fps, ms=dt * 1000.0, valid=n_valid, frame_id=fid)
                    if display_slot is not None:        # セッション動画の depth 側（サンプル&ホールドの供給元）
                        display_slot.update(color)
                    cv2.imshow(window_name, color)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):          # q or ESC で終了
                        break
            finally:
                if recorder is not None:
                    recorder.stop_and_join()           # grabber より先に止める（セッション終了後の書き込みを止める）
                grabber.stop_and_join()
                if sys.exc_info()[0] is not None:
                    # 計算ループ側の例外で finally に入った場合、grabber/recorder 側の例外で
                    # 元例外を置換しない（真因を保持する・2026-07 レビュー）。記録のみ行う。
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

    parser = argparse.ArgumentParser(description="rt-tilted-dfd リアルタイム実行（Phase C）")
    parser.add_argument("--source", choices=("webcam", "video", "still"), default="webcam")
    parser.add_argument("--path", default=None, help="video/still 用の入力パス")
    parser.add_argument("--params", default=str(config.PARAMS_PATH),
                        help="fit_result.csv のパス（既定: live＝root。baseline を使う場合は明示指定）")
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

    params = load_params(args.params)                  # 起動時に1回だけ（ループ内で再計算しない・§4.1）
    source = _build_source(args)
    run(source, params, record=not args.no_record, save_viz=not args.no_viz,
        record_video=args.record_video, colorbar=not args.no_colorbar,
        session=args.session, params_path=args.params)


if __name__ == "__main__":
    main()