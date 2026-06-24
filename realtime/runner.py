"""realtime/runner.py — オンライン実行のループ本体（要件書 §5.5・§7.2・§7.3）。

データフロー（§4.3 オンライン）を組み立てる:
    source.read() -> raw(1280x720) -> preprocess -> proc(352,528,3)
      -> pipeline.process_frame(proc, params) -> depth_map(NaN 伝播)
      -> 表示（gate2 を vmin/vmax で適用・カラーマップ）＋FPS
      -> 録画（消費フレームのみ: raw->captures/, depth->outputs/）

実行: リポジトリ root から `uv run python -m realtime.runner --source webcam`
      （camera.py と同じく config を絶対 import するため、パッケージとして起動する）。

スレッドモデル（§7.2）: 取得スレッドが source.read() を回し、最新フレームを Lock 保護の
単一スロットへ「上書き」する（キューに溜めない）。計算ループ（メインスレッド）はスロットから
最新フレームのみを消費し、間に合わなかった中間フレームは自然に破棄される（遅延が蓄積しない）。
Phase C では計算（ZNCC ~14s/frame）がカメラ(30fps)より遥かに遅く、計算が取得を追い越す
状況は設計上起きないため、同期は Lock のみで足りる（新フレーム待ちの Condition は不要）。

gate2（Z レンジ 0<z<50000）はここ（表示側）で vmin/vmax として適用し、process_frame が
返す depth マップは書き換えない（書き換えると evaluate.py の depth マップとビット一致しなく
なる・§9.2）。録画する depth は NaN を保持した float32 をそのまま np.save する（唯一の真実）。
表示用カラーマップは使い捨てコピー上で NaN を潰すだけで、元 depth には触れない。

realtime/ は core ではない（config import・I/O・cv2・threading を持ってよい・§4.1）。
計算の中核は持たず pipeline.process_frame（→ core 純関数）を呼ぶだけ。
入力検証は raise（assert は -O で無効化されるため使わない・リスク#10）。

注: 計算が1フレーム ~14s（Phase C）の間、cv2 ウィンドウのイベント処理（waitKey）は
    走らないためウィンドウは一時的に固まる（最初の表示も初回処理後＝起動から ~14s 後）。
    これは Phase C の既知挙動で、Phase D で 30fps 化すれば解消する。

§9.2 のクリップ再生「再現一致」検証（録った raw を1フレームずつ決定的に再生してライブ depth と
一致を確認）は、フレームを間引かない別ドライバの役割（連番PNG再生 source の camera.py 追加と
あわせて別途用意する）。本 runner はライブ取得（中間破棄あり）が責務。
"""
from __future__ import annotations

import argparse
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
    動画として見たとき色変化＝被写体変化と読める）。NaN（測定不能）は近接(z≈0=青)と
    弁別するため専用色（黒）で塗る。車載では警告/制動などの判断は元 depth(npy)を読む前提で、
    これはあくまで人間が眺めるための投影（情報を落とした見た目であり判断根拠にはしない）。
    """
    invalid = np.isnan(depth)                         # 無効画素の位置を先に確保（潰す前に）
    scaled = (np.nan_to_num(depth, nan=vmin) - vmin) / (vmax - vmin) * 255.0
    u8 = np.clip(scaled, 0, 255).astype(np.uint8)     # NaN を uint8 へ直接 astype すると不定値化するため必ず潰してから
    color = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    color[invalid] = (0, 0, 0)                        # 無効は黒（青=近接 と一目で弁別）
    return color


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

    def raise_if_error(self) -> None:
        if self._error is not None:
            raise self._error


def _ensure_session_dirs(session: str) -> tuple[Path, Path]:
    # 録画を実行（セッション）ごとに分離する。captures/{session}, outputs/{session} に保存する
    # 実行を跨いだ連番（000000..）の上書き・混在を論理的に防ぐ（§9.2 のクリーン再現は
    # 1セッション内のクリーンな連番が前提）。
    captures = Path(config.CAPTURES_DIR) / session
    outputs = Path(config.OUTPUTS_DIR) / session
    captures.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    return captures, outputs


def _draw_hud(img: np.ndarray, *, fps: float | None, ms: float, valid: int, frame_id: int) -> None:
    """表示画像へ FPS / ms / 有効画素数 / frame_id をオーバレイ（ASCII のみ・in-place）。

    黒の太線 + 白の細線で、背景色に依らず読めるようにする。HUD は表示専用画像にのみ描き、
    保存する viz（クリーン）には描かない。
    """
    fps_txt = f"{fps:.2f}" if fps is not None else "n/a"
    lines = [f"FPS: {fps_txt}  ms/frame: {ms:.0f}", f"valid: {valid}  id: {frame_id}"]
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
    session: str | None = None,
    window_name: str = "rt-tilted-dfd",
    log_every: int = 30,
) -> None:
    """ライブ取得→処理→表示→（消費フレームのみ）録画のループ本体。

    source は context manager で開き、取得スレッドを起動してから計算ループへ入る。終了・例外時は
    「取得スレッド停止 → join → source release → ウィンドウ破棄」を必ず通す。`with` の __enter__ が
    raise した場合 __exit__ は呼ばれないが、source 側（camera.py）が掴んだハンドルを自前で release
    する規約なので、ここでは外側 finally でウィンドウ破棄のみ保証すればよい。
    """
    if not isinstance(source, FrameSource):
        raise ValueError(f"run: source は FrameSource を想定。実際は {type(source)!r}")

    if record:
        session = session or time.strftime("%Y%m%d_%H%M%S")  # 既定は起動時刻。--session で固定可
        captures_dir, outputs_dir = _ensure_session_dirs(session)
        logger.info("録画セッション: captures=%s, outputs=%s", captures_dir, outputs_dir)
    else:
        captures_dir, outputs_dir = None, None
    processed = 0                                     # 処理フレーム数。captures/outputs の連番 index と 1:1
    last_id = -1
    ema_fps: float | None = None                      # 処理レートの指数移動平均（表示用の平滑値）

    try:
        with source as src:                           # __enter__ で open
            grabber = _FrameGrabber(src)
            grabber.start()
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
                    cv2.imshow(window_name, color)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):          # q or ESC で終了
                        break
            finally:
                grabber.stop_and_join()
                grabber.raise_if_error()               # 取得スレッドの例外を顕在化（無ければ no-op）
    finally:
        cv2.destroyAllWindows()

    logger.info("runner 終了: 処理フレーム数=%d, 最終 fps~%s",
                processed, f"{ema_fps:.3f}" if ema_fps is not None else "n/a")


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
    parser.add_argument("--params", default=str(config.BASELINE_PARAMS_PATH),
                        help="fit_result.csv のパス（既定: baseline。FR-8 後に live へ差し替え）")
    parser.add_argument("--loop", action="store_true", help="video source を EOF で巻き戻して継続")
    parser.add_argument("--no-record", action="store_true", help="録画を無効化")
    parser.add_argument("--no-viz", action="store_true", help="colormap PNG(viz) の保存を無効化")
    parser.add_argument("--session", default=None,
                        help="録画セッション名（既定: 起動時刻 YYYYmmdd_HHMMSS）。"
                             "captures/{session}, outputs/{session} に保存")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    params = load_params(args.params)                  # 起動時に1回だけ（ループ内で再計算しない・§4.1）
    source = _build_source(args)
    run(source, params, record=not args.no_record, save_viz=not args.no_viz,
        session=args.session)


if __name__ == "__main__":
    main()