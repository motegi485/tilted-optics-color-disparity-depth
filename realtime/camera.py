"""realtime/camera.py — フレーム供給の抽象化と共有 preprocess（project_knowledge/modules/module-realtime.md）。

各 FrameSource は raw（WEBCAM_NATIVE_SIZE=1280x720 BGR）を返し、共有 preprocess() を
全 source 共通に適用する。これにより live と clip 再生が preprocess 以降ビット一致し、
project_knowledge/requirements/requirements.md に記載の「ライブ／クリップ再生での再現一致」を担保する。録画は raw を captures/ に保存する
（processed を録らない。同じ raw を同じ preprocess に通せば再現するため・project_knowledge/modules/module-realtime.md）。

realtime/ は core ではない: config import・I/O・cv2・logging を持ってよい（project_knowledge/architecture/architecture.md）。
計算の中核（視差・深度）は持たず、pipeline.py が core 純関数を呼ぶ。
入力検証は raise（assert は -O で無効化されるため使わない・project_knowledge/implementation/known-issues.md #10）。
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


def preprocess(raw: np.ndarray) -> np.ndarray:
    """raw(1280x720 BGR) -> (352,528,3) BGR。横中央クロップで 3:2 化 → INTER_AREA 縮小。

    全 source 共通の唯一の前処理。横クロップは POF 距離マッピング軸（縦 y・project_knowledge/architecture/theory.md）に
    不干渉で縦の距離分解能を保つ。縮小は INTER_AREA でエイリアス抑制（4:2:0 で既に
    鈍る R/B をこれ以上荒らさない・project_knowledge/modules/module-realtime.md）。
    """
    # numpy の shape は (H, W, C)。config は (W, H) 順なので (w, h) に並べ替えて比較する。
    h, w = raw.shape[:2]
    if (w, h) != config.WEBCAM_NATIVE_SIZE:
        raise ValueError(
            f"preprocess: raw は {config.WEBCAM_NATIVE_SIZE} (W,H) を想定。実際は {(w, h)}"
        )
    if config.WEBCAM_CROP_W > w:
        raise ValueError(
            f"preprocess: WEBCAM_CROP_W({config.WEBCAM_CROP_W}) が raw 幅({w})を超えています（負インデックス crop 防止）"
        )
    x0 = (w - config.WEBCAM_CROP_W) // 2            # 中央クロップの左端: (1280-1080)//2 = 100
    crop = raw[:, x0:x0 + config.WEBCAM_CROP_W]     # 縦は全行保持・横のみ 1080 に（3:2）。view なのでコピーなし
    return cv2.resize(crop, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def _verify_native(frame: np.ndarray, who: str) -> None:
    """raw が WEBCAM_NATIVE_SIZE(1280x720) であることを早期に検証する（fail fast）。

    preprocess と同じ不変条件を、各 source の open() でループに入る前に確認するためのヘルパー。
    who は失敗箇所を示す呼び出し元名（エラーメッセージ用）。
    """
    h, w = frame.shape[:2]
    if (w, h) != config.WEBCAM_NATIVE_SIZE:        # preprocess と同じく (W,H) 順で比較
        raise ValueError(
            f"{who}: raw は {config.WEBCAM_NATIVE_SIZE} (W,H) を想定。実際は {(w, h)}"
        )


class FrameSource(ABC):
    """raw フレーム供給の抽象。webcam/動画/静止画を同一 IF で差し替え可能にする。

    read() は raw（WEBCAM_NATIVE_SIZE BGR）を返す。preprocess は呼び出し側（runner）が
    共有 preprocess() で行う（source 内では行わない＝録画する raw と processed を分離）。
    リソース解放は context manager で保証する（例外時も __exit__ で release）。
    """

    # open/read/release は source ごとに中身が違うプリミティブ（具象側で実装）。
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """-> (ok, raw_bgr)。raw は WEBCAM_NATIVE_SIZE(1280x720) BGR。失敗時 (False, None)。"""
        ...

    @abstractmethod
    def release(self) -> None: ...

    # __enter__/__exit__ は上記3つの上に乗る共通定型なので基底で1度だけ実装する。
    # これで全 source が `with` 構文と「例外時も確実に解放」を実装なしで継承する。
    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()                              # 戻り値 None ＝ 例外は握り潰さず素通りさせる


# ===== ライブ webcam: cv2.VideoCapture を CAP_MSMF で開き、実フィードの raw を返す =====
class WebcamSource(FrameSource):
    """EOS Webcam Utility Pro 経由のライブ webcam（CAP_MSMF・index=0・project_knowledge/modules/module-realtime.md）。"""

    def __init__(self, index: int | None = None, backend: int | None = None,
                 fps_probe_frames: int = 30):
        # index/backend は None なら config から解決（通常は config、テスト時のみ上書き）。
        self.index = config.CAPTURE_DEVICE_INDEX if index is None else index
        self.backend = config.CAPTURE_BACKEND if backend is None else backend
        self.fps_probe_frames = fps_probe_frames
        self.cap: cv2.VideoCapture | None = None    # None＝未オープン。このクラスの唯一の状態

    def open(self) -> None:
        if self.cap is not None:
            raise RuntimeError("WebcamSource: 多重 open です（先に release してください）")
        cap = cv2.VideoCapture(self.index, self.backend)
        if not cap.isOpened():
            cap.release()                           # 掴みかけた local ハンドルを解放してから raise
            raise RuntimeError(f"WebcamSource: VideoCapture(index={self.index}, backend={self.backend}) を開けません")
        # 解像度を要求（実測では無償枠でどう要求しても 1280x720 に張り付く・project_knowledge/requirements/requirements.md）。
        w, h = config.WEBCAM_NATIVE_SIZE
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap = cap

        # 以降の検証で失敗しても camera を掴んだまま落とさない。
        # （__enter__ が raise すると __exit__ は呼ばれず release されないため、ここで明示解放する）
        try:
            # 固定枚数 warmup の代わりに「実フィードが来るまで待つ」（来なければ raise）。
            # EOS Webcam Utility は cold-start で数秒 no-feed プレースホルダを流す（実測 ~3.3s）。
            frame = self._wait_until_live(config.LIVE_WAIT_TIMEOUT_S)
            _verify_native(frame, "WebcamSource.open")
            eff_fps = self._probe_fps()
        except BaseException:                       # Ctrl-C（最大8秒の待機中）も拾って必ず解放する
            self.release()
            raise

        logger.info("WebcamSource opened: native=%s, effective_fps≈%.1f (backend=%d)",
                    config.WEBCAM_NATIVE_SIZE, eff_fps, self.backend)

    def _wait_until_live(self, timeout_s: float, burst: int = 5, confirm: int = 2) -> np.ndarray:
        """実フィード（センサノイズで時間変動するフレーム）が来るまで待ち、最新フレームを返す。

        静止プレースホルダ（no-feed 画面）は連続フレームが完全一致（時間 std≈0）。実フィードは
        毎フレーム必ずノイズで変動（時間 std>0）。これを burst 内の時間 std で判定する。
        confirm 回連続で閾値超えを確認してから返す（cold→live 遷移時の一過性フレームに latch しない）。
        timeout 内に実フィードを検出できなければ raise（黙ってプレースホルダを掴まない・project_knowledge/architecture/architecture.md の fail-fast原則）。
        """
        if self.cap is None:
            raise RuntimeError("WebcamSource: _wait_until_live が open() 前に呼ばれました")
        deadline = time.perf_counter() + timeout_s
        last = None
        streak = 0                                  # 閾値超えバーストの連続回数
        while time.perf_counter() < deadline:
            frames = []
            for _ in range(burst):
                ok, f = self.cap.read()
                if ok and f is not None:
                    frames.append(f)
                    last = f
            if not frames:
                time.sleep(0.05)                    # read 全滅（一時的な取得失敗）時の busy loop 緩和
            elif len(frames) >= 2:
                # 時間方向の画素ばらつき: 静止画面≈0 / 実フィード>0。バーストで見るので
                # たまたま重複フレームを1枚読んでも誤判定しにくい。
                tstd = float(np.stack(frames).astype(np.float32).std(axis=0).mean())
                if tstd > config.LIVE_DIFF_STD_TH:
                    streak += 1
                    if streak >= confirm:
                        return last
                else:
                    streak = 0                      # 非ライブが挟まったら連続カウントをリセット
        raise RuntimeError(
            "WebcamSource: 実フィードを検出できません（プレースホルダのみ）。"
            "カメラを動画モードで起こし、ライブ供給されているか確認してください")

    def _probe_fps(self) -> float:
        """open() 内で一度だけ実効 fps を実測（連続 fps は runner 側で計測・project_knowledge/modules/module-realtime.md）。

        カメラ自己申告の CAP_PROP_FPS は仮想 UVC では当てにならないため、実際に読めた
        枚数/経過時間で測る。
        """
        if self.cap is None:
            raise RuntimeError("WebcamSource: _probe_fps が open() 前に呼ばれました")
        t0 = time.perf_counter()
        n = 0
        for _ in range(self.fps_probe_frames):
            ok, _f = self.cap.read()
            if ok:
                n += 1
        dt = time.perf_counter() - t0
        return (n / dt) if dt > 0 else float("nan")

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            raise RuntimeError("WebcamSource: open() 前に read() が呼ばれました")
        # サイズ検証はここではしない（毎フレームのホットパス）。番人は preprocess。
        return self.cap.read()

    def release(self) -> None:
        if self.cap is not None:                    # None ガードで冪等（多重 release・open 失敗時にも安全）
            self.cap.release()
            self.cap = None


# ===== 録画クリップ: 保存した raw クリップ（1280x720）を順送りで返す =====
class VideoFileSource(FrameSource):
    """録画した raw クリップ（1280x720）を順送り（カメラ非接続デバッグ・再現実行・project_knowledge/modules/module-realtime.md）。

    ※ project_knowledge/requirements/requirements.md に記載のビット一致再現に使うには、録画が「無劣化（lossless / 連番PNG等）」である
       必要がある（圧縮クリップは閲覧デバッグ用。録画形式は project_knowledge/modules/module-realtime.md で確定）。
    """

    def __init__(self, path: str | Path, loop: bool = False):
        self.path = Path(path)
        self.loop = loop                            # True: EOF で巻き戻して継続。False: EOF で停止
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self.cap is not None:
            raise RuntimeError("VideoFileSource: 多重 open です（先に release してください）")
        if not self.path.exists():
            raise FileNotFoundError(f"VideoFileSource: クリップが見つかりません: {self.path}")
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"VideoFileSource: クリップを開けません: {self.path}")
        self.cap = cap

        # 検証で失敗しても掴んだ cap を解放してから raise（WebcamSource.open と同じ理由）。
        try:
            ok, f = cap.read()                      # 先頭フレームで raw サイズを検証
            if not ok or f is None:
                raise RuntimeError(f"VideoFileSource: 先頭フレームを読めません: {self.path}")
            _verify_native(f, "VideoFileSource.open")
            rewound = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 検証で進めた位置を巻き戻す（lossless 前提）
            if not rewound:
                # バックエンドによっては set() が失敗することがある。release→再 open で確実に先頭へ戻す。
                cap.release()
                cap = cv2.VideoCapture(str(self.path))
                if not cap.isOpened():
                    raise RuntimeError(f"VideoFileSource: 巻き戻しのための再 open に失敗しました: {self.path}")
                self.cap = cap
        except BaseException:
            self.release()
            raise

        logger.info("VideoFileSource opened: %s (loop=%s)", self.path, self.loop)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            raise RuntimeError("VideoFileSource: open() 前に read() が呼ばれました")
        ok, frame = self.cap.read()
        if not ok or frame is None:                 # EOF
            if self.loop:
                # backend によっては set() が False を返して巻き戻せない。open() と同じく
                # release→再 open で確実に先頭へ戻す（seek 失敗を無音の EOF にしない）。
                if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    logger.warning(
                        "VideoFileSource: 先頭への seek に失敗しました。再 open します: %s", self.path)
                    self.cap.release()
                    self.cap = None
                    cap = cv2.VideoCapture(str(self.path))
                    if not cap.isOpened():
                        cap.release()
                        raise RuntimeError(
                            f"VideoFileSource: loop のための再 open に失敗しました: {self.path}")
                    self.cap = cap
                ok, frame = self.cap.read()
                return (ok, frame if ok else None)
            return False, None
        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ===== 静止画: 単一の raw（1280x720）を繰り返し返す =====
class StillImageSource(FrameSource):
    """単一の raw 静止画（1280x720 BGR）を繰り返し返す（配線テスト・カメラ非接続）。

    想定入力は「保存した capture の1フレーム（raw 1280x720）」。共有 preprocess を通す
    前提のため raw サイズを検証する。任意サイズの一般テスト画像は対象外（それは
    evaluate.py + params/baseline/ の役割・project_knowledge/architecture/architecture.md）。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.frame: np.ndarray | None = None        # 読み込んだ raw（OS ハンドルは持たない）

    def open(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"StillImageSource: 画像が見つかりません: {self.path}")
        frame = cv2.imread(str(self.path))          # BGR
        if frame is None:
            raise RuntimeError(f"StillImageSource: 画像を読めません: {self.path}")
        _verify_native(frame, "StillImageSource.open")
        self.frame = frame
        logger.info("StillImageSource opened: %s", self.path)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.frame is None:
            raise RuntimeError("StillImageSource: open() 前に read() が呼ばれました")
        return True, self.frame.copy()              # コピーを返し、下流の in-place 変更から元を保護

    def release(self) -> None:
        self.frame = None