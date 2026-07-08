"""tools/probe_webcam.py
Phase C 実機計測プローブ（診断用・使い捨て）。
EOS R10 + EOS Webcam Utility Pro 経由の実フレーム解像度・実効 fps・色特性を実測する。
core/ には依存しない。production コードではない。診断用ツールのため実行は root から行うこと
（`uv run python tools/probe_webcam.py`）。
"""
# --- tools/ から root の config を import するためのブートストラップ ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import time
import json
from datetime import datetime

import config

# Windows の日本語コンソール（cp932）では非対応文字（≈ 等）の print が
# UnicodeEncodeError で落ちる。それで計測自体が無に帰すのを防ぐため UTF-8 に固定。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 出力先（いずれも git 追跡外の想定）。mkdir は main() 内で行う（import 時の副作用を避ける）。
CAPTURE_DIR = config.CAPTURES_DIR / "probe"
OUTPUT_DIR = config.OUTPUTS_DIR

DEVICE_INDEX = 0
BACKENDS = [("CAP_DSHOW", cv2.CAP_DSHOW), ("CAP_MSMF", cv2.CAP_MSMF)]
# 解像度モード探索の候補（要求 → 実際に通る値を確認）
PROBE_MODES = [(640, 360), (640, 480), (1024, 576), (1280, 720), (1920, 1080)]
FPS_FRAMES = 300   # 実効 fps 計測のフレーム数
WARMUP = 30        # ウォームアップで破棄するフレーム数


def fourcc_to_str(v) -> str:
    v = int(v)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


def read_props(cap) -> dict:
    return {
        "prop_width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "prop_height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "prop_fps": cap.get(cv2.CAP_PROP_FPS),
        "fourcc": fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
    }


def horizontal_hf_energy(channel) -> float:
    """横方向の高周波エネルギー（Sobel-x の分散）。chroma subsampling の相対指標。"""
    gx = cv2.Sobel(channel.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    return float(np.var(gx))


def analyze_color(frame) -> dict:
    b, g, r = cv2.split(frame)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    return {
        "mean_bgr": [float(b.mean()), float(g.mean()), float(r.mean())],
        "std_bgr": [float(b.std()), float(g.std()), float(r.std())],
        # Y vs Cr/Cb の横方向高周波比（4:2:0/4:2:2 の間接的兆候）
        "hf_horizontal_YCrCb": {
            "Y": horizontal_hf_energy(y),
            "Cr": horizontal_hf_energy(cr),
            "Cb": horizontal_hf_energy(cb),
        },
        # R/G/B 個別の横方向高周波（本手法の信号は R/B の横ずれ）
        "hf_horizontal_BGR": {
            "B": horizontal_hf_energy(b),
            "G": horizontal_hf_energy(g),
            "R": horizontal_hf_energy(r),
        },
    }


def probe_backend(name: str, flag: int) -> dict:
    entry: dict = {}
    cap = cv2.VideoCapture(DEVICE_INDEX, flag)
    if not cap.isOpened():
        cap.release()
        entry["opened"] = False
        print(f"[{name}] open failed")
        return entry
    entry["opened"] = True
    entry["default_props"] = read_props(cap)

    # ウォームアップ + 実フレーム shape
    frame = None
    for _ in range(WARMUP + 1):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    if frame is None:
        entry["actual_frame_shape_HWC"] = None
        entry["note"] = "opened but read() returned no frame"
        cap.release()
        print(f"[{name}] opened but no frame")
        return entry
    entry["actual_frame_shape_HWC"] = list(frame.shape)  # (H, W, C)
    entry["dtype"] = str(frame.dtype)

    # 実効 fps（default/native モードで計測）
    t0 = time.perf_counter()
    n = 0
    for _ in range(FPS_FRAMES):
        ok, _f = cap.read()
        if ok:
            n += 1
    dt = time.perf_counter() - t0
    entry["effective_fps"] = (n / dt) if dt > 0 else None
    entry["fps_frames_read"] = n

    # 色特性 + サンプル保存（PNG 無圧縮。JPEG だと追加で圧縮/サブサンプリングが乗るため）
    entry["color"] = analyze_color(frame)
    cv2.imwrite(str(CAPTURE_DIR / f"sample_{name}.png"), frame)

    # WB 安定性（連続フレームの R/B mean の時間変動。WB 固定の効きを見る）
    rb_means = []
    for _ in range(60):
        ok, f = cap.read()
        if ok and f is not None:
            b2, g2, r2 = cv2.split(f)
            rb_means.append([float(r2.mean()), float(b2.mean())])
    if len(rb_means) > 1:
        arr = np.array(rb_means)
        entry["wb_stability"] = {
            "R_mean_std_over_time": float(arr[:, 0].std()),
            "B_mean_std_over_time": float(arr[:, 1].std()),
            "frames": len(rb_means),
        }

    # 解像度モード探索（set → read-back。set が成功を返しても実際は変わらないことがある）
    modes = []
    for (w, h) in PROBE_MODES:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        ok, f = cap.read()
        got = [int(f.shape[1]), int(f.shape[0])] if (ok and f is not None) else None  # [W, H]
        modes.append({
            "requested_WH": [w, h],
            "got_WH": got,
            "fourcc": fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
        })
    entry["resolution_modes"] = modes

    cap.release()
    print(f"[{name}] done: shape={entry['actual_frame_shape_HWC']} "
          f"fps~={entry.get('effective_fps')}")
    return entry


def main():
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "opencv_version": cv2.__version__,
        "device_index": DEVICE_INDEX,
        "backends": {},
    }
    for name, flag in BACKENDS:
        try:
            report["backends"][name] = probe_backend(name, flag)
        except Exception as e:
            report["backends"][name] = {"opened": False, "exception": f"{type(e).__name__}: {e}"}
            print(f"[{name}] exception: {type(e).__name__}: {e}")

    if all(not b.get("opened") for b in report["backends"].values()):
        report["error"] = ("どちらのバックエンドでも device index 0 が開きませんでした。"
                           "EOS Webcam Utility Pro の起動・カメラ電源・接続を確認してください。")
        print(report["error"])

    out = OUTPUT_DIR / "probe_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport -> {out}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
