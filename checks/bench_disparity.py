"""checks/bench_disparity.py — Phase D 用ベンチ（§8.2 実測ベンチ駆動）。

calcDisparityByZNCC 単体と process_frame 全体の ms/frame を実測し、Phase D の
Python↔Rust 同条件比較（§8.3）の基礎データを取る。負荷は E(エッジ画素数)×S(search)×K²
に比例し E は画像内容依存のため、時間だけでなく E を必ず併記する。

実行:
    uv run python -m checks.bench_disparity --source synthetic
    uv run python -m checks.bench_disparity --source test
    uv run python -m checks.bench_disparity --source captures --session clip92
"""
# --- checks/ から root の config / core / realtime を import するためのブートストラップ ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time

import cv2
import numpy as np

import config
from core.disparity import calcDisparityByZNCC
from core.params import DepthParams
from realtime.camera import preprocess
from realtime.pipeline import process_frame


# ベンチ専用のダミー係数（時間計測のみが目的。実距離換算には使わない・§6.3 の値とは無関係）
_BENCH_PARAMS = DepthParams(
    v_pos=400000.0, u_pos=50.0, a_pos=-30.0, w_pos=0.0,
    v_neg=400000.0, u_neg=50.0, a_neg=-30.0, w_neg=0.0,
)


def _load_synthetic() -> tuple[str, np.ndarray]:
    """乱数テクスチャ（seed 固定）。生の一様乱数はエッジ密度が90%超になり実写(2300.jpg実測5%)
    より1桁重い非現実的なベンチになるため、軽くぼかしてエッジ密度を実写相当に近づける。
    """
    rng = np.random.default_rng(0)
    h, w = config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]
    frame = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (15, 15), 0)
    return "synthetic(seed=0,blur=15)", frame


def _load_test() -> tuple[str, np.ndarray]:
    jpgs = sorted(config.TEST_DIR.glob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"data/test/*.jpg が見つかりません: {config.TEST_DIR}")
    path = jpgs[0]
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"画像を読めません: {path}")
    return path.name, cv2.resize(img, config.IMAGE_SIZE)


def _load_captures(session: str) -> tuple[str, np.ndarray]:
    cap_dir = Path(config.CAPTURES_DIR) / session
    pngs = sorted(p for p in cap_dir.glob("??????.png") if p.stem.isdigit())
    if not pngs:
        raise FileNotFoundError(f"{cap_dir} に連番 png(??????.png)がありません")
    path = pngs[0]
    raw = cv2.imread(str(path))
    if raw is None:
        raise RuntimeError(f"png を読めません: {path}")
    return path.name, preprocess(raw)


def _edge_pixel_count(red: np.ndarray) -> int:
    """core/disparity.py 内部と同じ手順でエッジ画素数 E を独立算出する（表示専用・core は読むだけ）。"""
    edge_red = cv2.convertScaleAbs(cv2.Sobel(red, cv2.CV_32F, 1, 0))
    _, edge = cv2.threshold(edge_red, config.EDGE_TH, 255, cv2.THRESH_TOZERO)
    return int(np.count_nonzero(edge))


def _bench(fn, n: int) -> tuple[float, float]:
    """fn を n 回計測し (平均ms, 最小ms) を返す。"""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.min(times))


def main(argv: list[str] | None = None) -> int:
    # Windows の日本語コンソール(cp932)は非対応文字の print で UnicodeEncodeError を出すため
    # UTF-8 に固定する（runner.py:252-255 と同じ対処）。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Phase D 用ベンチ: disparity/process_frame の ms/frame 計測")
    parser.add_argument("--source", choices=("test", "captures", "synthetic"), default="synthetic")
    parser.add_argument("--session", default=None, help="--source captures 用のセッション名")
    parser.add_argument("--n", type=int, default=5, help="計測回数（既定5）")
    args = parser.parse_args(argv)

    if args.source == "synthetic":
        name, frame = _load_synthetic()
    elif args.source == "test":
        name, frame = _load_test()
    else:
        if not args.session:
            raise ValueError("--source captures には --session が必要です")
        name, frame = _load_captures(args.session)

    blue, _, red = cv2.split(frame)
    E = _edge_pixel_count(red)

    disp_holder: dict[str, np.ndarray] = {}

    def run_disparity():
        disp_holder["d"] = calcDisparityByZNCC(
            red, blue, config.KERNEL_SIZE, config.SEARCH_RANGE_X, config.EDGE_TH)

    def run_pipeline():
        process_frame(frame, _BENCH_PARAMS)

    mean_d, min_d = _bench(run_disparity, args.n)
    n_valid = int(np.count_nonzero(~np.isnan(disp_holder["d"])))
    mean_p, min_p = _bench(run_pipeline, args.n)

    print(f"入力: {name}  shape={frame.shape}  N={args.n}")
    print(f"E(エッジ画素数)={E}  S(search)={config.SEARCH_RANGE_X}  K(kernel)={config.KERNEL_SIZE}  "
          f"有効視差画素数={n_valid}")
    print(f"calcDisparityByZNCC: 平均={mean_d:.1f}ms  最小={min_d:.1f}ms")
    print(f"process_frame(全体):  平均={mean_p:.1f}ms  最小={min_p:.1f}ms")

    # Phase D の同条件比較にそのまま使える機械可読な1行サマリ（§8.3）
    print(f"BENCH source={args.source} name={name} E={E} S={config.SEARCH_RANGE_X} "
          f"K={config.KERNEL_SIZE} n_valid={n_valid} "
          f"disp_mean_ms={mean_d:.3f} disp_min_ms={min_d:.3f} "
          f"pipeline_mean_ms={mean_p:.3f} pipeline_min_ms={min_p:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
