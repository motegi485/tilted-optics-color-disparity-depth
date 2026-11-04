"""checks/bench_disparity.py — 高速化（Rust移植）用ベンチ（project_knowledge/modules/module-core.md の実測ベンチ駆動の考え方）。

calcDisparityByZNCC 単体と process_frame 全体の ms/frame を実測し、
Python↔Rust 同条件比較（project_knowledge/implementation/verification-and-tools.md）の基礎データを取る。負荷は E(エッジ画素数)×S(search)×K²
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
from core.params import DepthParams
from realtime.camera import preprocess
from realtime.pipeline import process_frame


# ベンチ専用のダミー係数（時間計測のみが目的。実距離換算には使わない・project_knowledge/modules/module-calibration.md の値とは無関係）
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
    jpgs = sorted(config.TEST_DIR.glob("*.jpg", case_sensitive=False))
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


def _bench(fn, n: int) -> tuple[float, float, float]:
    """fn を1回ウォームアップ実行（計測対象外。rayonスレッドプール初期化等を計測から除く）した後、
    n 回計測し (平均ms, 最小ms, p95ms) を返す。"""
    fn()  # warmup
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.min(times)), float(np.percentile(times, 95))


def _resolve_disparity_fn(impl: str):
    """--impl python: core._disparity_reference（凍結 Python 参照実装）。
    --impl native: core.disparity（Rust 実装への薄いラッパー）。
    """
    if impl == "python":
        from core._disparity_reference import calcDisparityByZNCC
        return calcDisparityByZNCC
    from core.disparity import calcDisparityByZNCC
    return calcDisparityByZNCC


def main(argv: list[str] | None = None) -> int:
    # Windows の日本語コンソール(cp932)は非対応文字の print で UnicodeEncodeError を出すため
    # UTF-8 に固定する（realtime/runner.py の main() 冒頭と同じ対処）。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="disparity/process_frame の ms/frame 計測（30fps バジェット判定）")
    parser.add_argument("--source", choices=("test", "captures", "synthetic"), default="synthetic")
    parser.add_argument("--session", default=None, help="--source captures 用のセッション名")
    parser.add_argument("--n", type=int, default=5, help="計測回数（既定5。1回のwarmupは別途・計測対象外）")
    parser.add_argument("--impl", choices=("native", "python"), default="native",
                         help="native=core.disparity（Rust 実装への薄いラッパー） / "
                              "python=core._disparity_reference（既定native）")
    args = parser.parse_args(argv)

    # n<=0 だと _bench の times が空になり np.min([]) が ValueError を送出する（計測0件を
    # 「成功」とも「失敗」とも言えないため、入力不備として exit 2 で早期に止める）。
    if args.n < 1:
        print(f"[ERROR] --n は 1 以上が必要です（渡された値: {args.n}）")
        return 2

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

    calcDisparityByZNCC = _resolve_disparity_fn(args.impl)

    disp_holder: dict[str, np.ndarray] = {}

    def run_disparity():
        disp_holder["d"] = calcDisparityByZNCC(
            red, blue, config.KERNEL_SIZE, config.SEARCH_RANGE_X, config.EDGE_TH)

    def run_pipeline():
        process_frame(frame, _BENCH_PARAMS)

    mean_d, min_d, p95_d = _bench(run_disparity, args.n)
    n_valid = int(np.count_nonzero(~np.isnan(disp_holder["d"])))

    print(f"入力: {name}  shape={frame.shape}  N={args.n}  impl={args.impl}")
    print(f"E(エッジ画素数)={E}  S(search)={config.SEARCH_RANGE_X}  K(kernel)={config.KERNEL_SIZE}  "
          f"有効視差画素数={n_valid}")
    print(f"calcDisparityByZNCC[{args.impl}]: 平均={mean_d:.2f}ms  最小={min_d:.2f}ms  p95={p95_d:.2f}ms")

    # process_frame(全体) は realtime/pipeline.py 内で `from core.disparity import
    # calcDisparityByZNCC` と固定 import しているため、--impl の値によらず常に core.disparity
    # （Rust 実装）を使う。--impl python 選択時は disparity 単体の数値のみが
    # Python 参照実装になり、process_frame(全体) の数値とは実装が食い違う点に注意。
    mean_p, min_p, p95_p = _bench(run_pipeline, args.n)
    print(f"process_frame(全体・常にcore.disparity経由): 平均={mean_p:.2f}ms  最小={min_p:.2f}ms  p95={p95_p:.2f}ms"
          + ("" if args.impl == "native" else "  [注意: disparity単体はpython参照実装だがこちらは常にcore.disparity]"))

    # 実装間・設定間の同条件比較にそのまま使える機械可読な1行サマリ（project_knowledge/implementation/verification-and-tools.md）
    print(f"BENCH source={args.source} name={name} impl={args.impl} E={E} S={config.SEARCH_RANGE_X} "
          f"K={config.KERNEL_SIZE} n_valid={n_valid} "
          f"disp_mean_ms={mean_d:.3f} disp_min_ms={min_d:.3f} disp_p95_ms={p95_d:.3f} "
          f"pipeline_mean_ms={mean_p:.3f} pipeline_min_ms={min_p:.3f}")

    budget_ms = 1000.0 / 30.0
    if args.impl == "native":
        verdict_d = "PASS" if p95_d < budget_ms else "FAIL"
        verdict_p = "PASS" if p95_p < budget_ms else "FAIL"
        print(f"[{verdict_d}] disparity単体: 30fps バジェット {budget_ms:.2f}ms に対し p95={p95_d:.2f}ms")
        print(f"[{verdict_p}] process_frame全体: 30fps バジェット {budget_ms:.2f}ms に対し p95={p95_p:.2f}ms")
        # project_knowledge/modules/module-core.md の受入基準4（30fps）の機械的ゲートとして使えるよう、判定を終了コードへ接続する
        # （stdout の FAIL を人間が読む前提だと自動化から見て常に成功になる）。
        if verdict_d == "FAIL" or verdict_p == "FAIL":
            return 1
    # --impl python は Python 参照実装の速度比較が目的で 30fps 判定の対象外（常に 0）。
    return 0


if __name__ == "__main__":
    sys.exit(main())
