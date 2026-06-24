"""verify_clip_replay.py — §9.2 クリップ再生・再現一致の検証(root・camera.py 非依存)。

ライブ実行で保存した outputs/{session}/{i}.npy(depth・唯一の真実)と、同じ
captures/{session}/{i}.png を preprocess -> process_frame で再計算した depth が、
全フレームで NaN 位置＋有限値ともビット一致するかを照合する。

preprocess も process_frame も決定的(乱数なし)で、PNG は無劣化なので、再計算 depth は
ライブ depth とビット一致するはず。一致＝録画→再生の経路が無劣化で、ライブ結果が保存
クリップから完全再現可能(Phase D の Python<->Rust 照合 §8.3 の土台)であることの回帰確認。
照合ロジックは verify_pipeline_equivalence.py(task3)と同じ流儀(NaN 位置一致を確認して
から有限値をビット比較)。

params は録画時と同じものを使う必要がある(runner 既定＝baseline。--params で上書き可)。

実行: uv run python verify_clip_replay.py --session 20260624_153000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import config
from core.params import load_params
from realtime.camera import preprocess
from realtime.pipeline import process_frame


def _compare(live: np.ndarray, recomputed: np.ndarray) -> tuple[bool, str]:
    """live(保存 npy)と recomputed(png から再計算)を NaN 位置＋有限値で厳密比較する。"""
    if live.shape != recomputed.shape:
        return False, f"shape 不一致: live={live.shape}, recomputed={recomputed.shape}"
    nan_live = np.isnan(live)
    nan_re = np.isnan(recomputed)
    if not np.array_equal(nan_live, nan_re):
        n = int(np.count_nonzero(nan_live != nan_re))
        return False, f"NaN 位置不一致: {n} 画素"
    # NaN 位置が一致している前提で、有限値(同じマスク)をビット比較する
    if not np.array_equal(live[~nan_live], recomputed[~nan_live]):
        diff = np.abs(live[~nan_live] - recomputed[~nan_live])
        return False, (f"有限値ビット不一致: max|Δ|={float(diff.max()):.6g}, "
                       f"不一致画素={int(np.count_nonzero(diff > 0))}")
    return True, "exact match"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="§9.2 クリップ再生・再現一致の検証")
    parser.add_argument("--session", required=True,
                        help="セッション名(captures/{session}, outputs/{session} を照合)")
    parser.add_argument("--params", default=str(config.BASELINE_PARAMS_PATH),
                        help="fit_result.csv のパス(録画時と同じものを使う。既定: baseline)")
    args = parser.parse_args(argv)

    cap_dir = Path(config.CAPTURES_DIR) / args.session
    out_dir = Path(config.OUTPUTS_DIR) / args.session
    if not cap_dir.is_dir():
        print(f"[ERROR] captures セッションが見つかりません: {cap_dir}")
        return 2
    if not out_dir.is_dir():
        print(f"[ERROR] outputs セッションが見つかりません: {out_dir}")
        return 2

    pngs = sorted(p for p in cap_dir.glob("??????.png") if p.stem.isdigit())
    if not pngs:
        print(f"[ERROR] {cap_dir} に連番 png(??????.png)がありません")
        return 2

    params = load_params(args.params)
    print(f"session={args.session}  frames={len(pngs)}  params={args.params}")

    n_ok = 0
    first_fail: str | None = None
    for k, p in enumerate(pngs):
        idx = p.stem
        npy = out_dir / f"{idx}.npy"
        if not npy.exists():
            print(f"[FAIL] {idx}: 対応する depth npy がありません ({npy})")
            first_fail = first_fail or idx
            continue
        raw = cv2.imread(str(p))                       # 無劣化 PNG -> raw(1280x720 BGR) を完全復元
        if raw is None:
            print(f"[FAIL] {idx}: png を読めません")
            first_fail = first_fail or idx
            continue
        recomputed = process_frame(preprocess(raw), params)  # ライブと同一経路で再計算
        live = np.load(str(npy))
        if k == 0:
            print(f"  先頭フレーム: dtype(live)={live.dtype} "
                  f"dtype(recomputed)={recomputed.dtype} shape={live.shape}")
        ok, msg = _compare(live, recomputed)
        if ok:
            n_ok += 1
        else:
            print(f"[FAIL] {idx}: {msg}")
            first_fail = first_fail or idx

    total = len(pngs)
    print(f"\n結果: {n_ok}/{total} フレーム一致")
    if n_ok == total:
        print("[PASS] 全フレーム ビット一致(NaN 位置＋有限値)= §9.2 クリップ再生・再現一致 達成")
        return 0
    print(f"[FAIL] 不一致あり(最初の不一致フレーム: {first_fail})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
