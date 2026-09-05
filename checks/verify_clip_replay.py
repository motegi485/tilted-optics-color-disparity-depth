"""verify_clip_replay.py — docs/requirements-and-status.md のクリップ再生・再現一致の検証(root・camera.py 非依存)。

ライブ実行で保存した outputs/{session}/{i}.npy(depth・唯一の真実)と、同じ
captures/{session}/{i}.png を preprocess -> process_frame で再計算した depth が、
全フレームで NaN 位置＋有限値ともビット一致するかを照合する。

preprocess も process_frame も決定的(乱数なし)で、PNG は無劣化なので、再計算 depth は
ライブ depth とビット一致するはず。一致＝録画→再生の経路が無劣化で、ライブ結果が保存
クリップから完全再現可能(高速化段階の Python<->Rust 照合〈docs/verification-and-tools.md〉の土台)であることの回帰確認。
照合ロジックは verify_pipeline_equivalence.py(task3)と同じ流儀(NaN 位置一致を確認して
から有限値をビット比較)。

params は録画時と同じものを使う必要がある。--params 未指定時は outputs/{session}/meta.json
（B-6・録画開始時に runner.py が書く）があればその params_path を自動採用し、無ければ
live＝root fit_result.csv を既定とする。--params を明示指定すれば常にそちらが優先される。
FR-8 以前に録画したセッション(例: clip92)は meta.json が無く baseline params で録画されて
いるため、再検証時は --params params/baseline/fit_result.csv を明示すること。

実行: uv run python -m checks.verify_clip_replay --session 20260624_153000
"""
from __future__ import annotations

# --- checks/ から root の config / core / realtime を import するためのブートストラップ ---
# 推奨実行は `uv run python -m checks.<name>`（root を cwd に）。
# 直接実行 `python checks/<name>.py` でも動くよう root を sys.path に通す（calibration/ と同方式）。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

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
    # np.array_equal は dtype 差を無視して数値等値なら True を返す。canonical な depth は
    # float32（NaN 保持）と定められているので、dtype 一致自体を合否条件に含める
    # （float64 で保存された npy を「ビット一致」と呼ばないため・2026-09 レビュー）。
    if live.dtype != np.float32 or recomputed.dtype != np.float32:
        return False, (f"dtype 不一致: live={live.dtype}, recomputed={recomputed.dtype}"
                       f"（canonical は float32）")
    nan_live = np.isnan(live)
    nan_re = np.isnan(recomputed)
    if not np.array_equal(nan_live, nan_re):
        n = int(np.count_nonzero(nan_live != nan_re))
        return False, f"NaN 位置不一致: {n} 画素"
    if np.count_nonzero(~nan_live) == 0:
        # 両者とも全 NaN だと空集合同士の比較で match 判定が真になり得るため無条件 FAIL とする
        # (Phase D の Rust 照合の土台としての穴を塞ぐ・2026-07 レビュー)。
        return False, "有効画素 0 — 入力・params・ゲートを確認"
    # NaN 位置が一致している前提で、有限値(同じマスク)をビット比較する
    if not np.array_equal(live[~nan_live], recomputed[~nan_live]):
        diff = np.abs(live[~nan_live] - recomputed[~nan_live])
        return False, (f"有限値ビット不一致: max|Δ|={float(diff.max()):.6g}, "
                       f"不一致画素={int(np.count_nonzero(diff > 0))}")
    return True, "exact match"


def main(argv: list[str] | None = None) -> int:
    # Windows の日本語コンソール(cp932)は非対応文字の print で UnicodeEncodeError を出すため
    # UTF-8 に固定する（realtime/runner.py の main() 冒頭と同じ対処）。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="docs/requirements-and-status.md のクリップ再生・再現一致の検証")
    parser.add_argument("--session", required=True,
                        help="セッション名(captures/{session}, outputs/{session} を照合)")
    parser.add_argument("--params", default=None,
                        help="fit_result.csv のパス(録画時と同じものを使う。既定: セッションの "
                             "meta.json があればそれを採用、無ければ live＝root。baseline を使う"
                             "場合は明示指定)")
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

    if args.params is not None:
        params_path = args.params
        print(f"params: 明示指定 {params_path}")
    else:
        meta_path = out_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            params_path = meta["params_path"]
            print(f"params: meta.json から採用 {params_path}（B-6）")
        else:
            params_path = str(config.PARAMS_PATH)
            print(f"params: meta.json 無し。既定(live) {params_path}")

    params = load_params(params_path)
    print(f"session={args.session}  frames={len(pngs)}  params={params_path}")

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
        print("[PASS] 全フレーム ビット一致(NaN 位置＋有限値)= docs/requirements-and-status.md のクリップ再生・再現一致 達成")
        return 0
    print(f"[FAIL] 不一致あり(最初の不一致フレーム: {first_fail})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
