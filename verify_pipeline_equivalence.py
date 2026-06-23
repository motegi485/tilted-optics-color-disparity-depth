"""verify_pipeline_equivalence.py
realtime.pipeline.process_frame と evaluate.py の per-frame core パス（集計前の depth マップ）が
固定入力でビット一致することを検証する（§9.2 Phase C Done 基準の「経路の同値性」）。

Path A: realtime.pipeline.process_frame（実モジュール）
Path B: evaluate.py の core 手順を独立に再現（split → disparity → gate1 → compute_depth）
両者の depth マップが NaN 位置・有限値とも厳密一致することを確認する。

gate2(0<z<50000) は両経路とも適用しない（runner 表示側の責務）。深度精度ではなく
経路の同値性の確認なので、FR-8 webcam 校正前でも実施可。params は baseline 確定値を使う。

実行: uv run python verify_pipeline_equivalence.py
"""
import sys

import cv2
import numpy as np

import config
from core.disparity import calcDisparityByZNCC
from core.depth import compute_depth
from core.params import load_params
from realtime.pipeline import process_frame


def load_fixture():
    """固定 (352,528,3) uint8 BGR フレームを1枚返す。
    優先: data/test/2300.jpg を cv2.resize(IMAGE_SIZE)。無ければ data/test の任意 .jpg。
    """
    test_dir = config.TEST_DIR
    cand = test_dir / "2300.jpg"
    if not cand.exists():
        jpgs = sorted(test_dir.glob("*.jpg"))
        if not jpgs:
            raise FileNotFoundError(
                f"固定入力が見つかりません: {test_dir}/*.jpg を1枚用意してください"
            )
        cand = jpgs[0]
    img = cv2.imread(str(cand))
    if img is None:
        raise RuntimeError(f"画像を読めません: {cand}")
    frame = cv2.resize(img, config.IMAGE_SIZE)   # (W,H)=(528,352) -> array shape (352,528,3)
    return cand.name, frame


def evaluate_core_depthmap(frame, params):
    """Path B: evaluate.py の core 手順（split → disparity → gate1 → compute_depth）を
    pipeline.py を経由せず独立に再現し、集計前の depth マップを返す（gate2 は適用しない）。
    """
    blue, _, red = cv2.split(frame)
    disp = calcDisparityByZNCC(red, blue,
                               config.KERNEL_SIZE, config.SEARCH_RANGE_X, config.EDGE_TH)
    disp[np.abs(disp) > config.DISP_MAX] = np.nan
    return compute_depth(
        disp,
        params.v_pos, params.u_pos, params.a_pos, params.w_pos,
        params.v_neg, params.u_neg, params.a_neg, params.w_neg,
        config.EPS_DEN,
    )


def main():
    params = load_params(config.BASELINE_PARAMS_PATH)
    name, frame = load_fixture()
    print(f"固定入力: {name}  shape={frame.shape}  dtype={frame.dtype}")

    # 同じ固定フレーム・同じ params を Path A / Path B の両方に通し、depth マップを比較する
    depth_A = process_frame(frame, params)              # Path A: 実 pipeline
    depth_B = evaluate_core_depthmap(frame, params)     # Path B: evaluate.py core の独立再現

    # 比較1: NaN（無効画素）の位置が完全一致するか
    nan_A, nan_B = np.isnan(depth_A), np.isnan(depth_B)
    nan_match = np.array_equal(nan_A, nan_B)
    # 比較2: 有効画素の深度値が完全一致するか（NaN 位置が一致している前提）
    finite_match = np.array_equal(depth_A[~nan_A], depth_B[~nan_B])

    n_finite = int(np.sum(~nan_A))
    valid = depth_A[(depth_A > config.DEPTH_MIN) & (depth_A < config.DEPTH_MAX)]
    print(f"dtype(A)={depth_A.dtype}  有効画素={n_finite}  gate2後有効={valid.size}")
    print(f"NaN 位置一致: {nan_match}  /  有限値ビット一致: {finite_match}")

    if nan_match and finite_match:
        print("PASS: process_frame は evaluate.py core パスとビット一致")
        return 0

    # 不一致時は原因箇所の見当をつけやすいよう件数を出す
    print("FAIL: 不一致あり。呼び出し順・引数・ゲート位置を確認すること")
    if not nan_match:
        print(f"  NaN 位置不一致画素数: {int(np.sum(nan_A != nan_B))}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
