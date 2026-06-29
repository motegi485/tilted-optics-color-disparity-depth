"""Phase A 回帰チェック（オーケストレーション層・暫定）。
core/ の純関数を呼び、先行研究の参考コードの depth.py と同じ集計で比較する。
ゲーティングは config 値でこの層が適用する（core は純関数のまま）。
"""
# --- checks/ から root の config / core / realtime を import するためのブートストラップ ---
# 推奨実行は `uv run python -m checks.<name>`（root を cwd に）。
# 直接実行 `python checks/<name>.py` でも動くよう root を sys.path に通す（calibration/ と同方式）。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config
from core.disparity import calcDisparityByZNCC
from core.depth import compute_depth
from core.params import load_params


def evaluate(image_path: Path, params, gt_distance: int):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"画像読み込み失敗: {image_path}")

    image = cv2.resize(image, config.IMAGE_SIZE)   # (W,H)=(528,352)
    blue, _, red = cv2.split(image)                # BGR順: G は破棄

    disp = calcDisparityByZNCC(red, blue,
                               config.KERNEL_SIZE,
                               config.SEARCH_RANGE_X,
                               config.EDGE_TH)

    # ゲート1: 探索半幅超の視差を無効化（core の外で適用）
    disp[np.abs(disp) > config.DISP_MAX] = np.nan

    depth = compute_depth(
        disp,
        params.v_pos, params.u_pos, params.a_pos, params.w_pos,
        params.v_neg, params.u_neg, params.a_neg, params.w_neg,
        config.EPS_DEN,
    )

    # ゲート2: Z レンジ外を除外（NaN は比較で False となり自然に除外）
    valid = depth[(depth > config.DEPTH_MIN) & (depth < config.DEPTH_MAX)]
    if valid.size == 0:
        raise RuntimeError(f"{image_path.name}: 有効推定距離が0件")

    # float32 の累積誤差を避けるため集計は float64 で行う（§下記の注意点）
    mean_depth = float(np.mean(valid, dtype=np.float64))
    mae = float(np.mean(np.abs(valid.astype(np.float64) - gt_distance)))
    uniq, cnt = np.unique(valid, return_counts=True)
    mode_depth = float(uniq[np.argmax(cnt)])
    return mean_depth, mode_depth, mae, valid.size


def main():
    params = load_params(config.BASELINE_PARAMS_PATH)
    for img_path in sorted(config.TEST_DIR.glob("*.jpg")):
        gt = int(img_path.stem)            # ファイル名=正解距離[mm]
        mean_d, mode_d, mae, valid_size = evaluate(img_path, params, gt)
        print(f"{img_path.name}: 平均={mean_d:7.1f}mm "
              f"最頻={mode_d:7.1f}mm MAE={mae:6.1f}mm (正解={gt}mm) 有効画素数={valid_size}")


if __name__ == "__main__":
    main()