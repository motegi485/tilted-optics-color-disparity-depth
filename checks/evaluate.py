"""静止画回帰アンカーの照合（オーケストレーション層）。
core/ の純関数を呼び、既存の参照実装の depth.py と同じ集計で比較する。
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


# project_knowledge/implementation/verification-and-tools.md の回帰アンカー（DEPTH_MAX=50000・params/baseline/ 前提）。
# 2026-09-18 に設定変更（K17/S31/T30・gate1 のみ → K25/S41/T20・gate1＋推論帯 (0.45, 5)）に伴い取り直した値
# （実測 平均 2213.3627 / MAE 218.9271 / 有効 6339・記録票 2026-09-18 §8）。baseline 係数は旧設定の静止画校正値のままなので
# 距離の物理的な正しさではなく「計算経路が変わっていないこと」を検出するアンカーである。
# 旧設定のアンカー（2026-06-14 確定）: 平均 2313.9 / MAE 316.8 / 有効画素 7613（config を旧値に戻せば再現する）。
EXPECTED = {
    "2300.jpg": {"mean": 2213.4, "mae": 218.9, "valid": 6339},
}
TOL_MM = 0.05   # 平均・MAE の許容差（表示精度 0.1mm の半分）


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

    # 推論帯（2026-09-18 新設）: |d| <= INFER_BAND_LOW または |d| >= INFER_BAND_HIGH を無効化（pipeline と同一手順）
    abs_disp = np.abs(disp)
    disp[(abs_disp <= config.INFER_BAND_LOW) | (abs_disp >= config.INFER_BAND_HIGH)] = np.nan

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

    # 集計は float64 で行う（float32 累積誤差回避のため）
    mean_depth = float(np.mean(valid, dtype=np.float64))
    mae = float(np.mean(np.abs(valid.astype(np.float64) - gt_distance)))
    uniq, cnt = np.unique(valid, return_counts=True)
    mode_depth = float(uniq[np.argmax(cnt)])
    return mean_depth, mode_depth, mae, valid.size


def main() -> int:
    # Windows の日本語コンソール(cp932)は非対応文字の print で UnicodeEncodeError を出すため
    # UTF-8 に固定する（realtime/runner.py の main() 冒頭と同じ対処）。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    img_paths = sorted(config.TEST_DIR.glob("*.jpg", case_sensitive=False))
    if not img_paths:
        print(f"[ERROR] 検証対象画像が見つかりません: {config.TEST_DIR}/*.jpg")
        return 2

    params = load_params(config.BASELINE_PARAMS_PATH)
    expected_ci = {name.lower(): exp for name, exp in EXPECTED.items()}  # 拡張子の大小差を吸収

    all_pass = True
    matched = 0        # EXPECTED と照合できた枚数。0 のまま終わると「空振り成功」になる
    for img_path in img_paths:
        gt = int(img_path.stem)            # ファイル名=正解距離[mm]
        mean_d, mode_d, mae, valid_size = evaluate(img_path, params, gt)
        print(f"{img_path.name}: 平均={mean_d:7.1f}mm "
              f"最頻={mode_d:7.1f}mm MAE={mae:6.1f}mm (正解={gt}mm) 有効画素数={valid_size}")

        exp = expected_ci.get(img_path.name.lower())
        if exp is None:
            print("  [INFO] 期待値未登録（EXPECTED に無いファイル）のため照合はスキップ")
            continue
        matched += 1

        mean_ok = abs(mean_d - exp["mean"]) <= TOL_MM
        mae_ok = abs(mae - exp["mae"]) <= TOL_MM
        valid_ok = valid_size == exp["valid"]
        if mean_ok and mae_ok and valid_ok:
            print(f"  [PASS] 回帰アンカー一致（期待: 平均={exp['mean']}mm "
                  f"MAE={exp['mae']}mm 有効画素数={exp['valid']}）")
        else:
            all_pass = False
            print(f"  [FAIL] 回帰アンカー不一致: 期待 平均={exp['mean']}mm "
                  f"(diff={mean_d - exp['mean']:+.4f}) MAE={exp['mae']}mm "
                  f"(diff={mae - exp['mae']:+.4f}) 有効画素数={exp['valid']}(実測={valid_size})")

    # 回帰ゲートとして exit 0 を返してよいのは「アンカーを実際に照合して合格した」場合だけ。
    # 数値名だが未登録の JPG しか無い状態を成功にすると、アンカー画像の消失を検知できない。
    if matched == 0:
        print(f"[ERROR] 回帰アンカーの照合が1件も行われませんでした"
              f"（EXPECTED = {sorted(EXPECTED)}）。{config.TEST_DIR} に登録済みの"
              f"アンカー画像を置いてから再実行してください")
        return 2

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())