"""realtime/pipeline.py — 毎フレーム処理（docs/architecture.md のオンラインデータフロー・docs/module-realtime.md）。

evaluate.py の per-frame core パスと同一手順で depth マップを返す（R/B 分割 →
ZNCC → gate1(|d|>DISP_MAX) → 推論帯(|d|<=INFER_BAND_LOW または |d|>=INFER_BAND_HIGH を NaN・2026-09-18 新設)
→ depth）ことが、ふるまい一致（docs/requirements-and-status.md）の前提。
config は読むが注入、pipeline は core ではない（config import 可）が計算の中核は
core 純関数を呼ぶだけに留める。gate2(Z レンジ 0<z<50000) は適用しない――
runner の表示側で vmin/vmax として適用する（depth マップを書き換えると evaluate.py の
ものとふるまい一致しなくなるため）。入力検証は raise（assert は -O で無効化・docs/known-issues.md #10）。
"""
import cv2
import numpy as np

import config
from core.disparity import calcDisparityByZNCC
from core.depth import compute_depth


def process_frame(proc_bgr, params):
    """proc_bgr: preprocess 済み (352,528,3) uint8 BGR -> depth_map(float32, NaN 伝播)。

    形状と dtype をここで強制する（2026-09 レビュー）。参照式の y は行番号そのものであり、
    校正係数 v/u/ā は処理解像度 config.IMAGE_SIZE=(528,352) の y スケールに紐づく。別解像度の
    配列を渡すと同じ物理位置の y が変わるのに同じ係数が適用され、無音で誤った距離を返す。
    """
    if not isinstance(proc_bgr, np.ndarray):
        raise ValueError(f"process_frame: np.ndarray を想定。実際は {type(proc_bgr)!r}")
    expected_shape = (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0], 3)   # (H,W,C)=(352,528,3)
    if proc_bgr.shape != expected_shape:
        raise ValueError(
            f"process_frame: 入力は {expected_shape} (H,W,C) を想定。"
            f"実際は shape={proc_bgr.shape}（校正係数は処理解像度 {config.IMAGE_SIZE} 前提）")
    if proc_bgr.dtype != np.uint8:
        raise ValueError(
            f"process_frame: 入力は uint8 を想定。実際は dtype={proc_bgr.dtype}")

    # 1) R/B チャンネル分割（シアン（G+B を透過）, イエロー（G+R を透過））
    blue, _, red = cv2.split(proc_bgr)

    # 2) ZNCC で視差マップを算出
    disp = calcDisparityByZNCC(red, blue, config.KERNEL_SIZE, config.SEARCH_RANGE_X, config.EDGE_TH)

    # 3) gate1: 視差絶対値が探索半幅を超える画素を無効化（NaN）。S=41 では到達上限 19.5 < 20 で不発（安全網として維持）
    disp[np.abs(disp) > config.DISP_MAX] = np.nan

    # 3') 推論帯: 合焦面近傍の不感帯 |d| <= INFER_BAND_LOW と、大ぼけ域 |d| >= INFER_BAND_HIGH を無効化（NaN）。
    #     NaN との比較は False なので gate1 で落ちた画素はそのまま。定義は tools/analysis_common.apply_gates と同一
    abs_disp = np.abs(disp)
    disp[(abs_disp <= config.INFER_BAND_LOW) | (abs_disp >= config.INFER_BAND_HIGH)] = np.nan

    # 4) 経験式（参照式）で視差→深度に変換。NaN はそのまま伝播する
    depth = compute_depth(
        disp,
        params.v_pos, params.u_pos, params.a_pos, params.w_pos,
        params.v_neg, params.u_neg, params.a_neg, params.w_neg,
        config.EPS_DEN)

    # gate2（Z レンジ 0<z<50000）はここでは適用しない（runner 表示側の責務）
    return depth
