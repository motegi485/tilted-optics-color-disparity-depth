"""realtime/pipeline.py — 毎フレーム処理（§4.3 オンライン・§5.5）。

evaluate.py の per-frame core パスと同一手順で depth マップを返す（R/B 分割 →
ZNCC → gate1(|d|>DISP_MAX) → depth）ことが、ふるまい一致（§9.2）の前提。
config は読むが注入、pipeline は core ではない（config import 可）が計算の中核は
core 純関数を呼ぶだけに留める。gate2(Z レンジ 0<z<50000) は適用しない――
runner の表示側で vmin/vmax として適用する（depth マップを書き換えると evaluate.py の
ものとふるまい一致しなくなるため）。入力検証は raise（assert は -O で無効化・リスク#10）。
"""
import cv2
import numpy as np

import config
from core.disparity import calcDisparityByZNCC
from core.depth import compute_depth


def process_frame(proc_bgr, params):
    """proc_bgr: preprocess 済み (352,528,3) uint8 BGR -> depth_map(float32, NaN 伝播)。"""
    if proc_bgr.ndim != 3 or proc_bgr.shape[2] != 3:
        raise ValueError(f"process_frame: 3ch BGR を想定。実際は shape={proc_bgr.shape}")

    # 1) R/B チャンネル分割（シアン（G+B を透過）, イエロー（G+R を透過））
    blue, _, red = cv2.split(proc_bgr)

    # 2) ZNCC で視差マップを算出
    disp = calcDisparityByZNCC(red, blue, config.KERNEL_SIZE, config.SEARCH_RANGE_X, config.EDGE_TH)

    # 3) gate1: 視差絶対値が大きすぎる画素を無効化（NaN）
    disp[np.abs(disp) > config.DISP_MAX] = np.nan

    # 4) 経験式（参照式）で視差→深度に変換。NaN はそのまま伝播する
    depth = compute_depth(
        disp,
        params.v_pos, params.u_pos, params.a_pos, params.w_pos,
        params.v_neg, params.u_neg, params.a_neg, params.w_neg,
        config.EPS_DEN)

    # gate2（Z レンジ 0<z<50000）はここでは適用しない（runner 表示側の責務）
    return depth
