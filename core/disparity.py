import math
import cv2
import numpy as np


def calcDisparityByZNCC(red_img, blue_img, kernel_size, search_range_x, th_value):
    """
    ZNCC（正規化相互相関）を用いて R/B チャンネル間の符号付き視差マップを計算する。

    R チャンネル（基準）を固定し、B チャンネル（テンプレート）を
    水平方向に ±s_range ピクセルずらしながら ZNCC で比較。
    最も類似するずれ量（サブピクセル補間つき）を各画素の視差 d として返す。

    Args:
        red_img (np.ndarray): R チャンネル画像（1ch uint8, shape=(H,W)）
        blue_img (np.ndarray): B チャンネル画像（1ch uint8, shape=(H,W)）
        kernel_size (int): ZNCC パッチサイズ（奇数）
        search_range_x (int): 水平探索幅（奇数）。視差探索範囲 = ±(search_range_x-1)//2
        th_value (int): エッジ強度の下限閾値

    Returns:
        np.ndarray: 符号付き視差マップ（float32, shape=(H,W)）。
                    d=0 は POF 上、d>0/d<0 は前後ぼけ、np.nan は無効画素。
    """
    if len(red_img.shape) != 2 or len(blue_img.shape) != 2:
        raise ValueError("入力は1チャンネルの画像を入力してください")

    # 半窓計算 (k-1)//2 が整数の中心を持つには奇数が必要
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size は奇数である必要があります（渡された値: {kernel_size}）")
    if search_range_x % 2 == 0:
        raise ValueError(f"search_range_x は奇数である必要があります（渡された値: {search_range_x}）")

    k_size = (kernel_size - 1) // 2
    s_range = (search_range_x - 1) // 2

    # np.zeros でなく np.nan: d=0 は「POF 上の有効視差」を意味するため 0 では無効画素と区別できない
    disp_map = np.full(red_img.shape, np.nan, dtype=np.float32)

    # Sobel(dx=1, dy=0) で水平輝度勾配を算出 → 縦エッジを強調
    # CFA の水平視差は縦エッジ（∂I/∂x が大きい場所）でのみ計測できる
    edge_red_img = cv2.Sobel(red_img, cv2.CV_32F, 1, 0)
    edge_blue_img = cv2.Sobel(blue_img, cv2.CV_32F, 1, 0)
    edge_red_img = cv2.convertScaleAbs(edge_red_img)   # 絶対値 uint8 へ（エッジ方向は不問）
    edge_blue_img = cv2.convertScaleAbs(edge_blue_img)

    # R 側の勾配で処理対象画素を選別（弱エッジ画素は 0 になりスキップされる）
    _, edge = cv2.threshold(edge_red_img, th_value, 255, cv2.THRESH_TOZERO)

    for y in range(k_size, red_img.shape[0] - k_size):
        # x の除外幅が y より広いのは、B パッチの探索範囲（s_range 分）の余地が必要なため
        for x in range(s_range + k_size + 1, red_img.shape[1] - s_range - k_size - 1):
            if edge[y, x] == 0:
                continue  # エッジが弱い → disp_map[y,x] は NaN のまま

            # (y,x) を中心とした R 勾配パッチ（スイープ中は固定）
            roi_red = edge_red_img[y - k_size:y + k_size + 1, x - k_size:x + k_size + 1]

            max_sim = -math.inf
            max_x = -1  # -1 = 有効ピーク未発見のセンチネル
            sim_array = np.zeros(search_range_x, dtype=np.float32)

            # B パッチを右端（shift_x=0）から左端（shift_x=search_range_x-1）へ 1px ずつ移動
            for shift_x in range(search_range_x):
                shifted_x = x + s_range - k_size - shift_x  # B パッチの左端座標
                roi_blue = edge_blue_img[y - k_size:y + k_size + 1,
                                         shifted_x:shifted_x + kernel_size]

                # 同サイズパッチ同士の matchTemplate → 1×1 の出力 = このシフトの ZNCC 値 1 個
                match_result = cv2.matchTemplate(roi_red, roi_blue, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(match_result)

                sim_array[shift_x] = max_val

                # 端（shift_x=0 と search_range_x-1）は除外:
                # サブピクセル補間に隣接要素（sim_array[max_x±1]）が必要なため
                if max_val > max_sim and shift_x != 0 and shift_x != search_range_x - 1:
                    max_sim = max_val
                    max_x = shift_x  # インデックス（画素位置 shifted_x ではない）

            # max_x < 1: 有効ピーク未発見（センチネル -1 のまま）
            # max_x > search_range_x-2: sim_array[max_x+1] が配列外になる
            if max_x < 1 or max_x > search_range_x - 2:
                continue  # NaN のまま

            # サブピクセル補間: ピーク前後 3 点に放物線を当て頂点の小数部を算出
            sim_neg = sim_array[max_x - 1]
            sim_peak = sim_array[max_x]
            sim_pos = sim_array[max_x + 1]

            denominator = 4 * sim_peak - 2 * sim_neg - 2 * sim_pos
            # 放物線が上に凸（denominator > 0）かつ補正量 ≤ 0.5px のときのみ適用
            if denominator > 0 and abs(sim_pos - sim_neg) <= denominator:
                subpixel = (sim_pos - sim_neg) / denominator
            else:
                subpixel = 0.0  # 条件を満たさない場合は整数視差をそのまま採用

            # シフトインデックス → 符号付き視差（max_x = s_range のとき d=0 = POF 上）
            submin = max_x + subpixel
            abs_submin = submin - s_range
            disp_map[y, x] = abs_submin

    return disp_map