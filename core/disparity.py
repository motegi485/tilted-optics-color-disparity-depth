"""core/disparity.py — calcDisparityByZNCC（Phase D: Rust実装への薄いラッパー）。

数値計算は rust/disparity/（PyO3拡張 `_rt_disparity_native`）に委譲する。
入力の 2D 形状・dtype(uint8)・非空・kernel_size/search_range_x の正の整数・奇数性は
ここで検証し、既存呼び出し元との ValueError 契約（§4.1 リスク#10）を維持する
（PyO3側の型抽出失敗は ValueError ではなく TypeError/OverflowError になるため、
検証はネイティブ拡張呼び出しの前に済ませる。red/blue の形状不一致のみネイティブ側の
防御チェックが ValueError を出す）。

Python参照実装は core/_disparity_reference.py に凍結保存されている
（§8.3 の Rust↔Python 照合・checks/bench_disparity.py の速度比較・checks/verify_disparity_rust.py
の回帰ゲートで使用。変更しない）。
"""
import numpy as np

try:
    import _rt_disparity_native as _native
except ImportError as e:
    raise ImportError(
        "Rustネイティブ拡張 _rt_disparity_native が見つかりません（Phase D）。"
        "rust/disparity/ で以下を実行してビルド・導入してください:\n"
        "  uv tool install maturin\n"
        "  maturin build --release\n"
        "  uv pip install (Get-ChildItem target/wheels/*.whl).FullName\n"
        "（`uv sync` を実行すると本拡張は再インストールが必要になります。CLAUDE.md 参照）"
    ) from e


def calcDisparityByZNCC(red_img, blue_img, kernel_size, search_range_x, th_value):
    """
    ZNCC（正規化相互相関）を用いて R/B チャンネル間の符号付き視差マップを計算する。

    R チャンネル（基準）を固定し、B チャンネル（テンプレート）を
    水平方向に ±s_range ピクセルずらしながら ZNCC で比較。
    最も類似するずれ量（サブピクセル補間つき）を各画素の視差 d として返す。

    Args:
        red_img (np.ndarray): R チャンネル画像（1ch uint8, shape=(H,W)）
        blue_img (np.ndarray): B チャンネル画像（1ch uint8, shape=(H,W)）
        kernel_size (int): ZNCC パッチサイズ（正の奇数）
        search_range_x (int): 水平探索幅（正の奇数）。視差探索範囲 = ±(search_range_x-1)//2
        th_value (int): エッジ強度の下限閾値

    Returns:
        np.ndarray: 符号付き視差マップ（float32, shape=(H,W)）。
                    d=0 は POF 上、d>0/d<0 は前後ぼけ、np.nan は無効画素。
    """
    if len(red_img.shape) != 2 or len(blue_img.shape) != 2:
        raise ValueError("入力は1チャンネルの画像を入力してください")
    if red_img.dtype != np.uint8 or blue_img.dtype != np.uint8:
        raise ValueError(
            f"入力は uint8 画像である必要があります"
            f"（渡された dtype: red={red_img.dtype}, blue={blue_img.dtype}）")
    if red_img.size == 0 or blue_img.size == 0:
        raise ValueError(
            f"入力画像が空です（red={red_img.shape}, blue={blue_img.shape}）")
    # 負の奇数は Python の剰余（-3 % 2 == 1）で奇数チェックを通過し、PyO3 の usize 抽出で
    # OverflowError になるため、整数型・正値をここで確定させる
    for name, val in (("kernel_size", kernel_size), ("search_range_x", search_range_x)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise ValueError(f"{name} は正の奇数（int）である必要があります（渡された値: {val!r}）")
        if val < 1:
            raise ValueError(f"{name} は正の奇数である必要があります（渡された値: {val}）")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size は奇数である必要があります（渡された値: {kernel_size}）")
    if search_range_x % 2 == 0:
        raise ValueError(f"search_range_x は奇数である必要があります（渡された値: {search_range_x}）")

    return _native.calc_disparity_by_zncc(
        red_img, blue_img, kernel_size, search_range_x, float(th_value), True)
