import numpy as np


def compute_depth(disp_map, v_pos, u_pos, a_pos, w_pos, v_neg, u_neg, a_neg, w_neg, eps_den):
    """
    視差マップから距離マップを計算する（純関数）。

    参照式 z = v / (y - a·d - u) + w を各画素に適用する。
    I/O・表示・config への依存を持たない。

    視差の符号で 2 分岐を使い分ける（d > 0 → *_pos, d < 0 → *_neg）。
    符号と前ぼけ/後ろぼけの対応は校正の符号規約に依存するため、
    本関数では前後ぼけを断定せず符号でのみ区別する。

    Parameters
    ----------
    disp_map : np.ndarray, shape (H, W), dtype float32
        視差マップ。マッチング失敗画素は NaN。
        前提: 呼び出し側で |d| > disp_max の画素を事前に NaN クリップ済みであること。
    v_pos : float
        参照式の v パラメータ。d > 0 側。
    u_pos : float
        参照式の定数項 u。d > 0 側。
    a_pos : float
        各距離の直線傾き a の平均値 ā。d > 0 側。
    w_pos : float
        参照式の定数項 w。d > 0 側。
        理論的根拠により 0 固定（project_knowledge/architecture/theory.md の「経験式（参照式）」の節）。
        論文式との対応・将来拡張のため引数として保持。
    v_neg : float
        参照式の v パラメータ。d < 0 側。
    u_neg : float
        参照式の定数項 u。d < 0 側。
    a_neg : float
        各距離の直線傾き a の平均値 ā。d < 0 側。
    w_neg : float
        参照式の定数項 w。d < 0 側。
        理論的根拠により 0 固定（project_knowledge/architecture/theory.md の「経験式（参照式）」の節）。
        論文式との対応・将来拡張のため引数として保持。
    eps_den : float
        分母ゼロガードの閾値 [px]。
        |y - a·d - u| < eps_den の画素は発散危険として NaN にする。

    Returns
    -------
    depth_map : np.ndarray, shape (H, W), dtype float32
        距離マップ [mm]。下記の画素は NaN:
        - d が NaN  （マッチング失敗 / 事前クリップ済み）
        - d == 0    （POF 上・不感帯）
        - |分母| < eps_den  （発散危険）
    """
    # y 座標を (H, 1) で作る。(H, W) の disp_map とブロードキャストさせることで、
    # 参照式の y を全画素へループなしに与えられる。
    H, _ = disp_map.shape
    y_grid = np.arange(H, dtype=np.float64)[:, None]

    # NaN/0 除算は意図的（無効画素を NaN として伝播させる）なので警告を抑制する。
    with np.errstate(invalid='ignore', divide='ignore'):
        # d > 0 側: 分母を全画素で計算 → 発散危険を NaN マスク → z = v/den + w
        den_pos = y_grid - a_pos * disp_map - u_pos
        den_pos = np.where(np.abs(den_pos) < eps_den, np.nan, den_pos)
        z_pos = v_pos / den_pos + w_pos

        # d < 0 側: 前後ぼけで校正が独立なため、neg パラメータで同じ手順を別計算する。
        den_neg = y_grid - a_neg * disp_map - u_neg
        den_neg = np.where(np.abs(den_neg) < eps_den, np.nan, den_neg)
        z_neg = v_neg / den_neg + w_neg

    # 全画素 NaN で初期化し、視差の符号で z_pos / z_neg を割り当てる。
    # NaN 比較は False を返すため、NaN・d==0 の画素は書き込まれず NaN のまま残る。
    depth_map = np.full(disp_map.shape, np.nan, dtype=np.float32)
    depth_map[disp_map > 0] = z_pos[disp_map > 0].astype(np.float32)
    depth_map[disp_map < 0] = z_neg[disp_map < 0].astype(np.float32)
    
    return depth_map