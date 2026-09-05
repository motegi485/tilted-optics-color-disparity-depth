"""
calibration/calibrate.py

校正画像（ファイル名 = 距離[mm]。.png/.jpg/.jpeg。キャンペーン内での規格混在は下の入力ガードが拒否）から符号付き視差を求め、符号別に per-pixel RANSAC で
直線 row = a·disparity + b をフィットし、各距離の zero_disp(=切片 b)・傾き a・
インライヤ率・点数を params/params.csv に出力する。
（既存の参照実装の per-pixel RANSAC 版を本システムの構成で再現したもの）

パイプライン: 校正画像 → 視差(core/disparity) → 符号別の点群抽出 → RANSAC 直線フィット → params.csv
後段の fit_params.py が params.csv を受け取り、参照式 y0 = v/z + u をフィットする。

設計原則(docs/architecture.md):
  - calibration/ は core ではない。config import・I/O・matplotlib 可。
    ただし視差計算は純関数 core.disparity を config 注入で呼ぶ(校正と推論の設定を一致させる)。
  - 無効値は np.nan(0 埋め禁止)。入力検証は raise(assert は -O で無効化されるため使わない)。
"""

# --- サブパッケージから root の config / core を import するためのブートストラップ ---
# 推奨実行は `uv run python -m calibration.calibrate`(root を cwd に)。
# 直接実行 `python calibration/calibrate.py` でも動くよう root を sys.path に通す。
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

import config
from core.disparity import calcDisparityByZNCC
from sklearn.linear_model import RANSACRegressor


IMG_EXTS = {".jpg", ".jpeg", ".png"}   # 校正画像として扱う拡張子


def extract_sign_points(disp_map, sign, low, height):
    """指定符号の生視差画素を (disparity, row) の点群として返す(行平均はとらない)。

    pos 枝は 0<low<d<height、neg 枝は -height<d<-low の画素のみ採用する
    (POF 近傍の不感帯と探索端の誤マッチを除外)。
    NaN は比較が必ず False になるため、ここで自動的に除外される。
    """
    if sign > 0:
        mask = (disp_map > low) & (disp_map < height)
    else:
        mask = (disp_map < -low) & (disp_map > -height)
    rows, _cols = np.where(mask)               # x 座標(列)は直線モデルに不要なので捨てる
    return disp_map[mask], rows.astype(np.float64)


def ransac_fit_line(d_vals, row_vals, residual_threshold, random_state):
    """row = a·disparity + b を RANSAC でフィット(特徴量=disparity, 目的変数=row)。

    多数決ではなく幾何的コンセンサス(共線インライヤ最大化)で本物の直線を拾うため、
    散らばったスプリアス誤マッチが多くても頑健。
    戻り値: a(傾き), b(切片=zero_disp=視差0のy座標), inlier_mask。
    コンセンサス不成立時は RANSACRegressor が ValueError を送出する。
    """
    # 点数だけでは傾きの同定性を保証できない: d が全て同値だと設計行列 [d, 1] の階数が1に
    # なり、a は非同定のまま任意の値が返りうる（RANSAC_MIN_POINTS は階数を保証しない）。
    # 呼び出し側は ValueError を「コンセンサス不成立」と同じく NaN 行として扱う（2026-09 レビュー）。
    n_unique = int(np.unique(d_vals).size)
    if n_unique < 2:
        raise ValueError(
            f"視差の一意値が {n_unique} 個しかなく、直線の傾き a を同定できません"
            f"（設計行列 [d, 1] の階数が1）")

    X = d_vals.reshape(-1, 1)                  # sklearn は特徴量を (N,1) の2次元で要求する
    est = RANSACRegressor(residual_threshold=residual_threshold,
                          random_state=random_state)
    est.fit(X, row_vals)
    a = float(est.estimator_.coef_[0])         # 最終インライヤで再フィットした直線の傾き
    b = float(est.estimator_.intercept_)       # 切片 = zero_disp(fit_params.py の核心入力)
    return a, b, est.inlier_mask_


def _plot_branch(ax, d_vals, row_vals, mask, a, b, height_px, color, label):
    """1枝分の散布図と RANSAC 直線を描く(アウトライヤ=灰, インライヤ=色)。"""
    ax.scatter(d_vals[~mask], row_vals[~mask], s=1, c="lightgray", alpha=0.3)
    ax.scatter(d_vals[mask],  row_vals[mask],  s=1, c=color,       alpha=0.5)
    if np.isfinite(a) and np.isfinite(b) and a != 0:   # RANSAC 失敗(NaN)・ゼロ割は描画しない
        # row = a·disparity + b を disparity について解き、全行に渡って直線を引く
        rr = np.arange(height_px)
        ax.plot((rr - b) / a, rr, color=color, lw=2, label=label)


def main():
    parser = argparse.ArgumentParser(description="校正画像 → 視差 → 直線フィット → params.csv")
    parser.add_argument("--show-plot", action="store_true",
                        help="距離ごとに散布図＋RANSAC直線を表示する（既定: 非表示。CI/一気通貫実行を"
                             "ブロックしないため既定オフ）")
    args = parser.parse_args()

    # --- 入口チェック: 校正画像ディレクトリと画像の存在を先に確認する ---
    calib_dir = config.CALIB_IMGS_DIR
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"キャリブレーション画像ディレクトリがありません: {calib_dir}")

    image_paths = sorted(p for p in calib_dir.iterdir()
                         if p.suffix.lower() in IMG_EXTS)
    if not image_paths:
        raise FileNotFoundError(f"キャリブレーション画像が見つかりません: {calib_dir}")

    # --- 入力ガード(2026-07 レビュー): 距離名ファイルの規格混在・重複距離を事前検出する ---
    # webcam(.png)と静止画(.jpg/.jpeg)が同一キャンペーンに混在すると、1本の params.csv に
    # 異規格が無警告で統合されdocs/module-calibration.mdの混用禁止に違反するため、ループに入る前に止める。
    dist_paths: dict[int, list[Path]] = {}
    for p in image_paths:
        try:
            d = int(p.stem)
        except ValueError:
            continue  # 距離名でないファイルは後段のループで従来どおりスキップ表示される
        dist_paths.setdefault(d, []).append(p)

    dist_exts = {p.suffix.lower() for paths in dist_paths.values() for p in paths}
    if (".png" in dist_exts) and ({".jpg", ".jpeg"} & dist_exts):
        raise ValueError(
            "webcam(.png) と静止画(.jpg) の校正画像が混在しています。"
            "キャンペーン単位で単一規格にしてください"
        )

    duplicates = {d: paths for d, paths in dist_paths.items() if len(paths) > 1}
    if duplicates:
        detail = "; ".join(f"{d}mm={[p.name for p in paths]}" for d, paths in sorted(duplicates.items()))
        raise ValueError(f"同一距離の校正画像が重複しています（{detail}）。1距離1ファイルにしてください")

    low, height = config.DISP_PLOT_LOW, config.DISP_PLOT_HEIGHT
    rows_out = []                              # 距離ごとの結果を貯め、ループ後に一括で CSV 化する

    # --- 距離ループ: 画像1枚 = 1距離 = params.csv の1行 ---
    for path in image_paths:
        # ファイル名(拡張子除く)が距離値。整数化できないファイルは校正対象外として飛ばす
        try:
            depth_value = int(path.stem)
        except ValueError:
            print(f"スキップ: ファイル名が距離(整数)でない → {path.name}")
            continue

        image = cv2.imread(str(path))
        if image is None:                      # 壊れた画像など読み込み失敗時は None が返る
            print(f"スキップ: 画像読み込み失敗 → {path.name}")
            continue

        # 入力ガード(2026-07 レビュー): 処理解像度そのもの、または3:2アスペクト比(±1%)の
        # フル解像度静止画のみ許可する。raw(1280x720=16:9等)の誤投入は非等方リサイズで
        # 幾何が歪んだ params を無警告生成するため、ここで止める。
        h_raw, w_raw = image.shape[:2]
        if (w_raw, h_raw) != config.IMAGE_SIZE:
            expected_ratio = config.IMAGE_SIZE[0] / config.IMAGE_SIZE[1]   # 3:2
            actual_ratio = w_raw / h_raw
            if abs(actual_ratio - expected_ratio) / expected_ratio > 0.01:
                raise ValueError(
                    f"校正画像のサイズ・アスペクト比が不正です → {path.name}: "
                    f"実 shape=(W={w_raw},H={h_raw})（比={actual_ratio:.4f}）。"
                    f"期待は処理解像度 {config.IMAGE_SIZE} 一致、"
                    f"またはアスペクト比 3:2（={expected_ratio:.4f}）±1% のフル解像度静止画"
                )
        image = cv2.resize(image, config.IMAGE_SIZE)   # 推論と同一サイズにそろえる(docs/module-calibration.md)

        # G は分割型CF開口の両側を通過して色シフトが相殺されるため視差を持たない → 捨てる(docs/theory.md)
        blue, _, red = cv2.split(image)
        # 視差計算は純関数を config 注入で呼ぶ。これで校正・推論の設定一致がコード上で保証される
        disp_map = calcDisparityByZNCC(
            red, blue, config.KERNEL_SIZE, config.SEARCH_RANGE_X, config.EDGE_TH)

        # 符号別に生視差画素を点群として取り出す。n は画素レベルのサンプル数(数万オーダー)
        d_pos, y_pos = extract_sign_points(disp_map, +1, low, height)
        d_neg, y_neg = extract_sign_points(disp_map, -1, low, height)
        n_pos, n_neg = len(d_pos), len(d_neg)

        # 直線フィットに足る点数が無い枝が一方でもあれば、その距離は校正不能として飛ばす
        if n_pos < config.RANSAC_MIN_POINTS or n_neg < config.RANSAC_MIN_POINTS:
            print(f"スキップ: 生視差点が不足 (pos={n_pos}, neg={n_neg}) → {path.name}")
            continue

        # pos/neg は視差範囲も点群分布も異なるため独立にフィットする。
        # コンセンサス不成立時は NaN・全 False マスクで記録し、もう片枝は活かす
        try:
            a_pos, b_pos, mask_pos = ransac_fit_line(
                d_pos, y_pos, config.RANSAC_RESIDUAL_TH, config.RANSAC_RANDOM_STATE)
        except ValueError as e:
            print(f"  pos: 直線フィット不成立 → NaN（{e}）")
            a_pos, b_pos, mask_pos = np.nan, np.nan, np.zeros(n_pos, dtype=bool)
        try:
            a_neg, b_neg, mask_neg = ransac_fit_line(
                d_neg, y_neg, config.RANSAC_RESIDUAL_TH, config.RANSAC_RANDOM_STATE)
        except ValueError as e:
            print(f"  neg: 直線フィット不成立 → NaN（{e}）")
            a_neg, b_neg, mask_neg = np.nan, np.nan, np.zeros(n_neg, dtype=bool)

        # インライヤ率 = True の割合(bool 配列の平均)。校正品質の診断値(docs/module-calibration.md)
        inlier_pos = float(mask_pos.mean()) if n_pos else 0.0
        inlier_neg = float(mask_neg.mean()) if n_neg else 0.0

        print(f"====== {depth_value}mm ======\n"
              f"[pos] a={a_pos:.4f} zero_disp={b_pos:.4f} inlier={inlier_pos*100:.1f}% "
              f"({int(mask_pos.sum())}/{n_pos})\n"
              f"[neg] a={a_neg:.4f} zero_disp={b_neg:.4f} inlier={inlier_neg*100:.1f}% "
              f"({int(mask_neg.sum())}/{n_neg})\n")

        # この時点ではまだ書き込まない。全距離終了後に一括出力する(途中失敗で半端なファイルを残さない)
        rows_out.append({
            "depth": depth_value, "zero_disp_pos": b_pos, "zero_disp_neg": b_neg,
            "a_pos": a_pos, "a_neg": a_neg,
            "inlier_pos": inlier_pos, "inlier_neg": inlier_neg,
            "n_pos": n_pos, "n_neg": n_neg,
        })

        # 散布図でインライヤ(色)/アウトライヤ(灰)と RANSAC 直線を確認(表示専用)
        if args.show_plot:
            import matplotlib.pyplot as plt
            h_px = image.shape[0]
            fig, ax = plt.subplots()
            _plot_branch(ax, d_pos, y_pos, mask_pos, a_pos, b_pos, h_px,
                         "tab:blue", f"pos ({inlier_pos*100:.0f}%)")
            _plot_branch(ax, d_neg, y_neg, mask_neg, a_neg, b_neg, h_px,
                         "tab:orange", f"neg ({inlier_neg*100:.0f}%)")
            ax.set_xlim(-height - 1, height + 1); ax.set_ylim(0, h_px)
            ax.set_xlabel("disparity"); ax.set_ylabel("row (y)")
            ax.set_title(f"depth = {depth_value} mm"); ax.legend()
            plt.show(); plt.close(fig)

    # --- 出力: 全距離がスキップされた場合は明示的に失敗させる ---
    if not rows_out:
        raise RuntimeError("有効なキャリブレーション距離が1つもありませんでした。")

    config.PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    # 列順は既存の参照実装の params.csv に一致させる(near-match 照合のため)。
    # スキップで追加順が乱れ得るので、距離昇順に並べ直してから出力する
    df = pd.DataFrame(rows_out, columns=[
        "depth", "zero_disp_pos", "zero_disp_neg", "a_pos", "a_neg",
        "inlier_pos", "inlier_neg", "n_pos", "n_neg",
    ]).sort_values("depth").reset_index(drop=True)
    df.to_csv(config.PARAMS_CSV_PATH, index=False)         # fit_params.py への入力(全9列)
    print(f"\n{config.PARAMS_CSV_PATH} に保存（{len(df)} 距離）")

    # 診断サマリ(インライヤ率・点数のみ)。校正品質の記録用(docs/module-calibration.md)
    df[["depth", "inlier_pos", "inlier_neg", "n_pos", "n_neg"]].to_csv(
        config.DIAG_CSV_PATH, index=False)
    print(f"{config.DIAG_CSV_PATH} に診断サマリを保存")


if __name__ == "__main__":
    main()