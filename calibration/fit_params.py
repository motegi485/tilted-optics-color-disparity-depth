"""
calibration/fit_params.py

params.csv（calibrate.py の出力）から参照式 y0（d=0のy座標） = v/z + u（w=0 固定）を符号別にフィットし、
推論時に使う係数一式（v, u, w=0, ā × pos/neg）を params/fit_result.csv に書き出す。
（先行研究の参考コードの fit_params.py を rt 構成で再現したもの。参照式作成用の表計算シートの置換）

パイプライン上の位置: calibrate.py → params.csv → 【本ファイル】→ fit_result.csv → core/params.load_params（推論側）

参照式（要件書 §2.4・式(2) で d=0, w=0 とした形）:
    y0（d=0のy座標） = v / z + u
各距離の zero_disp（視差0の y 座標＝直線フィットの切片 b）を z に対して上式でフィットし v, u を得る。
ā は各距離の傾き a のサブセット平均。推論時はこれらを z = v/(y0 - ā·d - u) + w（w=0）に用いる。

設計原則（要件書 §4.1）:
  - calibration/ は core ではない。config import・I/O 可。scipy/pandas を直接使ってよい。
  - 入力検証は raise（assert は -O で無効化されるため使わない）。
  - w=0 固定（2パラメータ）。根拠は要件書 §2.4（過剰自由度による非物理解の回避、および
    Fukino et al., IWAIT 2023 式(11) の参照式の形 ȳ = A/z + B との一致）。w をフリーにしない。
    ※「Michi 経験式との整合」は誤り（Michi ISPACS 2012 式(9) は分母内の定数を持たない別形）。
      本書 式(2) の4パラメータ形の出所は Fukino et al., ICECET 2024 式(10)(14)。詳細は要件書 §2.4。
  - これは経験式（参照式）校正の再現であり、参考論文の幾何モデル校正（A・z_id・θ 同定）とは別物。

実行（推奨。リポジトリ root を cwd に）:
    uv run python -m calibration.fit_params
"""

# --- サブパッケージから root の config / core を import するためのブートストラップ ---
# 推奨実行は `uv run python -m calibration.fit_params`。
# 直接実行 `python calibration/fit_params.py` でも動くよう root を sys.path に通す（calibrate.py と同方式）。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import optimize

import config
from core.params import load_params   # 出力後のラウンドトリップ確認用（推論側と契約一致を保証）


# fit_result.csv に必要な係数キー（core/params.DepthParams.from_dict の要求と完全一致させること）。
# 行順は load_params に影響しないが、先行研究の参考コードの出力順を踏襲して near-match 照合をしやすくする。
PARAM_KEYS = ["v_pos", "u_pos", "w_pos", "a_pos",
              "v_neg", "u_neg", "w_neg", "a_neg"]

# params.csv に最低限必要な列。欠けていれば校正出力が壊れているとみなして raise する。
REQUIRED_COLS = {"depth", "zero_disp_pos", "zero_disp_neg", "a_pos", "a_neg"}


def _filter_finite(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """isfinite でその列の NaN 行(RANSAC コンセンサス不成立相当)のみを除外する(値域は制限しない)。"""
    return df[np.isfinite(df[col])].copy()


def model(z, v, u):
    """参照式 y0 = v/z + u（w=0 固定の2パラメータ）。

    w は理論的根拠（要件書 §2.4）で 0 に固定するため、モデル関数の引数に含めない。
    w=0 の値そのものは出力側（fit_result.csv の w_pos/w_neg）で明示的に 0.0 として書き出す。
    """
    return v / z + u


def _fit_branch(depths, zero_disp, label):
    """1枝（pos または neg）の curve_fit。戻り値 (v, u, 残差RMS[px])。

    p0/maxfev は先行研究の参考コードと同一にして再現性を担保する。
    """
    popt, _ = optimize.curve_fit(
        model, depths, zero_disp, p0=[400000, 50], maxfev=10000)
    v, u = popt
    res = zero_disp - model(depths, *popt)        # 切片フィットの残差（px 単位）
    rms = float(np.sqrt(np.mean(res ** 2)))
    print(f"=== {label} パラメータ ===")
    print(f"  v = {v}")
    print(f"  u = {u}")
    print(f"  w = 0.0 (固定)")
    print(f"  残差RMS = {rms:.4f} px\n")
    return float(v), float(u), rms


def main():
    # Windows の日本語コンソール(cp932)は「ā」等の非対応文字の print で UnicodeEncodeError を
    # 出すため UTF-8 に固定する（realtime/runner.py の main() 冒頭と同じ対処）。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    # --- 入口チェック: 入力ファイルと必要列の存在を先に確認する（入力検証は raise） ---
    if not config.PARAMS_CSV_PATH.exists():
        raise FileNotFoundError(
            f"params.csv が見つかりません: {config.PARAMS_CSV_PATH}"
            f"（先に calibrate.py を実行してください）")

    df = pd.read_csv(config.PARAMS_CSV_PATH)
    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"params.csv に必要な列がありません: {sorted(missing_cols)} "
            f"（{config.PARAMS_CSV_PATH}）")

    # --- 入口チェック: depth 列の妥当性（2026-09 レビュー） ---
    # 参照式 y0 = v/z + u は z>0 が前提（z=0 で v/z が発散し、負の距離は物理的に無意味）。
    # 重複 depth は独立な観測点にならず、2 パラメータフィットの同定性を落とす。
    # calibrate.py の正常出力なら常に満たされるが、手編集や別キャンペーンの連結を想定して
    # ここで止める（fit は入力を trusted としない）。
    depth_raw = pd.to_numeric(df["depth"], errors="coerce").values.astype(float)
    if not np.all(np.isfinite(depth_raw)):
        raise ValueError(
            f"params.csv の depth に有限でない値があります（{config.PARAMS_CSV_PATH}）")
    if np.any(depth_raw <= 0):
        bad = sorted(float(d) for d in depth_raw[depth_raw <= 0])
        raise ValueError(
            f"params.csv の depth は正の距離[mm]である必要があります（不正値: {bad}）")
    if np.unique(depth_raw).size != depth_raw.size:
        raise ValueError(
            f"params.csv に重複した depth があります（1距離1行が前提）: "
            f"{config.PARAMS_CSV_PATH}")

    # --- フィッティング失敗行の除外（pos/neg を独立にフィルタ） ---
    # RANSAC コンセンサス不成立の行は calibrate.py が NaN を書く。isfinite でその行のみを除外する
    # （値域は制限しない）。参照式 y0(z) = v/z + u は z が大きいほど y0 → u に漸近するため、
    # 「zero_disp > 定数」という旧フィルタは Phase E の長距離校正で正当な小/負の切片行を無警告で
    # 落としてしまう（2026-07 レビューで撤廃・旧 MIN_VALID_PX）。
    df_pos = _filter_finite(df, "zero_disp_pos")
    df_neg = _filter_finite(df, "zero_disp_neg")

    # 各枝3点未満では参照式（2パラメータ）を安定にフィットできない → 校正不能として停止
    if len(df_pos) < 3 or len(df_neg) < 3:
        raise RuntimeError(
            f"有効データ点が不足しています: pos={len(df_pos)}点, neg={len(df_neg)}点 "
            f"（各枝3点以上必要）")

    depths_pos    = df_pos["depth"].values.astype(float)
    depths_neg    = df_neg["depth"].values.astype(float)
    zero_disp_pos = df_pos["zero_disp_pos"].values.astype(float)
    zero_disp_neg = df_neg["zero_disp_neg"].values.astype(float)

    # --- 参照式フィット（符号別。w=0 固定の2パラメータ） ---
    # pos = 視差 d>0 側、neg = d<0 側。前ぼけ/後ろぼけの物理ラベルは【要検証】のため使わない（§2.6）。
    v_pos, u_pos, _ = _fit_branch(depths_pos, zero_disp_pos, "pos (d>0)")
    v_neg, u_neg, _ = _fit_branch(depths_neg, zero_disp_neg, "neg (d<0)")
    w_pos = w_neg = 0.0

    # --- 傾き ā の算出（フィルタ後サブセットの単純平均） ---
    # 各距離の直線フィットの傾き a を符号別に平均する。pos は a_pos 列、neg は a_neg 列を使う。
    # 平均を取る前に有限性を確認する。NaN が1つ混ざるだけで ā は NaN になり、その符号枝の
    # 推論結果が全画素 NaN になる（無音で全滅するため、ここで原因つきで止める）。
    for label, sub in (("pos", df_pos), ("neg", df_neg)):
        col = f"a_{label}"
        vals = sub[col].values.astype(float)
        if not np.all(np.isfinite(vals)):
            raise ValueError(
                f"params.csv の {col} に有限でない値があります（{label} 枝・"
                f"該当 depth: {sorted(sub['depth'].values[~np.isfinite(vals)].tolist())}）")

    a_pos = float(np.mean(df_pos["a_pos"].values))
    a_neg = float(np.mean(df_neg["a_neg"].values))
    print(f"\nā_pos = {a_pos}")
    print(f"ā_neg = {a_neg}")

    # --- fit_result.csv へ書き出し（param,value の2列・8行） ---
    # 出力キー名は core/params.DepthParams.from_dict の8キーと完全一致させる（ガードレール）。
    values = {
        "v_pos": v_pos, "u_pos": u_pos, "w_pos": w_pos, "a_pos": a_pos,
        "v_neg": v_neg, "u_neg": u_neg, "w_neg": w_neg, "a_neg": a_neg,
    }
    # 保存前の不変条件: 8係数すべて有限。curve_fit は収束しなくても例外を出さずに
    # inf/nan を返しうるため、壊れた fit_result.csv を書き残さない。
    non_finite = sorted(k for k, v in values.items() if not np.isfinite(v))
    if non_finite:
        raise ValueError(
            f"フィット結果に有限でない係数があります: {non_finite}。"
            f"校正入力（{config.PARAMS_CSV_PATH}）を確認してください")

    result = pd.DataFrame(
        {"param": PARAM_KEYS, "value": [values[k] for k in PARAM_KEYS]})

    config.PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.PARAMS_PATH, index=False)
    print(f"\n{config.PARAMS_PATH} に保存しました")

    # --- ラウンドトリップ確認（推論側 load_params が通ることをここで保証する） ---
    # 出力直後に推論経路と同じ読み取りを実行。キー名・列形式の不一致をその場で検知する。
    load_params(config.PARAMS_PATH)
    print("load_params によるラウンドトリップ確認: OK")


if __name__ == "__main__":
    main()