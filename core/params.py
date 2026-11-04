"""距離推定パラメータ（参照式 z = v/(y - a·d - u) + w の係数）の定義と読み込み。

DepthParams が係数の single source of truth。CSV からの読み込みは load_params。
core/ は本来 I/O を持たない純関数の場所だが、params.py は「params の読み書き」担当として
project_knowledge/modules/module-core.md で I/O を許可された例外として説明されている（pandas/pathlib に依存してよい唯一の core ファイル）。
"""
from dataclasses import dataclass
import math
import pandas as pd
from pathlib import Path


@dataclass(frozen=True)
class DepthParams:
    """参照式の係数一式（pos/neg の 2 セット）。生成後は不変（frozen）。

    pos = 視差 d > 0 側、neg = d < 0 側。前ぼけ/後ろぼけのラベルではなく符号で持つ
    （前後対応は校正の符号規約・リグ構成に依存するため、係数コンテナは符号のみを扱う）。
    w は理論的根拠で 0 固定だが、論文式との対応と将来の再検討のため欄として残す
    （0 固定の強制は fit_params.py 側の責務）。
    """
    v_pos: float
    u_pos: float
    a_pos: float
    w_pos: float
    v_neg: float
    u_neg: float
    a_neg: float
    w_neg: float

    @classmethod
    def from_dict(cls, mapping) -> "DepthParams":
        """名前→値の dict から DepthParams を構築する（検証のみ・I/O なし）。

        必要キーの欠損は KeyError。余分なキー（将来の生成条件メタデータ等）は無視する。
        値は float() で native float に正規化する（pandas 由来の numpy.float64 を core に持ち込まないため）。

        値の妥当性は ValueError で拒否する（入力検証は raise・assert は -O で無効化されるため
        使わない・project_knowledge/implementation/known-issues.md #10）。ここがシステム境界の唯一の検証点であり、通過後は
        compute_depth が無検査で使う:
          - 8係数すべてが有限。NaN/inf の係数は同符号枝の全有効画素を無音で NaN/inf にするため、
            推論に入る前に止める。
          - w_pos == w_neg == 0。w=0 固定は project_knowledge/architecture/theory.md に記載の理論的根拠に基づく設計判断であり、
            CSV の手編集や旧実験ファイルで 0 以外が紛れ込むと全画素に一律オフセットが乗る。
        """
        required = (
            "v_pos", "u_pos", "a_pos", "w_pos",
            "v_neg", "u_neg", "a_neg", "w_neg",
        )
        missing = set(required) - set(mapping)
        if missing:
            raise KeyError(
                f"params に必要なキーが不足しています: {sorted(missing)}"
            )

        values = {}
        for key in required:
            try:
                values[key] = float(mapping[key])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"params の {key} を float に変換できません（値: {mapping[key]!r}）"
                ) from e

        non_finite = [k for k in required if not math.isfinite(values[k])]
        if non_finite:
            detail = ", ".join(f"{k}={values[k]!r}" for k in non_finite)
            raise ValueError(f"params の係数が有限ではありません: {detail}")

        nonzero_w = [k for k in ("w_pos", "w_neg") if values[k] != 0.0]
        if nonzero_w:
            detail = ", ".join(f"{k}={values[k]!r}" for k in nonzero_w)
            raise ValueError(
                f"w は 0 固定です（project_knowledge/architecture/theory.md 参照）。0 以外の値が指定されています: {detail}"
            )

        return cls(**values)


def load_params(path: str | Path) -> DepthParams:
    """fit_result.csv（param, value の 2 列）を読み込んで DepthParams を返す。

    ファイル存在と列形式の検証はここ（I/O 層の責務）。必要キーの検証と型変換は
    DepthParams.from_dict に委譲する（検証ロジックを 1 か所に集約して drift を防ぐ）。
    path は config.py が持つ値を呼び出し側から渡す（パスをハードコードしない）。
    """
    fit_path = Path(path)
    if not fit_path.exists():
        raise FileNotFoundError(
            f"params ファイルが見つかりません: {fit_path}（先に fit_params.py を実行してください）"
        )

    fit_df = pd.read_csv(fit_path)
    if "param" not in fit_df.columns or "value" not in fit_df.columns:
        raise ValueError(
            f"params ファイルに 'param' 列と 'value' 列が必要です: {fit_path}"
        )

    # set_index(...).to_dict() は同名 param を「後勝ち」で黙って潰す。異なる値の v_pos が
    # 2行並んでいても後の行だけが採用され、CSV を目視しない限り気づけないため先に検出する。
    duplicated = sorted(set(fit_df["param"][fit_df["param"].duplicated()]))
    if duplicated:
        raise ValueError(
            f"params ファイルに重複した param があります: {duplicated}（{fit_path}）"
        )

    # pandas 依存はこの 1 行で完結させ、from_dict には素の dict だけを渡す
    fit = fit_df.set_index("param")["value"].to_dict()

    return DepthParams.from_dict(fit)