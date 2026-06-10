"""距離推定パラメータ（参照式 z = v/(y - a·d - u) + w の係数）の定義と読み込み。

DepthParams が係数の single source of truth。CSV からの読み込みは load_params。
core/ は本来 I/O を持たない純関数の場所だが、params.py は「params の読み書き」担当として
要件定義書 §5.3 で I/O を許可された例外（pandas/pathlib に依存してよい唯一の core ファイル）。
"""
from dataclasses import dataclass
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
        """
        required = {
            "v_pos", "u_pos", "a_pos", "w_pos",
            "v_neg", "u_neg", "a_neg", "w_neg",
        }
        missing = required - set(mapping)
        if missing:
            raise KeyError(
                f"params に必要なキーが不足しています: {sorted(missing)}"
            )
        return cls(
            v_pos=float(mapping["v_pos"]),
            u_pos=float(mapping["u_pos"]),
            a_pos=float(mapping["a_pos"]),
            w_pos=float(mapping["w_pos"]),
            v_neg=float(mapping["v_neg"]),
            u_neg=float(mapping["u_neg"]),
            a_neg=float(mapping["a_neg"]),
            w_neg=float(mapping["w_neg"]),
        )


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

    # pandas 依存はこの 1 行で完結させ、from_dict には素の dict だけを渡す
    fit = fit_df.set_index("param")["value"].to_dict()

    return DepthParams.from_dict(fit)