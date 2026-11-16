"""tools/w_offset_sim.py — w=0 固定と深度原点（センサ面 vs レンズ面）のずれの影響を数値で確認する。

背景:
  経験式 z = v/(y − ā·d − u) + w は w=0 に固定している。視差 0 の行 y0(z) = v/z + u は、
  合焦面（y–z 断面では直線）の透視投影から従う逆距離型で、z を投影中心（レンズ面近傍）から
  測る場合に厳密に成り立つ。本システムは z をセンサ面から測るため、レンズ面との原点差 δ
  （本リグでは ≈ 0.1 m）のぶんだけ形が厳密には一致しない。

何をするか:
  真の関係を y0 = A/(z − δ) + B とみなして校正距離ごとに y0 を生成し、w=0 の v/z + u を
  最小二乗でフィットしたときの残差 [px] と、それを深度に換算した誤差 [mm] を出す。
  ノイズは入れない（形の不一致だけを見る）。A, B は現行の係数（params/fit_result.csv）の曲線に
  校正範囲の両端で一致するように決める（＝現行の校正結果と同じ位置に合焦面がある前提）。

用途:
  w=0 固定の妥当性（原点ずれの影響が校正範囲内でどの程度か）を数値で確認する理論補足。
  結果は係数と校正範囲に依存するので、再校正したら再実行して数値を更新する。
  本番の処理経路（core/・realtime/・calibration/）には含まれず、係数・回帰アンカーを変更しない。

実行例（root で）:
  uv run python -m tools.w_offset_sim
  uv run python -m tools.w_offset_sim --delta 0 110 122 --sign pos
  uv run python -m tools.w_offset_sim --params params/baseline/fit_result.csv --z-min 1400 --z-max 3200 --z-step 200

δ の既定値 0 / 110 / 122 mm: 0 は同一性確認、110・122 は関連論文が報告するセンサ–レンズ距離
（≈106–122 mm）を本リグの目安として置いたもの。本リグの実測値は未記録（記録したらそれを渡す）。
"""
# --- tools/ から root の config / core を import するためのブートストラップ ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
from scipy import optimize

import config
from core.params import load_params

# Windows の日本語コンソール（cp932）で非対応文字の print が落ちないよう UTF-8 に固定。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 現行の校正（実機撮影）で使った距離レンジの既定値 [mm]。
DEFAULT_Z_MIN_MM = 1400
DEFAULT_Z_MAX_MM = 3200
DEFAULT_Z_STEP_MM = 200
DEFAULT_DELTAS_MM = (0.0, 110.0, 122.0)


def _w0_model(z: np.ndarray, v: float, u: float) -> np.ndarray:
    """w=0 の視差 0 行モデル y0(z) = v/z + u（fit_params.py と同じ関数形）。"""
    return v / z + u


def simulate(z_mm: np.ndarray, v: float, u: float, delta_mm: float) -> dict:
    """原点差 δ のもとで w=0 モデルをフィットしたときの残差と深度誤差を返す。

    Args:
        z_mm:     校正距離 [mm]（センサ面基準）。1 次元・3 点以上・すべて δ より大きいこと。
        v, u:     現行の係数（同一符号枝の v, u）。y_live(z) = v/z + u が「現行の合焦面位置」。
        delta_mm: 深度原点の差 δ [mm]。真の関係を y0 = A/(z − δ) + B とみなす。

    Returns:
        dict: v_fit, u_fit（w=0 フィット結果）、y_true / y_fit / resid_px / err_mm（距離ごと）、
              resid_rms_px / resid_max_px / err_rms_mm / err_max_mm（要約）。

    Raises:
        ValueError: 入力が条件を満たさないとき（assert は -O で無効化されるため使わない）。
    """
    z = np.asarray(z_mm, dtype=float)
    if z.ndim != 1 or z.size < 3:
        raise ValueError(f"z_mm は 1 次元で 3 点以上必要です（shape={z.shape}）")
    if not np.all(np.isfinite(z)) or np.unique(z).size != z.size:
        raise ValueError("z_mm に非有限値または重複があります")
    if not (np.isfinite(v) and np.isfinite(u) and np.isfinite(delta_mm)):
        raise ValueError(f"v, u, delta_mm は有限値が必要です（v={v!r}, u={u!r}, delta_mm={delta_mm!r}）")
    if delta_mm < 0.0:
        raise ValueError(f"delta_mm は 0 以上が必要です（{delta_mm!r}）")
    if np.any(z <= delta_mm):
        raise ValueError(f"すべての z_mm が delta_mm={delta_mm} より大きい必要があります（min z={z.min()}）")

    # 現行の曲線（w=0）。これを「現行の校正結果が示す合焦面位置」とみなす。
    y_live = _w0_model(z, v, u)

    # 真のモデル y = A/(z − δ) + B を、校正範囲の両端で現行の曲線に一致させる。
    f = 1.0 / (z - delta_mm)
    if f[0] == f[-1]:
        raise ValueError("校正範囲の両端が同一で A を決められません")
    a_true = (y_live[0] - y_live[-1]) / (f[0] - f[-1])
    b_true = y_live[0] - a_true * f[0]
    y_true = a_true * f + b_true

    # w=0 モデルを最小二乗でフィット（fit_params.py と同じ関数形・ノイズなし）。
    (v_fit, u_fit), _ = optimize.curve_fit(_w0_model, z, y_true, p0=[v, u], maxfev=10000)
    y_fit = _w0_model(z, v_fit, u_fit)
    resid_px = y_true - y_fit

    # 深度換算: 真の y0 を w=0 モデルで逆変換したときの深度と、真の深度の差。
    den = y_true - u_fit
    if np.any(np.abs(den) < config.EPS_DEN):
        raise ValueError("深度換算の分母が EPS_DEN 未満になりました（係数か距離レンジを確認してください）")
    z_est = v_fit / den
    err_mm = z_est - z

    return {
        "delta_mm": float(delta_mm),
        "v_fit": float(v_fit),
        "u_fit": float(u_fit),
        "z_mm": z,
        "y_true": y_true,
        "y_fit": y_fit,
        "resid_px": resid_px,
        "err_mm": err_mm,
        "resid_rms_px": float(np.sqrt(np.mean(resid_px ** 2))),
        "resid_max_px": float(np.max(np.abs(resid_px))),
        "err_rms_mm": float(np.sqrt(np.mean(err_mm ** 2))),
        "err_max_mm": float(np.max(np.abs(err_mm))),
    }


def format_result(sign: str, res: dict) -> str:
    """1 条件ぶんの結果を人が読む表形式に整える。"""
    z = res["z_mm"]
    lines = [
        f"[{sign}] delta={res['delta_mm']:6.1f} mm  fit v={res['v_fit']:.4e} u={res['u_fit']:.2f}  "
        f"resid RMS={res['resid_rms_px']:.2f} px max={res['resid_max_px']:.2f} px  "
        f"depth err RMS={res['err_rms_mm']:.1f} mm max|err|={res['err_max_mm']:.1f} mm",
        "   z [mm]    : " + " ".join(f"{int(round(zz)):6d}" for zz in z),
        "   resid [px]: " + " ".join(f"{r:6.2f}" for r in res["resid_px"]),
        "   err [mm]  : " + " ".join(f"{e:6.1f}" for e in res["err_mm"]),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="w=0 固定と深度原点（センサ面 vs レンズ面）のずれ δ の影響を、現行の係数（params/fit_result.csv）で数値確認する")
    parser.add_argument("--params", default=str(config.PARAMS_PATH),
                        help=f"fit_result.csv のパス（既定: {config.PARAMS_PATH}）")
    parser.add_argument("--delta", type=float, nargs="+", default=list(DEFAULT_DELTAS_MM),
                        help="原点差 δ [mm] の候補（既定: 0 110 122）")
    parser.add_argument("--sign", choices=("pos", "neg", "both"), default="both",
                        help="どちらの符号枝の係数を使うか（既定: both）")
    parser.add_argument("--z-min", type=int, default=DEFAULT_Z_MIN_MM, help="校正距離の下限 [mm]")
    parser.add_argument("--z-max", type=int, default=DEFAULT_Z_MAX_MM, help="校正距離の上限 [mm]（含む）")
    parser.add_argument("--z-step", type=int, default=DEFAULT_Z_STEP_MM, help="校正距離の刻み [mm]")
    args = parser.parse_args(argv)

    if args.z_step <= 0 or args.z_max <= args.z_min:
        parser.error("--z-step は正、--z-max は --z-min より大きい必要があります")
    z = np.arange(args.z_min, args.z_max + 1, args.z_step, dtype=float)

    params = load_params(args.params)
    branches = {"pos": (params.v_pos, params.u_pos), "neg": (params.v_neg, params.u_neg)}
    signs = ("pos", "neg") if args.sign == "both" else (args.sign,)

    print(f"params: {args.params}")
    print(f"z: {args.z_min}..{args.z_max} step {args.z_step} mm ({z.size} points)  "
          f"delta: {' '.join(f'{d:g}' for d in args.delta)} mm  (noise-free, shape mismatch only)")
    for sign in signs:
        v, u = branches[sign]
        for delta in args.delta:
            print(format_result(sign, simulate(z, v, u, delta)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
