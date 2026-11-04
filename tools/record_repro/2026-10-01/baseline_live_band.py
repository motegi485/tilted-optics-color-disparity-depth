"""B5 の比較基準: live 係数で再計算した値を出す。

既存の視差キャッシュ outputs/cache/disp/K25_S41_T20/（各 frame 10..39 の 30 フレーム・
gate1 前の生視差）に、live 係数（params/fit_result.csv を core.params.load_params
で読む＝runner と同じ経路）を当て、推論帯 (0.45, 5)（本番）／(0.45, 7.5)／なし で距離別と held-out・外挿の要約を出す。
held-out = 校正範囲 1400..3200 内の 9 距離（1500..3100）、外挿 = 1000..1300・3300..4000 の 8 距離。
自己検証として B3_recheck/K25/settings.csv の係数・推論帯 (0.45, 5) の interior MAE 平均を再現する。
読み取りのみ（キャッシュを書かない）。統計の定義は tools.analysis_common.evaluate_session / summarize_holdout。
実行（開発リポの root）: uv run python -u tools/record_repro/2026-10-01/baseline_live_band.py --out <csv>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import config
from checks.evaluate_distance_accuracy import load_manifest
from core.params import DepthParams, load_params
from tools.analysis_common import evaluate_session, load_stack, summarize_holdout

CACHE = Path(config.OUTPUTS_DIR) / "cache" / "disp" / "K25_S41_T20"
MANIFEST = Path("outputs/distance_manifest_all.csv")
RECHECK_CSV = Path("project_knowledge/records/2026-09-18/B3_recheck/K25/settings.csv")
KEYS = ["frames_used", "n_valid_total", "valid_rate", "mean_est_mm", "mean_err_mm", "mae_mm", "std_err_mm",
        "rmse_mm", "median_err_mm", "frame_mean_est_std_mm"]


def recheck_params() -> DepthParams:
    with RECHECK_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    r = next(r for r in rows if r["width"] == "528" and r["kernel_size"] == "25")  # 528/K25 の行
    return DepthParams.from_dict({k: float(r[k]) for k in ("v_pos", "u_pos", "a_pos", "v_neg", "u_neg", "a_neg")}
                                 | {"w_pos": 0.0, "w_neg": 0.0})


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    live = load_params(config.PARAMS_PATH)
    print("live params (load_params):", live, flush=True)
    sessions = [(s, gt, "held-out" if note.startswith("held-out") else "extrapolation")
                for s, gt, note in load_manifest(MANIFEST)]
    stacks = {}
    for s, _gt, _g in sessions:
        files = sorted((CACHE / s).glob("??????.npy"))
        if len(files) != 30:
            print(f"[ERROR] キャッシュ {s} が 30 枚ではありません: {len(files)}")
            return 2
        stacks[s] = load_stack(files)
    rows = []
    sets = [("live", live)]
    try:
        sets.append(("recheck_7.1 (self-check)", recheck_params()))
    except (OSError, StopIteration, KeyError, ValueError) as e:
        print(f"[WARN] §7.1 の係数を読めません: {e!r}")
    for pname, params in sets:
        for bname, band in (("band(0.45,5)", (0.45, 5.0)), ("band(0.45,7.5)", (0.45, 7.5)), ("none(gate1 only)", None)):
            res = {s: evaluate_session(stacks[s], gt, params, disp_max=config.DISP_MAX, band=band)
                   for s, gt, _g in sessions}
            for s, gt, g in sessions:
                rows.append({"params": pname, "band": bname, "session": s, "gt_mm": gt, "group": g,
                             **{k: res[s][k] for k in KEYS}})
            for g in ("held-out", "extrapolation"):
                h = summarize_holdout(res, [s for s, _gt, gg in sessions if gg == g])
                rows.append({"params": pname, "band": bname, "session": f"SUMMARY {g}", "gt_mm": "", "group": g,
                             "mae_mean_mm": h["mae_mean_mm"], "abs_bias_max_mm": h["abs_bias_max_mm"],
                             "rmse_mean_mm": h["rmse_mean_mm"], "valid_rate_mean": h["valid_rate_mean"],
                             "n_sessions": h["n_sessions"]})
                print(f"{pname:26s} {bname:17s} {g:13s} MAE_mean {h['mae_mean_mm']:.2f} |bias|max "
                      f"{h['abs_bias_max_mm']:.2f} RMSE_mean {h['rmse_mean_mm']:.2f} valid {h['valid_rate_mean']:.4f}",
                      flush=True)
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(args.out, "x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
