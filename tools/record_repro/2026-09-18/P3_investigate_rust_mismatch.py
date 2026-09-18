"""P3_investigate_rust_mismatch.py — Rust↔Python の全件照合で出た不一致の原因調査。

各セッションの先頭フレームで Python 参照実装と Rust 実装の視差を比べ、不一致画素の座標を特定し、
その画素の ZNCC 類似度配列（Python 側・cv2.matchTemplate の f32 値）を再計算して、
  - 厳密局所最大の判定（sim_peak > sim_neg かつ > sim_pos）の差分が f32 の 1ulp 程度か（NaN 位置不一致の場合）
  - 上位 2 ピークの差が 1ulp 程度か（別ピークが選ばれ |Δd| が数 px になる場合）
を数値で示す（project_knowledge/modules/module-core.md「実写照合で不一致が出た場合の解釈指針: f32 の 1ulp 差による近接タイの反転を第一仮説」の検証）。

実行（開発リポ root）: uv run python -u tools/record_repro/2026-09-18/P3_investigate_rust_mismatch.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2
import numpy as np

import config
from core._disparity_reference import calcDisparityByZNCC as calc_python
from core.disparity import calcDisparityByZNCC as calc_rust
from realtime.camera import preprocess

SESSIONS = ["20260715_181809", "20260806_103931", "dist_3500"]
K, S, TH = config.KERNEL_SIZE, config.SEARCH_RANGE_X, config.EDGE_TH


def sim_array_at(red, blue, y, x):
    """参照実装のループ本体をそのまま 1 画素分だけ再現し、sim_array と選択結果を返す。"""
    k_size, s_range = (K - 1) // 2, (S - 1) // 2
    er = cv2.convertScaleAbs(cv2.Sobel(red, cv2.CV_32F, 1, 0))
    eb = cv2.convertScaleAbs(cv2.Sobel(blue, cv2.CV_32F, 1, 0))
    roi_red = er[y - k_size:y + k_size + 1, x - k_size:x + k_size + 1]
    sim = np.zeros(S, dtype=np.float32)
    max_sim, max_x = -math.inf, -1
    for shift_x in range(S):
        sx = x + s_range - k_size - shift_x
        roi_blue = eb[y - k_size:y + k_size + 1, sx:sx + K]
        _, max_val, _, _ = cv2.minMaxLoc(cv2.matchTemplate(roi_red, roi_blue, cv2.TM_CCOEFF_NORMED))
        sim[shift_x] = max_val
        if max_val > max_sim and shift_x != 0 and shift_x != S - 1:
            max_sim, max_x = max_val, shift_x
    return sim, max_x


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    s_range = (S - 1) // 2
    print(f"K={K} S={S} TH={TH}  s_range={s_range}")
    for name in SESSIONS:
        pngs = sorted(p for p in (config.CAPTURES_DIR / name).glob("??????.png") if p.stem.isdigit())
        proc = preprocess(cv2.imread(str(pngs[0])))
        blue, _, red = cv2.split(proc)
        d_py = calc_python(red, blue, K, S, TH)
        d_rs = calc_rust(red, blue, K, S, TH)
        nan_mis = np.argwhere(np.isnan(d_py) != np.isnan(d_rs))
        both = ~np.isnan(d_py) & ~np.isnan(d_rs)
        delta = np.zeros_like(d_py)
        delta[both] = np.abs(d_py[both] - d_rs[both])
        val_mis = np.argwhere(delta > 1e-3)
        print(f"\n=== {name}/{pngs[0].name}: NaN 位置不一致 {len(nan_mis)} 画素・|Δd|>1e-3 {len(val_mis)} 画素 ===")
        for (y, x) in list(map(tuple, nan_mis)) + list(map(tuple, val_mis)):
            sim, max_x = sim_array_at(red, blue, y, x)
            print(f"  画素 (y={y}, x={x}): python d={d_py[y, x]!r}  rust d={d_rs[y, x]!r}")
            order = np.argsort(sim)[::-1]
            top = [(int(i), float(sim[i])) for i in order[:4]]
            print(f"    python の max_x={max_x}  sim 上位4 (shift, sim): {top}")
            if max_x >= 1:
                sp, sn, spos = float(sim[max_x]), float(sim[max_x - 1]), float(sim[max_x + 1])
                ulp = float(np.spacing(np.float32(sp)))
                print(f"    sim_peak={sp!r} sim_neg={sn!r} sim_pos={spos!r}  peak-neg={sp - sn:.3e} peak-pos={sp - spos:.3e}  1ulp(f32)={ulp:.3e}")
            # Rust が選んだシフト（|Δd| の場合）: d = max_x + δ − s_range から逆算
            if not np.isnan(d_rs[y, x]):
                mx_rs = int(round(float(d_rs[y, x]) + s_range))
                print(f"    rust が選んだ max_x≈{mx_rs} → python の sim[{mx_rs}]={float(sim[mx_rs])!r}"
                      f"（python の peak との差 {float(sim[max_x]) - float(sim[mx_rs]):.3e}）" if max_x >= 1 else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
