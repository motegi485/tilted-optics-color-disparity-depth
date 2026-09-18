"""P3_verify_rust_subset.py — checks.verify_disparity_rust の実写コーパスを代表に絞った照合。

公式の全件照合は数時間規模になるため、公式スクリプトの synthetic_fixtures（合成・敵対的 21 件）の全件と、
実写コーパスを E の小さい代表 3 件＋ノイズ板 1 件に絞って、同じ compare_one／_passes で判定する。
全件の照合は `uv run python -m checks.verify_disparity_rust` で別途実行する。

実行（開発リポ root）: uv run python -u tools/record_repro/2026-09-18/P3_verify_rust_subset.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2

import config
from checks.verify_disparity_rust import (
    FINE_DELTA_TOL,
    K,
    MAX_DELTA_TOL,
    OVER_FINE_RATE_MAX,
    S,
    TH,
    _passes,
    compare_one,
    synthetic_fixtures,
)
from realtime.camera import preprocess

# 代表フィクスチャ（セッション名 → 先頭フレーム）。E は checks.bench_disparity の定義（Sobel > EDGE_TH の画素数）
SESSIONS = ["scene_01", "sign_near", "sign_far", "dist_2300"]


def real_subset():
    fx = []
    for path in sorted(config.TEST_DIR.glob("*.jpg")):
        img = cv2.resize(cv2.imread(str(path)), config.IMAGE_SIZE)
        blue, _, red = cv2.split(img)
        fx.append((f"test/{path.name}", red, blue))
    for name in SESSIONS:
        pngs = sorted(p for p in (config.CAPTURES_DIR / name).glob("??????.png") if p.stem.isdigit())
        proc = preprocess(cv2.imread(str(pngs[0])))
        blue, _, red = cv2.split(proc)
        fx.append((f"captures/{name}/{pngs[0].name}", red, blue))
    return fx


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    print(f"K={K} S={S} TH={TH}  主 max|Δd| <= {MAX_DELTA_TOL:.0e}px / 副 |Δd|>{FINE_DELTA_TOL:.0e}px の画素率 <= {OVER_FINE_RATE_MAX*100:.2f}%")
    synth = synthetic_fixtures()
    real = real_subset()
    print(f"合成・敵対的: {len(synth)} 件（公式と同じ全件）  実写（絞り込み）: {len(real)} 件 {[n for n, *_ in real]}")
    results = []
    for item in synth + [(n, r, b, K, S, TH) for n, r, b in real]:
        name, red, blue, k, s, th = item
        t0 = time.perf_counter()
        r = compare_one(red, blue, k, s, th)
        dt = time.perf_counter() - t0
        ok = _passes(r, name)
        results.append((name, ok))
        print(f"[{'OK' if ok else 'FAIL'}] {name}: nan_match={r['nan_match']} n_mismatch={r['n_nan_mismatch']} "
              f"max|Δd|={r['max_delta']:.3e} over1e-4={r['n_over_fine']}({r['over_fine_rate']*100:.4f}%) "
              f"determinism={r['determinism_match']} n_valid={r['n_valid_py']}  ({dt:.1f}s)", flush=True)
    n_ok = sum(ok for _, ok in results)
    print(f"\n{n_ok}/{len(results)} PASS")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
