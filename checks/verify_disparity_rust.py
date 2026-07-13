"""checks/verify_disparity_rust.py — Phase D §5.6 受入基準1,2 の照合ゲート。

core._disparity_reference（凍結 Python 参照実装）と _rt_disparity_native（Rust 実装）を
同一入力で実行し、NaN 位置の完全一致・有限画素の max|Δd| <= 1e-4px を検証する。
あわせて Rust の sequential(parallel=False) vs parallel(True) の決定論一致も確認する
（rayon 並列化がデータ競合なく同じ結果を返すことの根拠）。

対象入力:
  - tests/test_core.py 由来の合成パターン（既知シフト・平坦blue）
  - 敵対的入力（全面平坦・templ側飽和隣接エッジ・周期縞・極小画像・縮退パラメータ・
    閾値で全画素除外）
  - data/test/*.jpg・data/captures/*/ の実写コーパス（存在すれば。現状は空でもFAILにはせず
    「実行できなかった」ことを明示する＝§5.6基準1-3はデータ復元後に別途確認が必要）

実行: uv run python -m checks.verify_disparity_rust
"""
# --- checks/ から root の config / core / realtime を import するためのブートストラップ ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config
from core._disparity_reference import calcDisparityByZNCC as calc_python

try:
    import _rt_disparity_native as native
except ImportError as e:
    print(f"[FATAL] _rt_disparity_native が見つかりません: {e}")
    print("  rust/disparity/ で `maturin build --release` -> "
          "`uv pip install (Get-ChildItem target/wheels/*.whl).FullName` を実行してください。")
    sys.exit(2)

K, S, TH = config.KERNEL_SIZE, config.SEARCH_RANGE_X, config.EDGE_TH
MAX_DELTA_TOL = 1e-4  # px（§5.6 受入基準2・確定）


def calc_rust(red, blue, k, s, th, parallel):
    return native.calc_disparity_by_zncc(red, blue, k, s, float(th), parallel)


# ===== 合成・敵対的フィクスチャ =====

def _make_shift_pattern(width, height, edge_col, seed=42):
    rng = np.random.default_rng(seed)
    base = 100 + rng.integers(-3, 4, size=width)
    base = base.astype(np.int32)
    base[edge_col:] += 150
    pattern = np.clip(base, 0, 255).astype(np.uint8)
    return np.broadcast_to(pattern, (height, width)).copy()


def synthetic_fixtures():
    fx = []

    # 既知シフト（tests/test_core.py 由来。blue(c) = red(c+shift) となるよう edge_col をずらす）
    for shift in [0, 1, -1, 3, -3, 7, -7]:
        width, height, edge_col = 140, 40, 70
        red = _make_shift_pattern(width, height, edge_col)
        blue = _make_shift_pattern(width, height, edge_col - shift)
        fx.append((f"known_shift_{shift}", red, blue, K, S, TH))

    # 平坦blue（飽和含む）: templ側分散ゼロの縮退分岐（§5.6・全シフトタイでNaN想定）
    for blue_const in [0, 128, 255]:
        height, width = 60, 100
        red = np.full((height, width), 50, dtype=np.uint8)
        red[:, 50:] = 200
        blue = np.full((height, width), blue_const, dtype=np.uint8)
        fx.append((f"flat_blue_{blue_const}", red, blue, K, S, TH))

    # 全面平坦（両方）: エッジが立たず全NaNになるはずのpanic-freeチェック
    red = np.full((80, 150), 100, dtype=np.uint8)
    blue = np.full((80, 150), 100, dtype=np.uint8)
    fx.append(("fully_flat_both", red, blue, K, S, TH))

    # templ(blue)側の探索範囲内に飽和ブロック: 実写のハイライト飛びを模した縮退分岐の混在
    height, width = 80, 150
    rng = np.random.default_rng(7)
    red = rng.integers(0, 255, (height, width)).astype(np.uint8)
    red[:, 70:] = np.clip(red[:, 70:].astype(np.int32) + 150, 0, 255).astype(np.uint8)
    blue = rng.integers(0, 255, (height, width)).astype(np.uint8)
    blue[:, 60:100] = 255
    fx.append(("saturated_patch_in_search_range", red, blue, K, S, TH))

    # 周期縞: 非縮退な近似タイのストレステスト（複数shiftが酷似したZNCC値を返し得る）
    height, width = 60, 160
    period = 4
    col = np.arange(width)
    stripe = ((col // period) % 2 * 200 + 20).astype(np.uint8)
    red = np.broadcast_to(stripe, (height, width)).copy()
    blue = np.broadcast_to(stripe, (height, width)).copy()
    fx.append(("periodic_stripe", red, blue, K, S, TH))

    # 極小画像: 有効範囲が空になるケースのpanic-freeチェック
    for h, w in [(5, 5), (K, K), (K + 2, S + K + 4)]:
        rng = np.random.default_rng(3)
        red = rng.integers(0, 255, (h, w)).astype(np.uint8)
        blue = rng.integers(0, 255, (h, w)).astype(np.uint8)
        fx.append((f"tiny_{h}x{w}", red, blue, K, S, TH))

    # 縮退パラメータ（極小 kernel_size/search_range_x）
    rng = np.random.default_rng(11)
    red = rng.integers(0, 255, (60, 120)).astype(np.uint8)
    blue = rng.integers(0, 255, (60, 120)).astype(np.uint8)
    for k, s in [(1, 1), (1, 3), (3, 1), (3, 3)]:
        fx.append((f"degenerate_k{k}_s{s}", red, blue, k, s, TH))

    # 閾値で全画素除外
    red = np.full((60, 120), 10, dtype=np.uint8)
    blue = np.full((60, 120), 10, dtype=np.uint8)
    fx.append(("th_excludes_all", red, blue, K, S, 250))

    return fx


def real_corpus_fixtures():
    """data/test/*.jpg・data/captures/*/ の実写コーパス。無ければ空リストを返す（FAILにしない）。

    captures は各セッションの先頭1フレームのみのサンプリング照合（Python 参照実装が
    ≈数秒〜数十秒/frame かかるため）。クリップ全フレームの再現一致は、タイ→NaN 化（C-1）
    以後に再録画したセッションに対する checks/verify_clip_replay.py の役割（§9.3）。
    """
    fx = []
    for path in sorted(config.TEST_DIR.glob("*.jpg")):
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, config.IMAGE_SIZE)
        blue, _, red = cv2.split(img)
        fx.append((f"test/{path.name}", red, blue, K, S, TH))

    for session_dir in sorted(config.CAPTURES_DIR.glob("*")):
        if not session_dir.is_dir():
            continue
        pngs = sorted(p for p in session_dir.glob("??????.png") if p.stem.isdigit())
        if not pngs:
            continue
        from realtime.camera import preprocess
        raw = cv2.imread(str(pngs[0]))
        if raw is None:
            continue
        proc = preprocess(raw)
        blue, _, red = cv2.split(proc)
        fx.append((f"captures/{session_dir.name}/{pngs[0].name}", red, blue, K, S, TH))

    return fx


def compare_one(red, blue, k, s, th):
    d_py = calc_python(red, blue, k, s, th)
    d_rust_par = calc_rust(red, blue, k, s, th, parallel=True)
    d_rust_seq = calc_rust(red, blue, k, s, th, parallel=False)

    nan_py = np.isnan(d_py)
    nan_par = np.isnan(d_rust_par)
    nan_seq = np.isnan(d_rust_seq)

    nan_match = np.array_equal(nan_py, nan_par)
    determinism_match = np.array_equal(nan_par, nan_seq) and np.array_equal(
        d_rust_par[~nan_par], d_rust_seq[~nan_seq])

    both_finite = ~nan_py & ~nan_par
    if np.any(both_finite):
        delta = np.abs(d_py[both_finite].astype(np.float64)
                        - d_rust_par[both_finite].astype(np.float64))
        max_delta = float(np.max(delta))
    else:
        max_delta = 0.0

    return {
        "nan_match": bool(nan_match),
        "n_nan_mismatch": int(np.sum(nan_py != nan_par)),
        "max_delta": max_delta,
        "determinism_match": bool(determinism_match),
        "n_valid_py": int(np.sum(~nan_py)),
        "dtype_ok": d_rust_par.dtype == np.float32,
        "writable_ok": d_rust_par.flags.writeable,
    }


def _passes(r):
    return (r["nan_match"] and r["max_delta"] <= MAX_DELTA_TOL
            and r["determinism_match"] and r["dtype_ok"] and r["writable_ok"])


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    print(f"_rt_disparity_native.__version__ = {native.__version__}")

    synth_fx = synthetic_fixtures()
    real_fx = real_corpus_fixtures()
    print(f"合成・敵対的フィクスチャ: {len(synth_fx)}件  実写コーパス: {len(real_fx)}件")
    if not real_fx:
        print("[NOTE] data/test・data/captures が空のため実写コーパス照合はスキップ"
              "（§5.6 受入基準1-3はデータ復元後に別途 `uv run python -m checks.verify_disparity_rust` "
              "を再実行して確認すること）")

    results = []
    for name, red, blue, k, s, th in synth_fx + real_fx:
        try:
            r = compare_one(red, blue, k, s, th)
        except Exception as e:
            print(f"[ERROR] {name}: {e!r}")
            results.append((name, {"nan_match": False, "n_nan_mismatch": -1, "max_delta": float("inf"),
                                    "determinism_match": False, "n_valid_py": -1,
                                    "dtype_ok": False, "writable_ok": False}))
            continue
        results.append((name, r))
        status = "OK" if _passes(r) else "FAIL"
        print(f"[{status}] {name}: nan_match={r['nan_match']} n_mismatch={r['n_nan_mismatch']} "
              f"max|Δd|={r['max_delta']:.3e} determinism={r['determinism_match']} "
              f"n_valid={r['n_valid_py']}")

    fail = [(name, r) for name, r in results if not _passes(r)]
    print(f"\n{len(results) - len(fail)}/{len(results)} PASS")

    # 契約チェック（native 単体の防御動作。ラッパー core/disparity.py の契約は
    # tests/test_core.py が担う）: 偶数値・形状不一致は ValueError、空画像は panic せず
    # 空の全NaNマップを返す（w=0×kernel_size=1 で par_chunks_mut(0) panic していた回帰・2026-07-13）
    contract_failures = []
    img = np.zeros((40, 100), dtype=np.uint8)

    def expect_value_error(desc, fn):
        try:
            fn()
            contract_failures.append(f"{desc}: 例外が発生しなかった")
        except ValueError:
            pass
        except BaseException as e:  # PanicException は Exception でなく BaseException 派生
            contract_failures.append(f"{desc}: ValueError でなく {type(e).__name__} が発生: {e}")

    expect_value_error("even kernel_size",
                       lambda: native.calc_disparity_by_zncc(img, img, 16, S, float(TH), True))
    expect_value_error("even search_range_x",
                       lambda: native.calc_disparity_by_zncc(img, img, K, 30, float(TH), True))
    expect_value_error("形状不一致",
                       lambda: native.calc_disparity_by_zncc(
                           img, np.zeros((40, 90), dtype=np.uint8), K, S, float(TH), True))
    try:
        empty = np.zeros((5, 0), dtype=np.uint8)
        r_empty = native.calc_disparity_by_zncc(empty, empty, 1, 1, float(TH), True)
        if r_empty.shape != (5, 0):
            contract_failures.append(f"空画像: 出力形状が {r_empty.shape}（期待 (5, 0)）")
    except BaseException as e:
        contract_failures.append(f"空画像: panic/例外 {type(e).__name__}: {e}")

    for msg in contract_failures:
        print(f"[FAIL] 契約チェック: {msg}")
    if not contract_failures:
        print("契約チェック: OK（ValueError 3種・空画像 panic なし）")

    if not real_fx:
        # 実写コーパスが無いこと自体は FAIL 条件にしない（ユーザー指示により実装優先で進行中のため）が、
        # 明示的に「未検証」であることを最終行に残す。
        print("[PARTIAL] 合成・敵対的入力での正当性検証は完了。実写コーパス(§5.6基準1-3)は未検証。")

    if fail or contract_failures:
        print(f"[FAIL] 不一致 {len(fail)}件: {[name for name, _ in fail]} / "
              f"契約チェック失敗 {len(contract_failures)}件")
        return 1
    print("PASS: 全フィクスチャで Rust版とPython参照版が§5.6基準(NaN一致・|Δd|<=1e-4px)を満たし、"
          "並列/逐次の決定論も確認できた")
    return 0


if __name__ == "__main__":
    sys.exit(main())
