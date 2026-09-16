"""tools/analysis_common.py — 設定値の最適化実験（別管理の論文計画「設定値の最適化実験ランブック」・2026-09-16）で
共有する解析基盤。本番の処理経路には含まれない。

3 つの部品からなる。いずれも **config.py の値は読むが書き換えない**（設定はすべて引数で受ける）。

1. 視差キャッシュ: 設定（K・S・T・処理解像度）ごとに、保存 PNG（校正画像 `data/calib_imgs/<距離>.png`・
   評価 raw `data/captures/<session>/<idx>.png`）から視差マップを 1 回だけ計算して
   `outputs/cache/disp/<設定キー>/...npy` に置く。校正帯・RANSAC・推論ゲートの探索は視差を変えないので、
   キャッシュを読むだけで何百通りでも回せる。保存する視差は **gate1 前の生値**（gate は評価側で適用する）。
2. 校正の純関数 `calibrate_from_disparity`: 「視差マップ群 → 点抽出（`calibration.calibrate.extract_sign_points`）→
   RANSAC（`calibration.calibrate.ransac_fit_line`）→ 切片 → 参照式 curve_fit（`calibration.fit_params.model`）→
   DepthParams」を、`calibrate.py`／`fit_params.py` の main と同じ手順・同じ既定値で、ファイル I/O と print なしに行う。
   現行設定（0.45, 7.5, th=8, seed=0）で `params/params.csv`・`params/fit_result.csv` を再現することがテストの前提
   （再現性チェック・ランブック「検証」）。
3. 評価器 `evaluate_session`: 「視差スタック ＋ DepthParams ＋ ゲート → 距離別の誤差統計」。有効画素・ROI・統計量の定義は
   `checks.evaluate_distance_accuracy` と同じ（`summarize_errors` を再利用）。推論側に校正帯を適用する実験（A4）のため、
   gate1（|d| > disp_max）に加えて任意の帯 `band=(low, high)` の外側を無効化できる。

用語: `DispSettings.disp_max` は gate1 の上限（config.DISP_MAX と同じ定義 (S−1)/2）、`reach_max` は ZNCC 実装が返しうる
|d| の上限 s_range − 0.5（端 2 ビン除外・|δ| < 0.5。docs/module-core.md）。両者の差が「gate1 は到達不能」の根拠。
"""
from __future__ import annotations

# --- tools/ から root の config / core / calibration / checks / realtime を import するためのブートストラップ ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import optimize

import config
from calibration.calibrate import extract_sign_points, ransac_fit_line
from calibration.fit_params import model as zero_disp_model
from checks.evaluate_distance_accuracy import summarize_errors
from core.depth import compute_depth
from core.disparity import calcDisparityByZNCC
from core.params import DepthParams


# ===== 設定 =====

@dataclass(frozen=True)
class DispSettings:
    """視差推定層の設定（config の既定値で初期化。解像度は (W, H)）。"""
    kernel_size: int = config.KERNEL_SIZE
    search_range_x: int = config.SEARCH_RANGE_X
    edge_th: float = float(config.EDGE_TH)
    image_size: tuple[int, int] = tuple(config.IMAGE_SIZE)

    def __post_init__(self):
        for name in ("kernel_size", "search_range_x"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0 or v % 2 == 0:
                raise ValueError(f"DispSettings.{name} は正の奇数: {v!r}")
        if self.edge_th < 0:
            raise ValueError(f"DispSettings.edge_th は 0 以上: {self.edge_th}")
        w, h = self.image_size
        if w <= 0 or h <= 0:
            raise ValueError(f"DispSettings.image_size は正: {self.image_size}")

    @property
    def disp_max(self) -> float:
        """gate1 の上限 (S−1)/2（config.DISP_MAX と同じ定義）。"""
        return (self.search_range_x - 1) / 2

    @property
    def reach_max(self) -> float:
        """ZNCC 実装が返しうる |d| の上限 s_range − 0.5（端 2 ビン除外・|δ| < 0.5）。"""
        return self.disp_max - 0.5

    def key(self) -> str:
        k = f"K{self.kernel_size}_S{self.search_range_x}_T{self.edge_th:g}"
        if tuple(self.image_size) != tuple(config.IMAGE_SIZE):
            k += f"_W{self.image_size[0]}xH{self.image_size[1]}"
        return k


# ===== 画像 → 視差（realtime.pipeline.process_frame の前半と同じ手順） =====

def preprocess_to(raw: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """realtime.camera.preprocess の処理解像度を引数化した版（3:2 中央クロップ → INTER_AREA 縮小）。

    image_size == config.IMAGE_SIZE のとき camera.preprocess と同一結果（テストで保証）。解像度実験（B3）用。
    """
    h, w = raw.shape[:2]
    if (w, h) != tuple(config.WEBCAM_NATIVE_SIZE):
        raise ValueError(f"preprocess_to: raw は {config.WEBCAM_NATIVE_SIZE} (W,H) を想定。実際は {(w, h)}")
    x0 = (w - config.WEBCAM_CROP_W) // 2
    crop = raw[:, x0:x0 + config.WEBCAM_CROP_W]
    return cv2.resize(crop, tuple(image_size), interpolation=cv2.INTER_AREA)


def calib_image_to_proc(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """calibrate.py と同じ入力ガード（処理解像度そのもの、または 3:2±1 % のフル解像度）→ resize。"""
    h_raw, w_raw = image.shape[:2]
    if (w_raw, h_raw) != tuple(image_size):
        expected_ratio = image_size[0] / image_size[1]
        actual_ratio = w_raw / h_raw
        if abs(actual_ratio - expected_ratio) / expected_ratio > 0.01:
            raise ValueError(f"校正画像のサイズ・アスペクト比が不正: (W={w_raw},H={h_raw})・期待 {image_size} か 3:2")
    return cv2.resize(image, tuple(image_size))


def disparity_of_proc(proc_bgr: np.ndarray, settings: DispSettings) -> np.ndarray:
    """preprocess 済み BGR → 生の視差マップ（float32・NaN・**gate1 は適用しない**）。"""
    if proc_bgr.ndim != 3 or proc_bgr.shape[2] != 3 or proc_bgr.dtype != np.uint8:
        raise ValueError(f"disparity_of_proc: (H,W,3) uint8 を想定。実際は shape={proc_bgr.shape} dtype={proc_bgr.dtype}")
    blue, _, red = cv2.split(proc_bgr)
    return calcDisparityByZNCC(red, blue, settings.kernel_size, settings.search_range_x, settings.edge_th)


def edge_strength_of_proc(proc_bgr: np.ndarray) -> np.ndarray:
    """赤チャンネルの水平 Sobel 絶対値（uint8）。core/_disparity_reference.py の手順 1（エッジマスクの元）と同じ。"""
    _, _, red = cv2.split(proc_bgr)
    return cv2.convertScaleAbs(cv2.Sobel(red, cv2.CV_32F, 1, 0))


# ===== 視差キャッシュ =====

def cache_root(settings: DispSettings, root: Path | None = None) -> Path:
    return (Path(root) if root is not None else Path(config.OUTPUTS_DIR)) / "cache" / "disp" / settings.key()


def list_frames(cap_dir: Path, *, skip_first: int = 0, max_frames: int | None = None, step: int = 1) -> list[Path]:
    """runner が保存した連番 PNG を昇順で返し、暖機除外・上限・間引きを適用する。"""
    if skip_first < 0 or step <= 0 or (max_frames is not None and max_frames <= 0):
        raise ValueError(f"list_frames: skip_first={skip_first} max_frames={max_frames} step={step} が不正")
    files = sorted(p for p in Path(cap_dir).glob("??????.png") if p.stem.isdigit())
    if not files:
        raise FileNotFoundError(f"list_frames: {cap_dir} に連番 png がありません")
    files = files[skip_first::step]
    return files[:max_frames] if max_frames is not None else files


def ensure_capture_disparity(settings: DispSettings, session: str, frames: list[Path], *,
                             root: Path | None = None, overwrite: bool = False) -> list[Path]:
    """評価 raw PNG → preprocess_to → 視差 → `<cache>/<session>/<idx>.npy`。既存はスキップ（overwrite で再計算）。"""
    out_dir = cache_root(settings, root) / session
    out_dir.mkdir(parents=True, exist_ok=True)
    outs: list[Path] = []
    for p in frames:
        dst = out_dir / f"{p.stem}.npy"
        if overwrite or not dst.exists():
            raw = cv2.imread(str(p))
            if raw is None:
                raise ValueError(f"ensure_capture_disparity: png を読めません: {p}")
            np.save(str(dst), disparity_of_proc(preprocess_to(raw, settings.image_size), settings))
        outs.append(dst)
    return outs


def ensure_calib_disparity(settings: DispSettings, calib_dir: Path, *,
                           root: Path | None = None, overwrite: bool = False) -> dict[int, Path]:
    """校正画像（距離名 .png/.jpg）→ calib_image_to_proc → 視差 → `<cache>/calib/<距離>.npy`。戻り値は距離 → npy。"""
    calib_dir = Path(calib_dir)
    paths = sorted(p for p in calib_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.stem.isdigit())
    if not paths:
        raise FileNotFoundError(f"ensure_calib_disparity: {calib_dir} に距離名の校正画像がありません")
    out_dir = cache_root(settings, root) / "calib"
    out_dir.mkdir(parents=True, exist_ok=True)
    outs: dict[int, Path] = {}
    for p in paths:
        depth = int(p.stem)
        if depth in outs:
            raise ValueError(f"ensure_calib_disparity: 同一距離の校正画像が重複: {depth}")
        dst = out_dir / f"{depth}.npy"
        if overwrite or not dst.exists():
            image = cv2.imread(str(p))
            if image is None:
                raise ValueError(f"ensure_calib_disparity: 画像を読めません: {p}")
            np.save(str(dst), disparity_of_proc(calib_image_to_proc(image, settings.image_size), settings))
        outs[depth] = dst
    return outs


def load_stack(paths: list[Path]) -> np.ndarray:
    """npy 群を (N, H, W) float32 に積む（形状不一致は raise）。"""
    if not paths:
        raise ValueError("load_stack: paths が空です")
    arrs = [np.load(str(p)) for p in paths]
    shape = arrs[0].shape
    for p, a in zip(paths, arrs):
        if a.shape != shape:
            raise ValueError(f"load_stack: 形状不一致 {p}: {a.shape} != {shape}")
    return np.stack(arrs).astype(np.float32, copy=False)


# ===== 校正（calibrate.py + fit_params.py の main と同じ手順・I/O なし） =====

@dataclass
class CalibResult:
    params: DepthParams
    rows: list[dict] = field(default_factory=list)      # params.csv と同じ列（距離ごと）
    fit_rms_px: dict = field(default_factory=dict)      # {"pos": ..., "neg": ...} 切片フィット残差 RMS [px]
    depth_rmse_mm: dict = field(default_factory=dict)   # {"pos", "neg", "combined"} 切片からの逆算誤差 RMSE [mm]
    depth_err_mm: dict = field(default_factory=dict)    # {"pos": {depth: err}, "neg": {...}}
    n_used: dict = field(default_factory=dict)          # {"pos": 有効距離数, "neg": ...}
    skipped: list[int] = field(default_factory=list)    # 点数不足で飛ばした距離


def calibrate_from_disparity(disp_by_depth: dict[int, np.ndarray], *, low: float, high: float,
                             ransac_th: float | None = config.RANSAC_RESIDUAL_TH,
                             ransac_seed: int = config.RANSAC_RANDOM_STATE,
                             min_points: int = config.RANSAC_MIN_POINTS) -> CalibResult:
    """距離 → 視差マップ（gate1 前でよい。帯 low<|d|<high で抽出するため）から 8 係数を求める。

    手順は calibrate.py（点抽出 → RANSAC・コンセンサス不成立や傾き非同定は NaN 行）と fit_params.py
    （isfinite フィルタ → 各枝 3 点未満で RuntimeError → curve_fit(p0=[400000,50], maxfev=10000) → ā は平均）に
    一致させる。ransac_th=None は sklearn の MAD 自動。深度換算 RMSE は記録票 §5 と同じ定義
    （各距離の切片 y0 から z = v/(y0 − u) を逆算した誤差の RMSE）。
    """
    if not (0.0 <= low < high):
        raise ValueError(f"calibrate_from_disparity: 0 <= low < high が必要: low={low} high={high}")
    if not disp_by_depth:
        raise ValueError("calibrate_from_disparity: disp_by_depth が空です")
    rows: list[dict] = []
    skipped: list[int] = []
    for depth in sorted(disp_by_depth):
        if depth <= 0:
            raise ValueError(f"calibrate_from_disparity: 距離は正: {depth}")
        disp = disp_by_depth[depth]
        d_pos, y_pos = extract_sign_points(disp, +1, low, high)
        d_neg, y_neg = extract_sign_points(disp, -1, low, high)
        n_pos, n_neg = len(d_pos), len(d_neg)
        if n_pos < min_points or n_neg < min_points:
            skipped.append(depth)
            continue
        fits = {}
        for label, d_vals, y_vals, n in (("pos", d_pos, y_pos, n_pos), ("neg", d_neg, y_neg, n_neg)):
            try:
                a, b, mask = ransac_fit_line(d_vals, y_vals, ransac_th, ransac_seed)
            except ValueError:
                a, b, mask = np.nan, np.nan, np.zeros(n, dtype=bool)
            fits[label] = (a, b, float(mask.mean()) if n else 0.0)
        rows.append({
            "depth": depth,
            "zero_disp_pos": fits["pos"][1], "zero_disp_neg": fits["neg"][1],
            "a_pos": fits["pos"][0], "a_neg": fits["neg"][0],
            "inlier_pos": fits["pos"][2], "inlier_neg": fits["neg"][2],
            "n_pos": n_pos, "n_neg": n_neg,
        })
    if not rows:
        raise RuntimeError(f"calibrate_from_disparity: 全距離が点数不足でスキップ（min_points={min_points}）")

    result = CalibResult(params=None)  # type: ignore[arg-type]
    coeffs: dict[str, tuple[float, float, float]] = {}
    for label in ("pos", "neg"):
        sub = [r for r in rows if np.isfinite(r[f"zero_disp_{label}"])]
        if len(sub) < 3:
            raise RuntimeError(f"calibrate_from_disparity: {label} 枝の有効点が {len(sub)} 点（3 点以上必要）")
        depths = np.array([r["depth"] for r in sub], dtype=float)
        y0 = np.array([r[f"zero_disp_{label}"] for r in sub], dtype=float)
        a_vals = np.array([r[f"a_{label}"] for r in sub], dtype=float)
        if not np.all(np.isfinite(a_vals)):
            raise ValueError(f"calibrate_from_disparity: {label} 枝の a に有限でない値があります")
        popt, _ = optimize.curve_fit(zero_disp_model, depths, y0, p0=[400000, 50], maxfev=10000)
        v, u = float(popt[0]), float(popt[1])
        res = y0 - zero_disp_model(depths, v, u)
        result.fit_rms_px[label] = float(np.sqrt(np.mean(res ** 2)))
        with np.errstate(divide="ignore", invalid="ignore"):
            z_hat = v / (y0 - u)
        err = z_hat - depths
        result.depth_err_mm[label] = {int(d): float(e) for d, e in zip(depths, err)}
        result.depth_rmse_mm[label] = float(np.sqrt(np.mean(err ** 2)))
        result.n_used[label] = len(sub)
        coeffs[label] = (v, u, float(np.mean(a_vals)))
    all_err = np.array(list(result.depth_err_mm["pos"].values()) + list(result.depth_err_mm["neg"].values()))
    result.depth_rmse_mm["combined"] = float(np.sqrt(np.mean(all_err ** 2)))
    values = {
        "v_pos": coeffs["pos"][0], "u_pos": coeffs["pos"][1], "w_pos": 0.0, "a_pos": coeffs["pos"][2],
        "v_neg": coeffs["neg"][0], "u_neg": coeffs["neg"][1], "w_neg": 0.0, "a_neg": coeffs["neg"][2],
    }
    result.params = DepthParams.from_dict(values)   # 8 係数の有限性はここで検証される
    result.rows = rows
    result.skipped = skipped
    return result


# ===== 評価（checks.evaluate_distance_accuracy と同じ定義） =====

def apply_gates(disp: np.ndarray, *, disp_max: float, band: tuple[float, float] | None = None) -> np.ndarray:
    """gate1（|d| > disp_max → NaN）と、任意の推論帯（|d| <= low または |d| >= high → NaN）を適用したコピーを返す。"""
    out = disp.astype(np.float32, copy=True)
    ad = np.abs(out)
    with np.errstate(invalid="ignore"):
        out[ad > disp_max] = np.nan
        if band is not None:
            low, high = band
            if not (0.0 <= low < high):
                raise ValueError(f"apply_gates: 0 <= low < high が必要: band={band}")
            out[(ad <= low) | (ad >= high)] = np.nan
    return out


def depth_of(disp: np.ndarray, params: DepthParams) -> np.ndarray:
    return compute_depth(disp, params.v_pos, params.u_pos, params.a_pos, params.w_pos,
                         params.v_neg, params.u_neg, params.a_neg, params.w_neg, config.EPS_DEN)


def evaluate_session(disp_stack: np.ndarray, gt_mm: float, params: DepthParams, *,
                     roi: tuple[int, int, int, int] | None = None,
                     depth_min: float = config.DEPTH_MIN, depth_max: float = config.DEPTH_MAX,
                     disp_max: float = config.DISP_MAX, band: tuple[float, float] | None = None) -> dict:
    """視差スタック (N,H,W) → gate → 参照式 → gate2 → pooled 誤差統計（distance_accuracy と同じ列名）。"""
    if disp_stack.ndim != 3:
        raise ValueError(f"evaluate_session: (N,H,W) を想定。実際は {disp_stack.shape}")
    n, h, w = disp_stack.shape
    x0, y0, x1, y1 = roi if roi is not None else (0, 0, w, h)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f"evaluate_session: ROI {roi} が (W={w}, H={h}) の範囲外")
    per_frame_valid: list[int] = []
    per_frame_mean: list[float] = []
    chunks: list[np.ndarray] = []
    for k in range(n):
        depth = depth_of(apply_gates(disp_stack[k], disp_max=disp_max, band=band), params)
        sub = depth[y0:y1, x0:x1].astype(np.float64)
        valid = sub[(sub > depth_min) & (sub < depth_max)]
        per_frame_valid.append(int(valid.size))
        if valid.size:
            per_frame_mean.append(float(valid.mean()))
            chunks.append(valid - float(gt_mm))
    n_roi = (x1 - x0) * (y1 - y0)
    n_valid_total = int(sum(per_frame_valid))
    out: dict = {
        "gt_mm": gt_mm, "frames_used": n, "n_valid_total": n_valid_total,
        "valid_rate": n_valid_total / (n_roi * n),
        "n_valid_min": int(min(per_frame_valid)), "n_valid_max": int(max(per_frame_valid)),
    }
    if n_valid_total == 0:
        out.update({k: float("nan") for k in ("mean_est_mm", "mean_err_mm", "mae_mm", "std_err_mm", "rmse_mm",
                                             "median_err_mm", "frame_mean_est_std_mm")})
        return out
    errors = np.concatenate(chunks)
    out["mean_est_mm"] = float(np.mean(errors) + gt_mm)
    out.update(summarize_errors(errors))
    fm = np.asarray(per_frame_mean, dtype=np.float64)
    out["frame_mean_est_std_mm"] = float(np.std(fm, ddof=1)) if fm.size >= 2 else float("nan")
    return out


def summarize_holdout(metrics_by_session: dict[str, dict], sessions: list[str]) -> dict:
    """複数セッションの指標を 1 行に要約する（最適化の目的関数の材料）。NaN のセッションは除外。"""
    rows = [metrics_by_session[s] for s in sessions if s in metrics_by_session]
    rows = [r for r in rows if r["n_valid_total"] > 0]
    if not rows:
        return {"n_sessions": 0, "mae_mean_mm": float("nan"), "abs_bias_max_mm": float("nan"),
                "rmse_mean_mm": float("nan"), "valid_rate_mean": float("nan")}
    return {
        "n_sessions": len(rows),
        "mae_mean_mm": float(np.mean([r["mae_mm"] for r in rows])),
        "abs_bias_max_mm": float(np.max([abs(r["mean_err_mm"]) for r in rows])),
        "rmse_mean_mm": float(np.mean([r["rmse_mm"] for r in rows])),
        "valid_rate_mean": float(np.mean([r["valid_rate"] for r in rows])),
    }
