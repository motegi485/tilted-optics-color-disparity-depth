"""rt-tilted-dfd-system 全体の定数・パスの一元管理。
core/ はこれを import しない（純関数・引数注入）。値の注入は呼び出し側の責務。
"""
from pathlib import Path
import cv2


# ===== パス（__file__ 基準で cwd 非依存）=====
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
TEST_DIR = DATA_DIR / "test"
PARAMS_DIR = ROOT_DIR / "params"
PARAMS_PATH = PARAMS_DIR / "fit_result.csv"
CALIB_IMGS_DIR = DATA_DIR / "calib_imgs"                      # 校正画像（距離名.png/.jpg。キャンペーン内での規格混在は calibrate.py が拒否）
PARAMS_CSV_PATH = PARAMS_DIR / "params.csv"                   # calibrate.py の出力
DIAG_CSV_PATH   = PARAMS_DIR / "calibration_diagnostics.csv" # 校正診断サマリ
CAPTURES_DIR = DATA_DIR / "captures"     # raw 録画先（gitignore）
OUTPUTS_DIR  = ROOT_DIR / "outputs"      # runner の結果保存（task 4 で使用）

# 凍結ベースライン（静止画検証フィクスチャ。evaluate.py が読む側）
BASELINE_DIR = PARAMS_DIR / "baseline"
BASELINE_PARAMS_PATH = BASELINE_DIR / "fit_result.csv"

# ===== 画像処理 =====
# cv2.resize の (width, height) 順。(528,352)=R10 の 3:2 を歪みなくリサイズ
IMAGE_SIZE = (528, 352)

# ===== 視差算出 (core/disparity.py へ注入) =====
KERNEL_SIZE = 17
SEARCH_RANGE_X = 31      # 奇数。長距離化で拡大
EDGE_TH = 30            # 校正・推論で統一

# ===== ゲーティング (evaluate/pipeline 層で適用) =====
DISP_MAX = (SEARCH_RANGE_X - 1) // 2   # = 15。search 半幅と一致
DEPTH_MIN = 0
DEPTH_MAX = 50000  # 現行の Z 範囲。一時変更する場合は必ず戻すこと
EPS_DEN = 1.0           # compute_depth へ注入（分母ゼロガード）

# ===== キャリブレーション: 視差プロットのゲート (calibration/calibrate.py) =====
# 生ピクセルを (disparity, row) として抽出する際の |d| のゲート [px]
DISP_PLOT_LOW    = 0.45   # POF 近傍の不感帯（小視差）を除外
DISP_PLOT_HEIGHT = 7.5    # 過大視差（探索端の誤マッチ）を除外

# ===== キャリブレーション: RANSAC 直線フィット =====
RANSAC_RESIDUAL_TH  = 8.0  # インライヤ残差閾値 [row]。None なら sklearn の MAD 自動
RANSAC_RANDOM_STATE = 0    # シード固定（再現性確保）
RANSAC_MIN_POINTS   = 10   # 有効な点の個数 n_pos/n_neg が、ぞれぞれフィットを始めるのに十分な数があるかの閾値

# ===== webcam capture =====
CAPTURE_BACKEND      = cv2.CAP_MSMF      # 実測で DSHOW は暗フレーム → MSMF
CAPTURE_DEVICE_INDEX = 0
WEBCAM_NATIVE_SIZE   = (1280, 720)       # (W,H) 実測ネイティブ。preprocess/source で検証(raise)
WEBCAM_CROP_W        = 1080              # 3:2 中央クロップ幅（高さは720全行）
LIVE_WAIT_TIMEOUT_S = 8.0    # 実フィード待ちの上限（cold-start 実測 ~3.3s に余裕）
LIVE_DIFF_STD_TH    = 0.5    # 時間方向の画素 std 閾値（placeholder=0.0 / live≈1.3）

# ===== セッション動画（閲覧デバッグ専用。canonical は連番 PNG/npy のまま）=====
SESSION_VIDEO_FPS = 30.0                      # 記録tickの目標fps。webcam実効fpsに合わせ等倍再生で実時間になる
SESSION_VIDEO_FOURCC = "mp4v"                 # OpenCV同梱FFmpegで追加DLL不要（avc1はWindowsで非同梱のことがある）
SESSION_VIDEO_RAW_NAME = "video_raw.mp4"      # captures/{session}/ 直下（前処理後FOVの連続録画）
SESSION_VIDEO_DEPTH_NAME = "video_depth.mp4"  # outputs/{session}/ 直下（表示そのまま・サンプル&ホールド）

# ===== 機材設定値 =====
TILT_THETA_DEG = 30.0  # レンズ傾斜角θ [deg]
CAMERA_HEIGHT_MM = 245.0  # 撮影条件メタデータ