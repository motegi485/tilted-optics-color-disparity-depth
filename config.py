"""tilted-optics-color-disparity-depth 全体の定数・パスの一元管理。
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
CALIB_IMGS_DIR = DATA_DIR / "calib_imgs"                      # 校正画像（距離名.png/.jpg。1 回の校正に使う画像群の中で規格が混在していると calibrate.py が拒否）
CALIB_RAW_DIR = DATA_DIR / "calib_raw"                        # 校正撮影の raw（1280x720・バースト平均）。capture_calib --save-raw の保存先。処理解像度を変える実験用
PARAMS_CSV_PATH = PARAMS_DIR / "params.csv"                   # calibrate.py の出力
DIAG_CSV_PATH   = PARAMS_DIR / "calibration_diagnostics.csv" # 校正診断サマリ
CAPTURES_DIR = DATA_DIR / "captures"     # raw 録画先（gitignore）
OUTPUTS_DIR  = ROOT_DIR / "outputs"      # runner の結果保存（gitignore）

# 凍結ベースライン（静止画検証フィクスチャ。evaluate.py が読む側）
BASELINE_DIR = PARAMS_DIR / "baseline"
BASELINE_PARAMS_PATH = BASELINE_DIR / "fit_result.csv"

# ===== 画像処理 =====
# cv2.resize の (width, height) 順。(528,352)=カメラ（Canon EOS R10）の 3:2 を歪みなくリサイズ
IMAGE_SIZE = (528, 352)

# ===== 視差算出 (core/disparity.py へ注入) =====
# KERNEL_SIZE と EDGE_TH は、精度と計算時間・有効画素数のバランスを見て実験的に選んだ値
KERNEL_SIZE = 25         # ZNCC 窓の一辺 [px]（奇数）。この画像サイズ（528×352）で実験的に最も精度が良かった大きさ
SEARCH_RANGE_X = 41      # 水平探索幅 [px]（奇数）。探索範囲は ±(S-1)/2 = ±20 px（ピークを探せるのは両端を除く約 ±19.5 px まで）。対象距離（約 1〜4 m）の予測最大視差は約 14 px で、それに余裕を持たせた値
EDGE_TH = 20            # 縦エッジとみなす水平勾配強度（0〜255）の下限。校正・推論で統一

# ===== ゲーティング (evaluate/pipeline 層で適用) =====
DISP_MAX = (SEARCH_RANGE_X - 1) // 2   # = 20。search 半幅と一致。ZNCC の到達上限は 19.5 px なので実質不発（推論帯の上限に吸収）。安全網として維持
# 推論帯 [px]: 推論時に |d| <= LOW または |d| >= HIGH の画素を NaN にする
# （残るのは LOW < |d| < HIGH。校正帯と同じ厳密不等号）。下限は合焦面（POF: Plane Of Focus）近傍の不感帯、
# 上限は大ぼけ域の誤マッチと遠距離側の偏りを除く値。どちらも実験的に選んだ値。
# 校正（calibrate.py）には掛けない（校正は DISP_PLOT_LOW/HEIGHT の校正帯を使う）。
INFER_BAND_LOW  = 0.45
INFER_BAND_HIGH = 5.0
DEPTH_MIN = 0
DEPTH_MAX = 5000  # 距離 Z の範囲 [mm]。ライブ表示の色の割り当てと、checks/ の距離範囲ゲート（0 < z < DEPTH_MAX）の両方に使う
EPS_DEN = 1.0           # compute_depth へ注入（分母ゼロガード）

# ===== キャリブレーション: 視差プロットのゲート (calibration/calibrate.py) =====
# 生ピクセルを (disparity, row) として抽出する際の |d| のゲート [px]
DISP_PLOT_LOW    = 0.45   # POF 近傍の不感帯（小視差）を除外
DISP_PLOT_HEIGHT = 7.5    # 大視差側で誤マッチが増え、推定が不安定になる領域を除外

# ===== キャリブレーション: RANSAC 直線フィット =====
RANSAC_RESIDUAL_TH  = 8.0  # インライヤ残差閾値 [row]。None なら sklearn の MAD 自動
RANSAC_RANDOM_STATE = 0    # シード固定（再現性確保）
RANSAC_MIN_POINTS   = 10   # 有効な点の個数 n_pos/n_neg が、それぞれフィットを始めるのに十分な数があるかの閾値

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
# 以下 2 つは撮影条件を記録しておくためのメタデータで、計算には使われない（コード中のどこからも参照していない）
TILT_THETA_DEG = 30.0  # レンズ傾斜角θ [deg]
CAMERA_HEIGHT_MM = 245.0  # 撮影条件のメモ（高さの基準は未確定。計算には使われない）
