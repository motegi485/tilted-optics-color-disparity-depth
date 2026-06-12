"""rt-tilted-dfd-system 全体の定数・パスの一元管理。
core/ はこれを import しない（純関数・引数注入）。値の注入は呼び出し側の責務。
"""
from pathlib import Path


# ===== パス（__file__ 基準で cwd 非依存）=====
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
TEST_DIR = DATA_DIR / "test"
PARAMS_DIR = ROOT_DIR / "params"
PARAMS_PATH = PARAMS_DIR / "fit_result.csv"

# ===== 画像処理 =====
# cv2.resize の (width, height) 順。(528,352)=R10 の 3:2 を歪みなくリサイズ
IMAGE_SIZE = (528, 352)

# ===== 視差算出 (core/disparity.py へ注入) =====
KERNEL_SIZE = 17
SEARCH_RANGE_X = 31      # 奇数。Phase E で拡大
EDGE_TH = 30            # 校正・推論で統一

# ===== ゲーティング (evaluate/pipeline 層で適用) =====
DISP_MAX = (SEARCH_RANGE_X - 1) // 2   # = 15。search 半幅と一致
DEPTH_MIN = 0
DEPTH_MAX = 50000
EPS_DEN = 1.0           # compute_depth へ注入（分母ゼロガード）