"""scene_cylinder_edge_rows.py — b5_scene_01 frame 10 で、筒の左右輪郭の列の視差を
16 行ごとにまとめ、1430 mm の物体に対する予測視差（live 係数の参照式）と比べる。

目的: 筒の輪郭の深度が一部の行で抜ける理由（推論帯 0.45 < |d| < 5 px）を、予測ではなく
実測の視差で確かめる。読み取りのみ（raw 1 フレームから視差を計算し、保存 npy の深度を読むだけ）。
列の範囲は frame 10 のエッジ画素が集中する列（左 101〜106・右 196〜199）を含むように取った。輪郭の画素には
背景（紙 2800 mm）の画素も混ざりうる。

実行（リポジトリ root）: uv run python tools/record_repro/2026-10-05/scene_cylinder_edge_rows.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from core.params import load_params  # noqa: E402
from realtime.camera import preprocess  # noqa: E402
from tools.analysis_common import DispSettings, disparity_of_proc  # noqa: E402

Z_CYL = 1430.0                     # 筒の距離（レーザ・カメラに最も近い面）
INDEX = 10
EDGES = {"left": (95, 115), "right": (185, 210)}
p = load_params(config.PARAMS_PATH)


def predicted_d(y: float, z: float) -> float:
    """距離 z の物体が行 y にあるときの予測視差 d = (y − y0(z))/ā（符号の合う枝・合焦行の間は 0）。"""
    dp = (y - (p.v_pos / z + p.u_pos)) / p.a_pos
    dn = (y - (p.v_neg / z + p.u_neg)) / p.a_neg
    return dp if dp > 0 else (dn if dn < 0 else 0.0)


raw = cv2.imread(str(config.CAPTURES_DIR / "b5_scene_01" / f"{INDEX:06d}.png"))
proc = preprocess(raw)
red = proc[:, :, 2]
edge = cv2.convertScaleAbs(cv2.Sobel(red, cv2.CV_32F, 1, 0)) >= config.EDGE_TH
col = edge[150:340].sum(axis=0)
print("columns with the most edge pixels (rows 150-339):",
      sorted((int(c), int(col[c])) for c in np.argsort(col)[::-1][:12]))
d = disparity_of_proc(proc, DispSettings())
z = np.load(config.OUTPUTS_DIR / "b5_scene_01" / f"{INDEX:06d}.npy")
print(f"frame {INDEX}; predicted for z = {Z_CYL:g} mm; band {config.INFER_BAND_LOW} < |d| < {config.INFER_BAND_HIGH}")
for name, (c0, c1) in EDGES.items():
    print(f"== {name} edge, columns {c0}-{c1 - 1}")
    print("rows      n_d  median_d  predicted_d  n_depth(after band)  median_depth_mm")
    for r0 in range(0, 352, 16):
        bd = d[r0:r0 + 16, c0:c1]
        bz = z[r0:r0 + 16, c0:c1]
        fd = bd[np.isfinite(bd)]
        fz = bz[np.isfinite(bz)]
        md = f"{np.median(fd):8.2f}" if fd.size else "     nan"
        mz = f"{np.median(fz):8.0f}" if fz.size else "     nan"
        print(f"{r0:3d}-{r0 + 15:3d}  {fd.size:4d}  {md}  {predicted_d(r0 + 7.5, Z_CYL):10.2f}  {fz.size:8d}  {mz}")
