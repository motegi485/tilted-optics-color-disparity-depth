# エッジの向きと「基線方向の一次元探索」の感度の関係（幾何のみ・理想的な直線エッジ）
# 基線ベクトル b（向き beta）に対し、向き phi のエッジで実効的に使える基線は b*|sin(phi-beta)|。
# 法線方向の位置誤差 sigma_n が基線方向の視差誤差に化ける倍率は 1/|sin(phi-beta)|
# （Hirschmuller & Gehrig 2009 の e_d = e_e * tan(90deg - delta) と同じ形）。
import math
import sys

sys.stdout.reconfigure(encoding="utf-8")
B = 0.85  # 研究方針.md §3 の現行 C/Y 左右分割の R-B 重心間距離（開口半径 1）
angles = [0, 5, 10, 15, 20, 30, 45, 60, 75, 90]  # エッジの向き（水平から測る, deg）
print("edge_deg | 現行(基線0deg) 実効基線 | 誤差倍率 | 45deg回転 実効基線 | 誤差倍率")
for a in angles:
    s0 = abs(math.sin(math.radians(a - 0)))
    s45 = abs(math.sin(math.radians(a - 45)))
    f0 = "inf" if s0 < 1e-12 else f"{1/s0:.2f}"
    f45 = "inf" if s45 < 1e-12 else f"{1/s45:.2f}"
    print(f"{a:3d} | {B*s0:.3f} | {f0} | {B*s45:.3f} | {f45}")
# 一様分布のエッジ向きに対し、実効基線が 0.5*B 未満（誤差倍率 2 超）になる向きの割合
for beta in (0, 45):
    n = 18000
    bad = sum(1 for i in range(n) if abs(math.sin(math.radians(i * 180 / n - beta))) < 0.5)
    print(f"beta={beta}: |sin|<0.5 の向きの割合（一様分布） = {bad/n:.3f}")
