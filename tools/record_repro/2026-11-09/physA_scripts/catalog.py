# -*- coding: utf-8 -*-
"""開口形状のカタログ（既知の形と候補）。各設計は f(x, y) -> T (N,3)。

寸法の出典と仮定は DESIGNS の note に書く。研究方針.md §7 の定義（扇形 120°・角度は扇形の中心）に従う。
回折なし・ケラレなし・アオリなし（aperture_lib の仮定）。
"""
import numpy as np
from aperture_lib import (in_disk, in_sector, in_halfplane, in_convex_polygon, regular_polygon,
                          paint, rotate_design, FILTERS)


# ---- 研究方針.md §3 の 6 形状 ----
def cy_lr(x, y):
    """現行: 左 Y（R+G）・右 C（G+B）"""
    return paint(x, y, [(x < 0, "Y"), (x >= 0, "C")])


cy_lr_45 = rotate_design(cy_lr, 45.0)


def rgb3(x, y):
    return paint(x, y, [(in_sector(x, y, 210, 120), "R"), (in_sector(x, y, 90, 120), "G"),
                        (in_sector(x, y, 330, 120), "B")])


def cmy3(x, y):
    return paint(x, y, [(in_sector(x, y, 90, 120), "M"), (in_sector(x, y, 210, 120), "C"),
                        (in_sector(x, y, 330, 120), "Y")])


def ycg3(x, y):
    return paint(x, y, [(in_sector(x, y, 150, 120), "Y"), (in_sector(x, y, 30, 120), "C"),
                        (in_sector(x, y, 270, 120), "G")])


def quad_ycrb(x, y):
    """四分割: 左上 Y・右上 C・左下 R・右下 B（R=左半分・G=上半分・B=右半分）"""
    up = y >= 0
    return paint(x, y, [((x < 0) & up, "Y"), ((x >= 0) & up, "C"), ((x < 0) & ~up, "R"),
                        ((x >= 0) & ~up, "B")])


# ---- 先行研究の形（寸法は近似） ----
def bando(x, y):
    """[Bando2008] Fig.1(a) の写真からの近似: 45° 回した正方形 3 つ（辺 s=0.707）。
    R 中心 (−0.5,0)・B (0.5,0)・G (0,0.5)、G は R・B と辺を共有。全体が半径 1 に内接 [仮定]"""
    s = 1 / np.sqrt(2)
    rc = s / np.sqrt(2)  # 45° 回した正方形の外接半径 = s/√2
    lay = []
    for (cx, cy), f in [((-0.5, 0.0), "R"), ((0.5, 0.0), "B"), ((0.0, 0.5), "G")]:
        lay.append((in_convex_polygon(x, y, regular_polygon(cx, cy, rc, 4, 0.0)), f))
    return paint(x, y, lay)


def par_cmy_circles(x, y):
    """[Paramonov2016] Fig.3 左上（~40%）: 半径 1 に内接する等円 3 つ（r=0.464・中心距離 0.536）。Y 上・M 左下・C 右下"""
    r = 1.0 / (1.0 + 2.0 / np.sqrt(3.0))
    d = 1.0 - r
    lay = []
    for ang, f in [(90, "Y"), (210, "M"), (330, "C")]:
        t = np.deg2rad(ang)
        lay.append((in_disk(x, y, d * np.cos(t), d * np.sin(t), r), f))
    return paint(x, y, lay)


def par_cmy_sectors(x, y):
    """[Paramonov2016] Fig.3 上段 2 番目（~65%）: Y 上・M 左下・C 右下の 120° 扇形"""
    return paint(x, y, [(in_sector(x, y, 90, 120), "Y"), (in_sector(x, y, 210, 120), "M"),
                        (in_sector(x, y, 330, 120), "C")])


PAR_OPEN_S = 0.34  # [Paramonov2016] Fig.3 "CY with open area" の図から読んだ円の中心のずれ（三日月の最大幅 ≈0.34R）[仮定]


def par_cy_open(x, y):
    """[Paramonov2016] Fig.3 右上（~85%）の近似: 中心を ±s ずらした半径 1 の 2 円 A（+s）・B（−s）。
    A∩B=素通し、B のみ=Y（左の三日月）、A のみ=C（右）、どちらでもない上下=G（Y と C の重なり）。
    → R=円B∩開口、B=円A∩開口、G=全開口。s=0.34 で平均光量 0.857（図の ~85% と整合）[仮定]"""
    s = PAR_OPEN_S
    inA = in_disk(x, y, s, 0, 1.0)
    inB = in_disk(x, y, -s, 0, 1.0)
    return paint(x, y, [(inA & inB, "W"), (inB & ~inA, "Y"), (inA & ~inB, "C"), (~inA & ~inB, "G")])


def par_cmy_gradient(x, y):
    """[Paramonov2016] Fig.3 下段左（~75%）の近似: CMY 扇形の濃度を半径に比例させる（中心で素通し）[仮定]。
    濃度 = r（中心 0・縁 1）。M 扇形は G を r だけ遮る、など。平均光量 ≈ 1 − 2/9 = 0.78"""
    T = np.ones((x.size, 3))
    r = np.sqrt(x * x + y * y)
    for ang, f in [(90, "Y"), (210, "M"), (330, "C")]:
        m = in_sector(x, y, ang, 120)
        blocked = 1.0 - np.array(FILTERS[f])  # 遮るチャネル
        T[m] = 1.0 - r[m, None] * blocked[None, :]
    return paint(x, y, [(np.ones_like(x, bool), T)])


def par_cy_gradient(x, y):
    """[Paramonov2016] Fig.3 下段中（~85%）の近似: 左が Y・右が C で、濃度が中心 0 から縁へ |x| に比例 [仮定]。
    B の透過 = 1 − max(0,−x)、R の透過 = 1 − max(0,x)、G = 1。平均光量 ≈ 0.86"""
    T = np.ones((x.size, 3))
    T[:, 2] = 1.0 - np.clip(-x, 0, 1)
    T[:, 0] = 1.0 - np.clip(x, 0, 1)
    return paint(x, y, [(np.ones_like(x, bool), T)])


def par_hex(x, y):
    """[Paramonov2016] Fig.3 下段右（~50%）の近似: 正六角形 7 つ（中央 W、周りに C0°・B60°・M120°・R180°・Y240°・G300°）。
    六角形の外接半径 ρ=1/√7（全体が半径 1 に内接）、隣との中心距離 √3ρ、残りは不透明 [仮定]"""
    rho = 1 / np.sqrt(7)
    lay = [(in_convex_polygon(x, y, regular_polygon(0, 0, rho, 6, 30.0)), "W")]
    for ang, f in [(0, "C"), (60, "B"), (120, "M"), (180, "R"), (240, "Y"), (300, "G")]:
        t = np.deg2rad(ang)
        cx, cy = np.sqrt(3) * rho * np.cos(t), np.sqrt(3) * rho * np.sin(t)
        lay.append((in_convex_polygon(x, y, regular_polygon(cx, cy, rho, 6, 30.0)), f))
    return paint(x, y, lay)


KANEKO_RHO = 5.6 / 22.0  # F5.6 の開口に対する F22 の小開口の半径比 ≈ 0.255


def kaneko_g(x, y):
    """[Kaneko2025] 型: 左半円 R・右半円 B（F5.6）＋中央の小開口 G（F22, 半径比 0.255）。
    中央は G だけを通すと仮定（本文は「G チャネル用」「R・B を多く透過し G を強く減衰」）。左右は R・B フィルタ [仮定]"""
    c = in_disk(x, y, 0, 0, KANEKO_RHO)
    return paint(x, y, [((x < 0) & ~c, "R"), ((x >= 0) & ~c, "B"), (c, "G")])


def kaneko_w(x, y):
    """[Kaneko2025] 型の変種: 中央の小開口を素通し（W）とした場合 [仮定]"""
    c = in_disk(x, y, 0, 0, KANEKO_RHO)
    return paint(x, y, [((x < 0) & ~c, "R"), ((x >= 0) & ~c, "B"), (c, "W")])


def mca(x, y, r=0.3, d=0.6):
    """MCA 型（RGB の小円 3 つの三角配置）。半径 0.3・中心距離 0.6 は [仮定]（文献の寸法は未確認）。G 上・R 左下・B 右下"""
    lay = []
    for ang, f in [(90, "G"), (210, "R"), (330, "B")]:
        t = np.deg2rad(ang)
        lay.append((in_disk(x, y, d * np.cos(t), d * np.sin(t), r), f))
    return paint(x, y, lay)


# ---- 候補（素通しの追加・G の重心を外す最小の改変など） ----
def w_top_cy(x, y):
    """上半分を素通し（W）、下半分を左 Y・右 C に分ける"""
    up = y >= 0
    return paint(x, y, [(up, "W"), (~up & (x < 0), "Y"), (~up & (x >= 0), "C")])


def ycw3(x, y):
    """Y・C・W の 120° 扇形（Y 150°・C 30°・W 270°）"""
    return paint(x, y, [(in_sector(x, y, 150, 120), "Y"), (in_sector(x, y, 30, 120), "C"),
                        (in_sector(x, y, 270, 120), "W")])


def make_cy_m(width_deg):
    """現行の左右 Y/C の下側に中心角 width の M 扇形（R+B を通し G を遮る）を入れる"""
    def f(x, y):
        return paint(x, y, [(x < 0, "Y"), (x >= 0, "C"), (in_sector(x, y, 270, width_deg), "M")])
    return f


def wheel6(x, y):
    """色相環 6 分割（R 0°・Y 60°・G 120°・C 180°・B 240°・M 300° の 60° 扇形）。
    R・G・B が中心を通る半円を 120° ずつずらした形と同じ（各チャネル光量 1/2）"""
    lay = []
    for ang, f in [(0, "R"), (60, "Y"), (120, "G"), (180, "C"), (240, "B"), (300, "M")]:
        lay.append((in_sector(x, y, ang, 60), f))
    return paint(x, y, lay)


def ywcg4(x, y):
    """G 全開口の条件での重心の上限に当たる形（光量 2/3）: R=0° を向く半円・B=60° を向く半円・G=全開口。
    重なり（−30°〜90°）=W、R のみ（−90°〜−30°）=Y、B のみ（90°〜150°）=C、どちらでもない（150°〜270°）=G"""
    return paint(x, y, [(in_sector(x, y, 30, 120), "W"), (in_sector(x, y, 300, 60), "Y"),
                        (in_sector(x, y, 120, 60), "C"), (in_sector(x, y, 210, 120), "G")])


def ywcg4_v(x, y):
    """ywcg4 を回して、縦エッジ（法線が水平）の基線が最大になる向きにしたもの（R−B の差ベクトルが水平）"""
    return rotate_design(ywcg4, -120.0)(x, y)


DESIGNS = [
    # key, 関数, 表示名, 出典・仮定
    ("cy_lr", cy_lr, "現行 C/Y 左右", "研究方針.md §7"),
    ("cy_lr_45", cy_lr_45, "C/Y を 45° 回転", "研究方針.md §7"),
    ("rgb3", rgb3, "RGB 三扇形", "研究方針.md §7"),
    ("cmy3", cmy3, "CMY 三扇形", "研究方針.md §7"),
    ("ycg3", ycg3, "Y/C/G 三扇形", "研究方針.md §7"),
    ("quad", quad_ycrb, "四分割 Y/C/R/B", "研究方針.md §7"),
    ("bando", bando, "Bando 型 RGB 正方形 3 つ", "[Bando2008] Fig.1(a) から近似 [仮定]"),
    ("par_cmy_circ", par_cmy_circles, "Paramonov CMY 円 3 つ", "[Paramonov2016] Fig.3 [近似]"),
    ("par_cmy_sect", par_cmy_sectors, "Paramonov CMY 扇形", "[Paramonov2016] Fig.3"),
    ("par_cy_open", par_cy_open, "Paramonov CY＋素通し", "[Paramonov2016] Fig.3 から s=0.34 と読む [仮定]"),
    ("par_cmy_grad", par_cmy_gradient, "Paramonov CMY 濃度勾配", "[Paramonov2016] Fig.3 [近似・濃度∝半径は仮定]"),
    ("par_cy_grad", par_cy_gradient, "Paramonov CY 濃度勾配", "[Paramonov2016] Fig.3 [近似・濃度∝|x| は仮定]"),
    ("par_hex", par_hex, "Paramonov 六角 7 つ", "[Paramonov2016] Fig.3 [近似]"),
    ("kaneko_g", kaneko_g, "Kaneko 型（中央 G）", "[Kaneko2025] §2.2 [中央=G のみは仮定]"),
    ("kaneko_w", kaneko_w, "Kaneko 型（中央 W）", "[Kaneko2025] の変種 [仮定]"),
    ("mca", mca, "MCA 型 RGB 小円 3 つ", "寸法 r=0.3・d=0.6 は [仮定]"),
    ("w_top_cy", w_top_cy, "上半分 W・下 Y/C", "候補"),
    ("ycw3", ycw3, "Y/C/W 三扇形", "候補"),
    ("cy_m30", make_cy_m(30), "C/Y＋下に M 30°", "候補（G の重心を外す最小の改変）"),
    ("cy_m60", make_cy_m(60), "C/Y＋下に M 60°", "候補"),
    ("cy_m90", make_cy_m(90), "C/Y＋下に M 90°", "候補"),
    ("wheel6", wheel6, "色相環 6 分割", "候補（設計空間の探索で前線に出る形）"),
    ("ywcg4_v", ywcg4_v, "Y/W/C/G 4 分割（縦優先の向き）", "候補（G 全開口の上限の形）"),
]
