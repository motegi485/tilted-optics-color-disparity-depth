# -*- coding: utf-8 -*-
"""c1_decenter.py (critic, 2026-11-09)

目的: フィルタが絞り位置にない場合（光束のフィルタ面上の足跡が像高とともにずれる）や、
      手作業でフィルタを開口の中心からずらして貼った場合に、各チャネルの部分開口の
      重心（＝基線）がどれだけ変わるかを見積もる。両者は「フィルタ模様に対して
      円形の光束がずれる」という同じ幾何なので 1 本で扱う。
仮定: 理想フィルタ（0/1 透過）、幾何光学、光束の足跡は半径 1 の円（フィルタ模様の座標で
      中心が s·(cosψ, sinψ) にずれる）。フィルタ模様は十分大きい（足跡の外にも続く）。
      アオリによる足跡の楕円化・瞳の倍率・ケラレは入れない。
入力: なし。 出力: critic_out/c1_decenter.txt（UTF-8）
"""
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "critic_out"
OUT.mkdir(exist_ok=True)

N = 1001
g = np.linspace(-1.6, 1.6, N)
X, Y = np.meshgrid(g, g)
ANG = (np.degrees(np.arctan2(Y, X)) + 360) % 360


def sector(center, width):
    d = (ANG - center + 540) % 360 - 180
    return np.abs(d) <= width / 2


def filt_masks(kind):
    """フィルタ面上のチャネル透過（足跡で切る前の模様）。R,G,B の bool 配列。"""
    left = X < 0
    right = ~left
    if kind == "current":  # 左 Y（R+G）・右 C（G+B）
        R, G, B = left, np.ones_like(left), right
    elif kind == "rgb3":  # R=210°, G=90°, B=330°
        R, G, B = sector(210, 120), sector(90, 120), sector(330, 120)
    elif kind == "cmy3":  # M=90°, C=210°, Y=330° → R=M∪Y, G=C∪Y, B=M∪C
        M, C, Yy = sector(90, 120), sector(210, 120), sector(330, 120)
        R, G, B = M | Yy, C | Yy, M | C
    elif kind.startswith("famB"):  # 下（270°中心）に G 扇形
        w = float(kind[4:])
        gs = sector(270, w)
        R, G, B = left & ~gs, np.ones_like(left), right & ~gs
    elif kind.startswith("famA"):  # 下に M 扇形（G を遮る）
        w = float(kind[4:])
        ms = sector(270, w)
        R, G, B = left | ms, ~ms, right | ms
    else:
        raise ValueError(kind)
    return {"R": R, "G": G, "B": B}


def metrics(kind, s, psi_deg):
    cx, cy = s * np.cos(np.radians(psi_deg)), s * np.sin(np.radians(psi_deg))
    foot = (X - cx) ** 2 + (Y - cy) ** 2 <= 1.0
    m = filt_masks(kind)
    area_full = foot.sum()
    p = {}
    a = {}
    for k in "RGB":
        sel = m[k] & foot
        a[k] = sel.sum() / area_full
        # 足跡の中心を原点にした重心（像の上での変位は足跡中心基準の重心で決まる）
        p[k] = np.array([(X[sel] - cx).mean(), (Y[sel] - cy).mean()]) if sel.any() else np.array([np.nan, np.nan])
    P = np.array([p["R"], p["G"], p["B"]])
    vert = np.nanmax(P[:, 0]) - np.nanmin(P[:, 0])  # 縦エッジ（法線が水平）の基線
    horz = np.nanmax(P[:, 1]) - np.nanmin(P[:, 1])  # 横エッジの基線
    # 最悪の向き = 三角形の最小幅
    sides = [np.linalg.norm(P[i] - P[j]) for i, j in ((0, 1), (1, 2), (0, 2))]
    area2 = abs((P[1, 0] - P[0, 0]) * (P[2, 1] - P[0, 1]) - (P[2, 0] - P[0, 0]) * (P[1, 1] - P[0, 1]))
    worst = area2 / max(sides)
    rb = P[2] - P[0]
    rb_ang = np.degrees(np.arctan2(rb[1], rb[0]))
    return dict(aR=a["R"], aG=a["G"], aB=a["B"], vert=vert, horz=horz, worst=worst,
                rb_len=np.linalg.norm(rb), rb_ang=rb_ang, pG=P[1])


kinds = ["current", "rgb3", "cmy3", "famA90", "famB90", "famB180"]
lines = []
lines.append("# フィルタ模様に対する光束の足跡のずれ s（足跡半径 1 で正規化）と基線の変化")
lines.append("# 各行: s, 向き ψ の 12 方向（0〜330°）での最小〜最大。worst は三角形の最小幅、rb_ang は R→B ベクトルの向き [deg]")
for kind in kinds:
    base = metrics(kind, 0.0, 0.0)
    lines.append(f"\n== {kind}: s=0  aR/aG/aB={base['aR']:.3f}/{base['aG']:.3f}/{base['aB']:.3f}  "
                 f"vert={base['vert']:.3f} horz={base['horz']:.3f} worst={base['worst']:.3f} "
                 f"rb_ang={base['rb_ang']:.1f} pG=({base['pG'][0]:+.3f},{base['pG'][1]:+.3f})")
    for s in (0.05, 0.10, 0.20, 0.30):
        rs = [metrics(kind, s, psi) for psi in range(0, 360, 30)]
        def rng(key):
            v = np.array([r[key] for r in rs])
            return f"{v.min():.3f}〜{v.max():.3f}"
        def rel(key):
            v = np.array([r[key] for r in rs])
            b = base[key]
            if b < 1e-6:
                return "（基準 0）"
            return f"（基準比 {v.min()/b:.2f}〜{v.max()/b:.2f}）"
        angs = np.array([r["rb_ang"] for r in rs])
        dang = (angs - base["rb_ang"] + 180) % 360 - 180
        lines.append(f"  s={s:.2f}: vert {rng('vert')}{rel('vert')}  horz {rng('horz')}{rel('horz')}  "
                     f"worst {rng('worst')}{rel('worst')}  aR {rng('aR')}  R→B 向きの変化 {dang.min():+.1f}〜{dang.max():+.1f}°")

# 取り付け位置と s の目安（[仮定]: 入射瞳半径 8.0 mm＝90/5.6/2、像の隅までの画角の正接 ≈ 13.4/109、アオリと瞳倍率は無視）
lines.append("\n# 目安: フィルタが絞りから光軸方向に Δ 離れているとき、像の隅の光束の足跡のずれ s ≈ Δ·tanα / r")
r = 90 / 5.6 / 2
for tan_a, lab in ((13.4 / 109, "像の隅（対角 13.4 mm, v≈109 mm）"), (7.45 / 109, "上下端（7.45 mm）"), (11.15 / 109, "左右端（11.15 mm）")):
    vals = ", ".join(f"Δ={d} mm → s={d * tan_a / r:.3f}" for d in (2, 5, 10, 20, 30))
    lines.append(f"  {lab}: {vals}")
lines.append("# 手作業の貼りずれ: 開口半径 r≈8 mm（入射瞳。絞り位置での実寸は瞳倍率により異なる [未確認]）で 0.4 mm のずれ → s≈0.05、0.8 mm → s≈0.10")

txt = "\n".join(lines)
(OUT / "c1_decenter.txt").write_text(txt, encoding="utf-8", newline="\n")
print(txt)
