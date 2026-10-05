"""figstyle.py — 記録図の共通スタイル。

各 fig*.py から import する。図の幅は COLUMN_WIDTH_IN / TEXT_WIDTH_IN の 1 か所で変える
（既定は 1 段幅 3.5 in・2 段幅 7.16 in）。
色は既定パレット（明色）の先頭から順に使う。先頭 3 色は全ペアで色覚の検証済み、
4 色目は隣接ペアのみ検証済みなので、系列は色に加えてマーカーと線種でも区別する（白黒印刷でも読める）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                 # tools/record_repro/2026-10-05/figures → リポジトリ root
RECORDS = ROOT / "docs" / "records"
if str(ROOT) not in sys.path:          # config・core・calibration・realtime・tools を import するため
    sys.path.insert(0, str(ROOT))

COLUMN_WIDTH_IN = 3.5                  # 1 段幅
TEXT_WIDTH_IN = 7.16                   # 2 段幅

# 系列色（明色）と、系列ごとのマーカー・線種
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
LINESTYLES = ["-", "--", "-.", ":"]

INK = "#0b0b0b"          # 主要な文字
INK2 = "#52514e"         # 補助の文字
MUTED = "#898781"        # 参照線・目盛りの文字
GRID = "#e1e0d9"         # グリッド（細線・実線）
AXIS = "#c3c2b7"         # 軸線
SHADE = "#f0efec"        # 校正範囲などの背景帯
POINT_BG = "#d9d8d2"     # 背景に置く点群（帯の外の画素など）


def apply_style() -> None:
    """全図に共通の rcParams。文字は 8 pt 前後（1 段幅の図で本文と同程度）。"""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,              # 文字を埋め込む（Type 42）
    })


def panel_label(ax, text: str, desc: str = "") -> None:
    """パネル記号 (a)(b)… と短い見出しを、軸の左上の外側に左寄せの見出しとして置く。"""
    ax.set_title(f"{text} {desc}".strip(), loc="left", pad=3, fontsize=8, color=INK)


def save(fig, stem: str) -> list[Path]:
    """PDF（ベクタ・点群はラスタ化）と PNG（確認用・300 dpi）を書き出す。
    メタデータの日時を消して、同じ入力なら同じ出力になるようにする。"""
    out = []
    for ext, meta in (("pdf", {"CreationDate": None, "ModDate": None}), ("png", {"Software": None})):
        path = HERE / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02, metadata=meta)
        out.append(path)
    plt.close(fig)
    return out


def check(cond: bool, message: str) -> None:
    """自己照合。成り立たなければ理由を出して exit 1（図は書き出さない）。"""
    if not cond:
        print(f"[CHECK FAILED] {message}", file=sys.stderr)
        sys.exit(1)
    print(f"[ok] {message}")
