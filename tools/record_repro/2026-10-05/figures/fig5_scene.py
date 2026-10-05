"""fig5_scene.py — 通常シーン b5_scene_01 の入力・エッジ画素・深度（推論帯の前／後）の図。

入力（読み取りのみ・git 追跡外）: data/captures/b5_scene_01/<idx>.png（raw 1280×720）と
outputs/b5_scene_01/<idx>.npy（runner が保存した本番の深度・推論帯適用後）、outputs/b5_scene_01/meta.json。
図: (a) 入力フレーム (b) 視差を探す縦エッジの画素（赤チャンネルの
水平 Sobel ≥ EDGE_TH・core/disparity と同じ手順）(c) 推論帯の前（gate1 のみ）の深度 (d) 推論帯の後の深度＝本番出力。
(c)(d) は元画像を薄く敷いた上に深度を viridis（明るさが単調に増え、色覚の違いや白黒印刷でも順序が読める）で描く。
点は太らせない（密度をそのまま見せる）。表示レンジ 1000〜4000 mm、範囲外は両端の色（件数を出力）。
保存の *_viz.png（DEPTH_MAX=50000 の反転 JET）は使わない。npy は変更しない。
自己照合: meta.json の設定と係数が現行と一致／raw を本番の前処理と process_frame で処理し直した深度が保存 npy と
ビット一致／(d) を解析経路（disparity_of_proc → apply_gates → depth_of）で作った値も保存 npy とビット一致
（同じ経路で作る (c) が本番と同じ計算であることの確認）／(b) のエッジ画素数が checks.bench_disparity と同じ数え方で一致。

実行（リポジトリ root）: uv run python tools/record_repro/2026-10-05/figures/fig5_scene.py
出力: project_knowledge/records/2026-10-05/figures/ の fig5_scene.pdf / .png
"""
from __future__ import annotations

import json

import cv2
import numpy as np

import figstyle as fs
import config
from checks.bench_disparity import _edge_pixel_count
from core.params import load_params
from realtime.camera import preprocess
from realtime.pipeline import process_frame
from tools.analysis_common import DispSettings, apply_gates, depth_of, disparity_of_proc

SESSION = "b5_scene_01"
INDEX = 10                         # 暖機 10 フレームを除いた最初のフレーム（集計の --skip-first 10 と同じ）
VMIN, VMAX = 1000.0, 4000.0        # 対象距離レンジ
CMAP = "viridis"
EDGE_COLOR = np.array([0xEB, 0x68, 0x34]) / 255.0   # figstyle.SERIES[1]（橙）


def same_depth(a: np.ndarray, b: np.ndarray) -> bool:
    """NaN 位置と有限値がビット単位で一致するか。"""
    return (a.dtype == b.dtype and a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b))
            and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]))


def main() -> None:
    raw_path = config.CAPTURES_DIR / SESSION / f"{INDEX:06d}.png"
    npy_path = config.OUTPUTS_DIR / SESSION / f"{INDEX:06d}.npy"
    meta = json.loads((config.OUTPUTS_DIR / SESSION / "meta.json").read_text(encoding="utf-8"))

    # --- 自己照合: 設定・係数・深度の再現 ---
    cur = {"IMAGE_SIZE": list(config.IMAGE_SIZE), "KERNEL_SIZE": config.KERNEL_SIZE,
           "SEARCH_RANGE_X": config.SEARCH_RANGE_X, "EDGE_TH": config.EDGE_TH, "DISP_MAX": config.DISP_MAX,
           "INFER_BAND_LOW": config.INFER_BAND_LOW, "INFER_BAND_HIGH": config.INFER_BAND_HIGH}
    fs.check(meta["config"] == cur, "meta.json の config 7 キーが現行 config と一致")
    params = load_params(config.PARAMS_PATH)
    fs.check(all(getattr(params, k) == float(v) for k, v in meta["params"].items()),
             "meta.json の 8 係数が live の係数（load_params）と一致")
    raw = cv2.imread(str(raw_path))
    fs.check(raw is not None and raw.shape == (720, 1280, 3), f"raw {raw_path.name} を読めて 1280×720")
    proc = preprocess(raw)
    saved = np.load(str(npy_path))
    fs.check(same_depth(saved, process_frame(proc, params)), f"frame {INDEX}: process_frame の再計算が保存 npy とビット一致")
    disp = disparity_of_proc(proc, DispSettings())
    band = (config.INFER_BAND_LOW, config.INFER_BAND_HIGH)
    z_on = depth_of(apply_gates(disp, disp_max=config.DISP_MAX, band=band), params)
    z_off = depth_of(apply_gates(disp, disp_max=config.DISP_MAX, band=None), params)
    fs.check(same_depth(saved, z_on), "解析経路で作った推論帯の後の深度が保存 npy とビット一致")
    red = proc[:, :, 2]
    sobel = cv2.convertScaleAbs(cv2.Sobel(red, cv2.CV_32F, 1, 0))
    _, edge_t = cv2.threshold(sobel, config.EDGE_TH, 255, cv2.THRESH_TOZERO)
    edge = edge_t > 0
    fs.check(int(edge.sum()) == _edge_pixel_count(red), f"エッジ画素 {int(edge.sum())} が bench_disparity の数え方と一致")

    n = saved.size
    shown = {}
    for name, z in (("before band", z_off), ("after band", z_on)):
        v = np.isfinite(z) & (z > config.DEPTH_MIN) & (z < config.DEPTH_MAX)   # gate2（集計と同じ 0 < z < 50 m）
        shown[name] = v
        print(f"{name}: 有効 {int(v.sum())}（{v.mean():.2%}）・{VMIN:g} mm 未満 {int((z[v] < VMIN).sum())}・"
              f"{VMAX:g} mm 超 {int((z[v] > VMAX).sum())}（両端の色で表示）・gate2 で除外 {int(np.isfinite(z).sum() - v.sum())}")
    print(f"エッジ画素 {int(edge.sum())}（{edge.mean():.2%}）・視差あり {int(np.isfinite(disp).sum())}・全 {n}")
    low_part = saved[301:, :][np.isfinite(saved[301:, :])]
    print(f"処理画像の 301 行目以下の有効画素 {low_part.size}・深度 p5 {np.percentile(low_part, 5):.0f}・"
          f"中央値 {np.median(low_part):.0f}・p95 {np.percentile(low_part, 95):.0f} mm"
          f"（全フレームの分布は ../scene_bottom_check.py）")

    # --- 描画 ---
    fs.apply_style()
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
    light = 0.45 + 0.55 * gray                      # 元画像を明るく薄めた背景（深度の点を目立たせる）
    fig = fs.plt.figure(figsize=(fs.TEXT_WIDTH_IN, 5.05))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.05], hspace=0.22, wspace=0.04)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[2, :])
    h, w = saved.shape

    ax_a.imshow(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB), interpolation="nearest")
    fs.panel_label(ax_a, "(a)", f"Input frame ({w}×{h})")

    rgb = np.dstack([light] * 3)
    rgb[edge] = EDGE_COLOR
    ax_b.imshow(rgb, interpolation="nearest")
    fs.panel_label(ax_b, "(b)", f"Vertical-edge pixels ({edge.mean():.1%})")

    for ax, z, key, label, desc in ((ax_c, z_off, "before band", "(c)", "Depth before inference band"),
                                    (ax_d, z_on, "after band", "(d)", "Depth output, after inference band")):
        ax.imshow(light, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        im = ax.imshow(np.ma.masked_where(~shown[key], z), cmap=CMAP, vmin=VMIN, vmax=VMAX,
                       interpolation="nearest")
        fs.panel_label(ax, label, f"{desc} ({shown[key].mean():.1%})")
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(fs.AXIS)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("Depth [mm]")
    cb.set_ticks(range(1000, 4001, 500))
    cb.outline.set_edgecolor(fs.AXIS)
    for p in fs.save(fig, "fig5_scene"):
        print(f"written: {p}")


if __name__ == "__main__":
    main()
