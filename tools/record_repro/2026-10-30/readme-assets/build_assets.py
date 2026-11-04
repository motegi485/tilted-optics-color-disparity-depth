"""README掲載図を保存済みの画像・深度・校正CSVから描く（入力は読み取り専用）。

実行: リポジトリ直下で uv run --offline --no-sync python tools/record_repro/2026-10-30/readme-assets/build_assets.py
元画像とNPYはGit管理外。校正散布図は保存画像から再現し、project_knowledge/records の校正CSVと照合する。
保存データを現行関数で再計算して表示する。校正係数や入力データには書き込まない。
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Wedge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from realtime.camera import preprocess
import config
from core.disparity import calcDisparityByZNCC
from core.params import load_params
from realtime.pipeline import process_frame
from calibration.calibrate import extract_sign_points, ransac_fit_line, _plot_branch

ASSETS = ROOT / "assets/readme"
RAW = ROOT / "data/captures/b5_scene_01/000010.png"
DEPTH = ROOT / "outputs/b5_scene_01/000010.npy"
META = ROOT / "outputs/b5_scene_01/meta.json"
CALIB = ROOT / "project_knowledge/records/2026-09-18/params_final/params.csv"
RECORD_DIR = ROOT / "project_knowledge/records/2026-10-30/readme-assets"
CALIB_IMAGE = ROOT / "data/calib_imgs/2400.png"
CALIB_RAW = ROOT / "data/calib_raw/2400.png"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    inputs = [RAW, DEPTH, META, CALIB, CALIB_IMAGE, CALIB_RAW, ROOT / "params/fit_result.csv"]
    before = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in inputs}
    plt.rcParams.update({"font.family": "Yu Gothic", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#243344", "text.color": "#243344",
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    ASSETS.mkdir(parents=True, exist_ok=True)
    raw = cv2.imread(str(RAW))
    if raw is None or raw.shape != (720, 1280, 3):
        raise ValueError("デモ入力の形状が想定と異なる")
    meta = json.loads(META.read_text(encoding="utf-8"))
    if meta["config"]["IMAGE_SIZE"] != [528, 352]:
        raise ValueError("保存セッションの処理解像度が異なる")
    rgb = cv2.cvtColor(preprocess(raw), cv2.COLOR_BGR2RGB)
    depth = np.load(DEPTH, allow_pickle=False)
    if depth.shape != rgb.shape[:2] or depth.dtype != np.float32:
        raise ValueError("深度マップの形状・型が想定と異なる")
    # 全有限画素をその位置に描く。表示範囲外の有限値は両端色で示す。
    valid = np.isfinite(depth)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float) / 255
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    fig.subplots_adjust(left=.015, right=.92, top=.88, bottom=.025, wspace=.035)
    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title("入力画像", pad=12)
    axes[1].imshow(.45 + .55 * gray, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    im = axes[1].imshow(np.ma.masked_invalid(depth / 1000), cmap="viridis",
                        vmin=1, vmax=4, interpolation="nearest")
    axes[1].set_title("距離マップ", pad=12)
    for ax in axes:
        ax.axis("off")
    cax = fig.add_axes([.935, .10, .014, .70])
    cb = fig.colorbar(im, cax=cax, orientation="vertical", extend="both")
    cb.set_label("距離 [m]", labelpad=3)
    cb.set_ticks([1, 2, 3, 4])
    fig.savefig(ASSETS / "readme-demo.png", dpi=160)
    plt.close(fig)

    with CALIB.open(encoding="utf-8", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r["depth"]))
    # 実写によるぼけ・色ずれの説明。加工は位置を記録した切出しと表示のみ。
    calib_raw = cv2.cvtColor(cv2.imread(str(CALIB_RAW)), cv2.COLOR_BGR2RGB)
    if calib_raw.shape != (720, 1280, 3):
        raise ValueError("校正rawの形状が想定と異なる")
    photo = calib_raw[:, 100:1180]
    rois = {"上部": [600, 120, 60, 60], "下部": [800, 370, 60, 60]}
    fig, ax = plt.subplots(figsize=(8.4, 5.7))
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    ax.imshow(photo, interpolation="nearest")
    for label, (x, y, w, h) in rois.items():
        ax.add_patch(Rectangle((x-100, y), w, h, fill=False, edgecolor="white", linewidth=1.5))
        ax.text(x-100+w+12, y+30, label, color="white", va="center",
                bbox={"facecolor":"#243344", "alpha":.8, "edgecolor":"none", "pad":3})
    ax.axis("off")
    fig.savefig(ASSETS / "readme-optics-photo.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), gridspec_kw={"wspace":.16})
    fig.subplots_adjust(left=.02, right=.98, top=.85, bottom=.04)
    axs[0].add_patch(Wedge((0, 0), 1, 90, 270, facecolor="#76d8e4", edgecolor="#536273"))
    axs[0].add_patch(Wedge((0, 0), 1, -90, 90, facecolor="#f5d765", edgecolor="#536273"))
    axs[0].text(-.5, 0, "C", ha="center", va="center", fontsize=20)
    axs[0].text(.5, 0, "Y", ha="center", va="center", fontsize=20)
    axs[0].set_xlim(-1.15, 1.15); axs[0].set_ylim(-1.15, 1.15); axs[0].set_aspect("equal")
    axs[0].set_title("分割開口（模式図）")
    for ax, (label, (x,y,w,h)) in zip(axs[1:], rois.items()):
        ax.imshow(calib_raw[y:y+h, x:x+w], interpolation="nearest")
        ax.set_title(label + "の拡大")
    for ax in axs: ax.axis("off")
    fig.savefig(ASSETS / "readme-color-shift.png", dpi=160)
    plt.close(fig)

    # 同じ実写フレームの処理途中を図示。最後は保存NPYとの一致を確認。
    proc = preprocess(raw)
    blue, _, red = cv2.split(proc)
    disparity = calcDisparityByZNCC(red, blue, config.KERNEL_SIZE, config.SEARCH_RANGE_X, config.EDGE_TH)
    sobel = cv2.convertScaleAbs(cv2.Sobel(red, cv2.CV_32F, 1, 0))
    edge = sobel > config.EDGE_TH
    calc = process_frame(proc, load_params(ROOT / "params/fit_result.csv"))
    pipeline_equal = np.array_equal(np.isnan(calc), np.isnan(depth)) and np.array_equal(calc[valid], depth[valid])
    if not pipeline_equal: raise ValueError("掲載フレームの保存深度と現行処理が一致しない")
    fig, axs = plt.subplots(1, 4, figsize=(15.5, 3.05))
    fig.subplots_adjust(left=.006, right=.97, top=.84, bottom=.06, wspace=.24)
    axs[0].imshow(rgb, interpolation="nearest")
    axs[1].imshow(edge, cmap="gray", interpolation="nearest")
    di = axs[2].imshow(np.ma.masked_invalid(disparity), cmap="coolwarm", vmin=-5, vmax=5, interpolation="nearest")
    de = axs[3].imshow(np.ma.masked_invalid(depth/1000), cmap="viridis", vmin=1, vmax=4, interpolation="nearest")
    for ax, label in zip(axs, ["01 入力", "02 縦エッジ", "03 色視差", "04 距離"]):
        ax.set_title(label); ax.axis("off")
    for ax, im, ticks, unit in [(axs[2], di, [-5,0,5], "px"), (axs[3], de, [1,2,3,4], "m")]:
        bar = fig.colorbar(im, ax=ax, fraction=.035, pad=.025, extend="both")
        bar.set_ticks(ticks); bar.ax.set_title(unit, fontsize=10); bar.ax.tick_params(labelsize=9)
    fig.savefig(ASSETS / "readme-pipeline.png", dpi=160)
    plt.close(fig)

    # 校正プログラムの表示関数をそのまま使用。main()は呼ばずparams/への書込みを避ける。
    calib_img = cv2.imread(str(CALIB_IMAGE))
    blue, _, red = cv2.split(calib_img)
    disp = calcDisparityByZNCC(red, blue, config.KERNEL_SIZE, config.SEARCH_RANGE_X, config.EDGE_TH)
    expected = next(r for r in rows if float(r["depth"]) == 2400)
    calibration_match = {}
    with plt.rc_context(plt.rcParamsDefault):
        fig, ax = plt.subplots()
        for sign, name, color in [(1,"pos","tab:blue"),(-1,"neg","tab:orange")]:
            d,y = extract_sign_points(disp, sign, config.DISP_PLOT_LOW, config.DISP_PLOT_HEIGHT)
            a,b,mask = ransac_fit_line(d,y,config.RANSAC_RESIDUAL_TH,config.RANSAC_RANDOM_STATE)
            calibration_match[name] = bool(np.isclose(a,float(expected[f"a_{name}"]),rtol=1e-9) and
                np.isclose(b,float(expected[f"zero_disp_{name}"]),rtol=1e-9) and
                np.isclose(mask.mean(),float(expected[f"inlier_{name}"]),rtol=1e-9) and
                len(d)==int(expected[f"n_{name}"]))
            if not calibration_match[name]: raise ValueError("校正再計算が保存CSVと不一致")
            _plot_branch(ax,d,y,mask,a,b,calib_img.shape[0],color,f"{name} ({mask.mean()*100:.0f}%)")
        ax.set_xlim(-config.DISP_PLOT_HEIGHT-1,config.DISP_PLOT_HEIGHT+1)
        ax.set_ylim(0,calib_img.shape[0]); ax.set_xlabel("disparity"); ax.set_ylabel("row (y)")
        ax.set_title("depth = 2400 mm"); ax.legend()
        fig.savefig(ASSETS / "readme-calibration-scatter.png", dpi=170)
        plt.close(fig)

    after = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in inputs}
    if before != after:
        raise RuntimeError("入力ファイルが変更された")
    manifest = {"inputs_sha256": before, "demo_session": "b5_scene_01", "frame": 10,
                "depth_shape": list(depth.shape), "finite_pixels": int(valid.sum()),
                "color_range_m": [1, 4], "finite_below_range": int((depth[valid] < 1000).sum()),
                "finite_above_range": int((depth[valid] > 4000).sum()),
                "calibration_scatter_matches_saved_csv": calibration_match,
                "pipeline_matches_saved_depth": pipeline_equal, "color_shift_rois_raw_xywh": rois,
                "outputs_sha256": {p.name: digest(p) for p in
                    [ASSETS / n for n in ["readme-demo.png", "readme-optics-photo.png",
                     "readme-color-shift.png", "readme-pipeline.png", "readme-calibration-scatter.png"]]}}
    (RECORD_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"inputs_unchanged": True, "finite_pixels": int(valid.sum()),
                      "calibration_scatter_matches_saved_csv": calibration_match, "pipeline_matches_saved_depth": pipeline_equal}, ensure_ascii=False))


if __name__ == "__main__":
    main()
