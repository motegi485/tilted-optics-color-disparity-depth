"""fig2_fig3_calibration.py — 校正の (d, y) 散布図と RANSAC 直線、切片 y₀(z) と参照式 y₀ = v/z + u のフィットの図。

入力（読み取りのみ）: live 校正の入力 data/calib_imgs/<距離>.png（10 距離・528×352・git 追跡外）、
project_knowledge/records/2026-09-18/params_final/params.csv・fit_result.csv（live の写し）、params/params.csv・fit_result.csv（live）。
処理は calibration/calibrate.py と同じ関数（calcDisparityByZNCC → extract_sign_points → ransac_fit_line）を
現行 config（K25・S41・T20・校正帯 (0.45, 7.5)・RANSAC th 8・seed 0）で呼ぶ。
自己照合: 10 距離すべてで再計算した傾き・切片・インライヤ率・点数が params_final/params.csv と一致し、
fit_result の係数による切片の残差 RMS が記録の値と一致すること。

実行（リポジトリ root）: uv run python tools/record_repro/2026-10-05/figures/fig2_fig3_calibration.py
出力: project_knowledge/records/2026-10-05/figures/ の fig2_calibration_scatter.pdf / .png と fig3_reference_fit.pdf / .png
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

import figstyle as fs
import config
from calibration.calibrate import extract_sign_points, ransac_fit_line
from calibration.fit_params import model
from core.disparity import calcDisparityByZNCC

FINAL = fs.RECORDS / "2026-09-18" / "params_final"
SCATTER_DISTANCES = (1400, 3000)        # 散布図に描く 2 距離


def calibrate_one(z: int) -> dict:
    """1 距離分を calibrate.py と同じ手順で計算する（I/O は画像の読み込みだけ）。"""
    image = cv2.imread(str(config.CALIB_IMGS_DIR / f"{z}.png"))
    fs.check(image is not None and image.shape[:2] == (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]),
             f"{z} mm: 校正画像を読めて 528×352")
    image = cv2.resize(image, config.IMAGE_SIZE)
    blue, _, red = cv2.split(image)
    disp = calcDisparityByZNCC(red, blue, config.KERNEL_SIZE, config.SEARCH_RANGE_X, config.EDGE_TH)
    low, high = config.DISP_PLOT_LOW, config.DISP_PLOT_HEIGHT
    out = {"depth": z, "disp": disp}
    for sign, name in ((+1, "pos"), (-1, "neg")):
        d, y = extract_sign_points(disp, sign, low, high)
        a, b, mask = ransac_fit_line(d, y, config.RANSAC_RESIDUAL_TH, config.RANSAC_RANDOM_STATE)
        out.update({f"a_{name}": a, f"zero_disp_{name}": b, f"inlier_{name}": float(mask.mean()),
                    f"n_{name}": len(d), f"d_{name}": d, f"y_{name}": y, f"mask_{name}": mask})
    return out


def main() -> None:
    # --- 前提: 現行 config が live 校正の設定と同じ ---
    fs.check((config.KERNEL_SIZE, config.SEARCH_RANGE_X, config.EDGE_TH) == (25, 41, 20)
             and (config.DISP_PLOT_LOW, config.DISP_PLOT_HEIGHT) == (0.45, 7.5)
             and (config.RANSAC_RESIDUAL_TH, config.RANSAC_RANDOM_STATE) == (8.0, 0),
             "config = K25・S41・T20・校正帯 (0.45, 7.5)・th 8・seed 0")
    final = pd.read_csv(FINAL / "params.csv").sort_values("depth").reset_index(drop=True)
    live = pd.read_csv(config.PARAMS_CSV_PATH).sort_values("depth").reset_index(drop=True)
    fs.check(final.equals(live), "params_final/params.csv = live params/params.csv")
    fit_final = pd.read_csv(FINAL / "fit_result.csv").set_index("param")["value"]
    fit_live = pd.read_csv(config.PARAMS_PATH).set_index("param")["value"]
    fs.check(fit_final.equals(fit_live), "params_final/fit_result.csv = live params/fit_result.csv")

    # --- 自己照合: 10 距離を再計算して params.csv と一致 ---
    results = {}
    for _, row in final.iterrows():
        z = int(row.depth)
        r = calibrate_one(z)
        results[z] = r
        same = all(abs(r[k] - row[k]) <= 1e-9 * max(1.0, abs(row[k]))
                   for k in ("zero_disp_pos", "zero_disp_neg", "a_pos", "a_neg", "inlier_pos", "inlier_neg"))
        fs.check(same and r["n_pos"] == row.n_pos and r["n_neg"] == row.n_neg,
                 f"{z} mm: 再計算の切片・傾き・インライヤ率・点数が params.csv と一致")
    rms = {}
    for name in ("pos", "neg"):
        res = final[f"zero_disp_{name}"] - model(final.depth, fit_final[f"v_{name}"], fit_final[f"u_{name}"])
        rms[name] = float(np.sqrt(np.mean(res ** 2)))
    fs.check(round(rms["pos"], 4) == 1.0843 and round(rms["neg"], 4) == 1.9188,
             f"切片の残差 RMS pos {rms['pos']:.4f}・neg {rms['neg']:.4f} px（記録票 §8.1: 1.0843・1.9188）")

    fs.apply_style()
    h_px = config.IMAGE_SIZE[1]
    low, high = config.DISP_PLOT_LOW, config.DISP_PLOT_HEIGHT

    # --- (d, y) 散布図と RANSAC 直線 ---
    fig, axes = fs.plt.subplots(1, 2, sharey=True, figsize=(fs.COLUMN_WIDTH_IN, 2.75),
                                gridspec_kw={"wspace": 0.12})
    for ax, z, label in zip(axes, SCATTER_DISTANCES, ("(a)", "(b)")):
        r = results[z]
        disp = r["disp"]
        rows, _ = np.nonzero(np.isfinite(disp))
        ax.axvspan(-low, low, color=fs.SHADE, lw=0, zorder=0)
        ax.scatter(disp[np.isfinite(disp)], rows, s=0.15, c=fs.POINT_BG, lw=0, rasterized=True, zorder=1,
                   label="outside calibration band")
        for i, name in enumerate(("pos", "neg")):
            d, y, m = r[f"d_{name}"], r[f"y_{name}"], r[f"mask_{name}"]
            ax.scatter(d[~m], y[~m], s=0.15, c=fs.MUTED, lw=0, rasterized=True, zorder=2,
                       label="RANSAC outlier" if i == 0 else None)
            ax.scatter(d[m], y[m], s=0.15, c=fs.SERIES[i], lw=0, rasterized=True, zorder=3,
                       label=f"inlier, {name} branch" + (" (d > 0)" if name == "pos" else " (d < 0)"))
            a, b = r[f"a_{name}"], r[f"zero_disp_{name}"]
            sgn = 1 if name == "pos" else -1
            dd = np.array([low, high]) * sgn
            ax.plot(dd, a * dd + b, color=fs.INK, lw=0.9, zorder=4)
            d0 = np.array([0.0, low]) * sgn
            ax.plot(d0, a * d0 + b, color=fs.INK, lw=0.9, ls=":", zorder=4)   # 切片までの外挿
            ax.plot(0, b, ls="none", marker=fs.MARKERS[i], ms=4, mfc="white", mec=fs.INK, mew=0.8, zorder=5)
        ax.set_xlim(-9.5, 9.5)
        ax.set_ylim(h_px, 0)                       # 行 0 を上に（画像と同じ向き）
        ax.set_xticks([-8, -4, 0, 4, 8])
        ax.set_xlabel("Disparity d [px]")
        fs.panel_label(ax, label, f"{z} mm")
    axes[0].set_ylabel("Image row y [px]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0), markerscale=18,
               handletextpad=0.3, columnspacing=1.0)
    for p in fs.save(fig, "fig2_calibration_scatter"):
        print(f"written: {p}")

    # --- 切片 y₀(z) と参照式 ---
    fig, ax = fs.plt.subplots(figsize=(fs.COLUMN_WIDTH_IN, 2.35))
    for row_line, txt in ((0, "image top (row 0)"), (h_px - 1, f"image bottom (row {h_px - 1})")):
        ax.axhline(row_line, color=fs.MUTED, lw=0.8, ls="--", zorder=1)
        ax.text(4050, row_line + (12 if row_line == 0 else -6), txt, ha="right",
                va="top" if row_line == 0 else "bottom", fontsize=6.5, color=fs.INK2)
    ax.axvspan(final.depth.min(), final.depth.max(), color=fs.SHADE, lw=0, zorder=0)
    z_all = np.linspace(1000, 4000, 301)
    z_in = np.linspace(final.depth.min(), final.depth.max(), 181)
    for i, name in enumerate(("pos", "neg")):
        v, u = fit_final[f"v_{name}"], fit_final[f"u_{name}"]
        ax.plot(z_all, model(z_all, v, u), color=fs.SERIES[i], lw=0.9, ls=":", zorder=2)
        ax.plot(z_in, model(z_in, v, u), color=fs.SERIES[i], lw=1.4, ls=fs.LINESTYLES[i], zorder=3,
                label=f"{name} branch: $y_0 = v/z + u$ (RMS {rms[name]:.2f} px)")
        ax.plot(final.depth, final[f"zero_disp_{name}"], ls="none", marker=fs.MARKERS[i], color=fs.SERIES[i],
                mfc="white", mew=1.0, ms=4, zorder=4)
    ax.set_xlim(950, 4050)
    ax.set_ylim(420, -15)
    ax.set_xticks(range(1000, 4001, 500))
    ax.set_xlabel("Calibration distance z [mm]")
    ax.set_ylabel("Intercept $y_0$ [row]")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.2), handlelength=2.2)
    for p in fs.save(fig, "fig3_reference_fit"):
        print(f"written: {p}")


if __name__ == "__main__":
    main()
