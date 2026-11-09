# -*- coding: utf-8 -*-
"""既存の撮影画像で、照合窓（25×25）ごとの「横方向の情報の割合」と、開口形状ごとの窓の情報量を数える（読み取りのみ）。

目的: a5 は画素ごとのエッジの向きを数えたが、照合は窓で行うので、窓の中に少しでも横方向の変化があれば左右分割でも測れうる
      （physB のメモ §6 B）。そこで窓の構造テンソル T = mean_窓 [gx², gxgy; gxgy, gy²] から
        h = Txx / (Txx + Tyy)（窓の勾配のエネルギーのうち横方向の割合。0 なら横方向の変化が全く無い）
      を出し、さらに開口形状 D の Fisher 行列 S_D（a2_catalog.csv の光量と重心から再構成）について
        J_D = tr(S_D T)（スカラーのぼけ c に対する窓の情報量。光子雑音律速・共通のずれは nuisance・各チャネルの模様は同じ）
      を出す。現行の左右分割では J = S_xx·Txx。
      「現行の選別（R の |∂R/∂x|>20）が既に受け入れている窓の J の 10 パーセンタイル」を J0 として、
      |∇R|>20 の画素を中心とする窓のうち J_D ≥ J0 の割合を形状ごとに数える（情報量で見た有効画素の目安）。
入力: argv[1] = 開発リポ root（既定 C:/Users/t5424001/dev/tilted_optics_color_disparity_depth）、physA_out/a2_catalog.csv、
      physA_out/a10_families.csv（あれば。族 B の w=90°・180° を形状に加える。先に a10_families.py を実行）
出力: physA_out/a11_window_info.csv（セッション×形状）、a11_window_h.csv（セッション×h の分位）、a11_window_info.png、標準出力
前処理と画素の選び方: a5_edge_orientation.py と同じ（中央 1080×720 を切り出し INTER_AREA で 528×352・参照実装の走査範囲・
      Sobel 3×3）。窓は cv2.boxFilter（25×25・正規化あり・端は既定の反射）。
仮定・限界: 画像は現行リグ（θ=30° のアオリ・C/Y 左右）を通したもので、R は左半円の PSF でぼけている（G でも同じ量を出して比べる）。
      J_D は「各チャネルに同じ模様が同じ形のぼけで写る」とした Fisher の目安で、ぼけ形状の不一致（physB §7）や色は入らない。
      場面は屋内に偏る（a5 の docstring 参照）。
"""
import csv
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import a5_edge_orientation as a5e  # GROUPS・frames_of・preprocess・定数を共有（argv[1] を ROOT に使う）

OUT = os.path.join(os.path.dirname(HERE), "physA_out")
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
KEYS = ["cy_lr", "cy_lr_45", "rgb3", "cmy3", "ycg3", "quad", "wheel6", "ywcg4_v", "cy_m60", "cy_m90", "kaneko_g", "bando", "mca"]


def design_S():
    rows = list(csv.DictReader(open(os.path.join(OUT, "a2_catalog.csv"), encoding="utf-8")))
    out = {}
    for r in rows:
        if r["key"] not in KEYS:
            continue
        a = np.array([float(r["a_R"]), float(r["a_G"]), float(r["a_B"])])
        p = np.array([[float(r["pR_x"]), float(r["pR_y"])], [float(r["pG_x"]), float(r["pG_y"])],
                      [float(r["pB_x"]), float(r["pB_y"])]])
        w = np.where(a > 0.005, a, 0.0)
        pb = (w[:, None] * p).sum(0) / w.sum()
        dp = p - pb
        S = np.einsum("i,ij,ik->jk", w, dp, dp)
        out[r["key"]] = (r["label"], S)
    # a10_families.py の族 B（下に G 扇形）の代表 2 つを足す（a10_families.csv があれば）
    fp = os.path.join(OUT, "a10_families.csv")
    if os.path.exists(fp):
        for r in csv.DictReader(open(fp, encoding="utf-8")):
            if r["family"].startswith("B") and int(r["width_deg"]) in (90, 180):
                a = np.array([float(r["a_R"]), float(r["a_G"]), float(r["a_B"])])
                p = np.array([[float(r["pR_x"]), float(r["pR_y"])], [float(r["pG_x"]), float(r["pG_y"])],
                              [float(r["pB_x"]), float(r["pB_y"])]])
                pb = (a[:, None] * p).sum(0) / a.sum()
                dp = p - pb
                key = f"famB{int(r['width_deg'])}"
                out[key] = (f"C/Y＋下に G {int(r['width_deg'])}°", np.einsum("i,ij,ik->jk", a, dp, dp))
                if key not in KEYS:
                    KEYS.append(key)
    return out


def window_tensor(c):
    gx = cv2.Sobel(c, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(c, cv2.CV_32F, 0, 1)
    k = (a5e.K, a5e.K)
    Txx = cv2.boxFilter(gx * gx, cv2.CV_32F, k)
    Tyy = cv2.boxFilter(gy * gy, cv2.CV_32F, k)
    Txy = cv2.boxFilter(gx * gy, cv2.CV_32F, k)
    return gx, gy, Txx, Tyy, Txy


def main():
    Sd = design_S()
    rows, hrows = [], []
    Hq = [0.02, 0.05, 0.1, 0.2]
    pool_h = {}
    for grp, sessions in a5e.GROUPS.items():
        for ses in sessions:
            fr = a5e.frames_of(ses)
            if not fr:
                continue
            for ch in ("R", "G"):
                hs_any, hs_cur, Ts_any, J0s = [], [], [], []
                n_cur = n_any = 0
                for f in fr:
                    img = a5e.preprocess(cv2.imread(f, cv2.IMREAD_COLOR))
                    c = img[:, :, {"R": 2, "G": 1}[ch]]
                    m_i, m_ii, _, _, _ = a5e.analyze(img, ch)
                    gx, gy, Txx, Tyy, Txy = window_tensor(c)
                    tr = Txx + Tyy
                    h = np.where(tr > 0, Txx / np.maximum(tr, 1e-12), np.nan)
                    hs_any.append(h[m_ii])
                    hs_cur.append(h[m_i])
                    Ts_any.append(np.stack([Txx[m_ii], Tyy[m_ii], Txy[m_ii]], 1))
                    J0s.append(Txx[m_i])  # 現行の選別が受け入れた窓の Txx（S_xx を掛ける前）
                    n_cur += int(m_i.sum())
                    n_any += int(m_ii.sum())
                h_any = np.concatenate(hs_any)
                h_cur = np.concatenate(hs_cur)
                T = np.concatenate(Ts_any)
                Sxx_cur = Sd["cy_lr"][1][0, 0]
                J0 = Sxx_cur * np.percentile(np.concatenate(J0s), 10)
                hr = dict(group=grp, session=ses, channel=ch, n_any=n_any / len(fr), n_cur=n_cur / len(fr),
                          h_median_any=float(np.nanmedian(h_any)), h_median_cur=float(np.nanmedian(h_cur)))
                for q in Hq:
                    hr[f"frac_h_lt_{q}"] = float(np.nanmean(h_any < q))
                hrows.append(hr)
                if ch == "R":
                    rng = np.random.default_rng(0)
                    pick = rng.choice(h_any.size, size=min(20000, h_any.size), replace=False)
                    pool_h.setdefault(grp, []).append(h_any[pick])
                for key, (label, S) in Sd.items():
                    vals = []
                    for sgn in (1.0, -1.0):  # 画像の y 下向きと開口図の y 上向きの取り違えに備えて Sxy の符号の両方
                        J = S[0, 0] * T[:, 0] + S[1, 1] * T[:, 1] + 2 * sgn * S[0, 1] * T[:, 2]
                        e = J / np.maximum(T[:, 0] + T[:, 1], 1e-12)
                        vals.append((float(np.mean(J >= J0)), float(np.median(e)), float(np.percentile(e, 10))))
                    v = min(vals, key=lambda t: t[0])
                    rows.append(dict(group=grp, session=ses, channel=ch, key=key, label=label,
                                     frac_J_ge_J0=v[0], e_median=v[1], e_p10=v[2],
                                     eff_pixels=v[0] * n_any / len(fr), cur_pixels=n_cur / len(fr)))
    with open(os.path.join(OUT, "a11_window_h.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hrows[0].keys()))
        w.writeheader()
        w.writerows(hrows)
    with open(os.path.join(OUT, "a11_window_info.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # 要約（セッション等重み）
    print("窓の横方向の情報の割合 h = Txx/(Txx+Tyy)（|∇|>20 の画素を中心とする 25×25 窓・セッション等重み）")
    for grp in a5e.GROUPS:
        for ch in ("R", "G"):
            rr = [r for r in hrows if r["group"] == grp and r["channel"] == ch]
            if not rr:
                continue
            f = lambda k: np.mean([r[k] for r in rr])
            print(f"  {grp} {ch}: h の中央値 {f('h_median_any'):.3f}（現行の選別の窓 {f('h_median_cur'):.3f}）"
                  f" h<0.02 {f('frac_h_lt_0.02'):.3f} h<0.05 {f('frac_h_lt_0.05'):.3f} h<0.1 {f('frac_h_lt_0.1'):.3f} h<0.2 {f('frac_h_lt_0.2'):.3f}")
    print("\n情報量で見た有効画素の目安: |∇R|>20 の窓のうち J_D ≥ J0 の割合（J0 = 現行の選別が受け入れた窓の J の 10 パーセンタイル）"
          "\n  と、その画素数 / 現行の選別の画素数（R・セッション等重み）")
    for grp in a5e.GROUPS:
        rr = [r for r in rows if r["group"] == grp and r["channel"] == "R"]
        if not rr:
            continue
        parts = []
        for key in KEYS:
            sub = [r for r in rr if r["key"] == key]
            if not sub:
                continue
            fr_ = np.mean([r["frac_J_ge_J0"] for r in sub])
            gain = np.mean([r["eff_pixels"] / max(r["cur_pixels"], 1) for r in sub])
            parts.append(f"{sub[0]['label']} {fr_:.2f}（×{gain:.2f}）")
        print(f"  {grp}: " + "; ".join(parts))
    # 図: h の分布
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e87ba4"]
    bins = np.linspace(0, 1, 41)
    for (grp, lst), col in zip(pool_h.items(), pal):
        hh = np.concatenate(lst)
        hh = hh[np.isfinite(hh)]
        ax.hist(hh, bins, density=True, histtype="step", lw=2, color=col, label=grp)
    ax.set_xlabel("窓の横方向の情報の割合 h = Txx/(Txx+Tyy)（0 = 窓の中に横方向の変化が無い）")
    ax.set_ylabel("密度（R の |∇R|>20 の画素を中心とする 25×25 窓）")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "a11_window_info.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
