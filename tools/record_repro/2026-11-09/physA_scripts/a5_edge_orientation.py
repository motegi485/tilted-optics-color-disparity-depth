# -*- coding: utf-8 -*-
"""既存の撮影画像で、エッジの向きの分布と、左右分割が捨てているエッジの割合を数える（読み取りのみ）。

入力: argv[1] = 開発リポ root（既定 C:/Users/t5424001/dev/tilted_optics_color_disparity_depth）
      data/captures/<session>/*.png（1280×720 の raw）、data/test/2300.JPG
出力: physA_out/a5_edge_stats.csv（セッションごと）、a5_edge_hist.csv（角度ヒストグラム）、a5_edge_orientation.png、
      a5_design_scene_weighted.csv（カタログ形状の基線を場面のエッジ向きで平均した値。a2_profiles.npy が必要）
前処理（開発リポ tools/analysis_common.py の preprocess_to と同じ）: 1280×720 の中央 1080×720 を切り出し、INTER_AREA で 528×352。
      2300.JPG は checks/verify_pipeline_equivalence.py と同じく cv2.resize（既定補間）で 528×352。
エッジの数え方:
  (i)  現行の選別（core/_disparity_reference.py）: convertScaleAbs(Sobel_x(R)) を THRESH_TOZERO(20) → 値 > 20 の画素
  (ii) 向きを問わない: |∇R| = hypot(Sobel_x, Sobel_y) を同じ丸め（四捨五入）で > 20 の画素
  どちらも参照実装の走査範囲（y∈[k, H−k)、x∈[s+k+1, W−s−k−1)、K=25→k=12、S=41→s=20）に限る。
  (iii) (ii) の画素の法線と水平のなす角 φ=atan2(|gy|,|gx|)（0°=縦エッジ、90°=横エッジ）。左右分割の基線は cosφ に比例。
  同じことを G でも数える（R は半開口のぼけで横に細い PSF になり縦エッジを強く見せる偏りがありうるので、その確認）。
仮定・限界: 撮影はすべて現行リグ（θ=30° のアオリ・左右 C/Y）を通した画像で、ぼけと色フィルタの影響を含む。
  場面は屋内（車の背面の印刷物・ブラインド・机・椅子など）に偏り、屋外の塀・壁は含まない。7 月・8 月のセッションの撮影条件は記録票に見当たらない。
"""
import csv
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/t5424001/dev/tilted_optics_color_disparity_depth"
OUT = os.path.join(os.path.dirname(HERE), "physA_out")
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

IMAGE_SIZE = (528, 352)  # config.IMAGE_SIZE（確認 2026-11-09）
CROP_W = 1080  # config.WEBCAM_CROP_W
K, S, TH = 25, 41, 20  # config.KERNEL_SIZE / SEARCH_RANGE_X / EDGE_TH
NFRAMES = 5

GROUPS = {
    "室内の一般的な場面": ["20260714_181338", "20260715_155659", "20260715_182517", "20260805_180017", "20260715_180809"],
    "車の背面の印刷＋ブラインド等": ["20260703_174944", "20260710_143749", "20260714_180936", "20260715_155442",
                                "20260715_180845", "20260715_181809", "20260715_184751", "20260806_103931",
                                "20260806_111817", "clip92"],
    "正式結果の通常シーンと標識撮影": ["b5_scene_01", "scene_01", "sign_far", "sign_near"],
    "参考: ノイズ板（等方的）": ["b5_dist_2050", "b5_dist_1000", "b5_dist_4000"],
    "参考: test/2300.JPG": ["test/2300.JPG"],
}


def preprocess(raw):
    h, w = raw.shape[:2]
    if (w, h) != (1280, 720):
        return cv2.resize(raw, IMAGE_SIZE)
    x0 = (w - CROP_W) // 2
    return cv2.resize(raw[:, x0:x0 + CROP_W], IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def frames_of(session):
    if session.startswith("test/"):
        return [os.path.join(ROOT, "data", session)]
    d = os.path.join(ROOT, "data", "captures", session)
    fs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    if not fs:
        return []
    idx = [int(len(fs) * (i + 0.5) / NFRAMES) for i in range(NFRAMES)]
    return [os.path.join(d, fs[i]) for i in sorted(set(idx))]


def analyze(img_bgr, ch):
    c = img_bgr[:, :, {"B": 0, "G": 1, "R": 2}[ch]]
    gx = cv2.Sobel(c, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(c, cv2.CV_32F, 0, 1)
    ex = cv2.convertScaleAbs(gx)
    _, sel_i = cv2.threshold(ex, TH, 255, cv2.THRESH_TOZERO)
    mag = np.hypot(gx, gy)
    mag_u8 = np.clip(np.rint(mag), 0, 255)  # convertScaleAbs と同じ丸め・飽和
    H, W = c.shape
    k, s = (K - 1) // 2, (S - 1) // 2
    roi = np.zeros((H, W), bool)
    roi[k:H - k, s + k + 1:W - s - k - 1] = True
    m_i = (sel_i > 0) & roi
    m_ii = (mag_u8 > TH) & roi
    phi = np.degrees(np.arctan2(np.abs(gy), np.abs(gx)))
    theta_img = np.degrees(np.arctan2(gy, gx)) % 180.0  # 画像座標（y 下向き）での法線の向き
    return m_i, m_ii, phi, mag, theta_img


def main():
    rows, hist_rows = [], []
    bins = np.arange(0, 91, 5)
    agg = {}
    theta_pool = {}
    for grp, sessions in GROUPS.items():
        for ses in sessions:
            fr = frames_of(ses)
            if not fr:
                print("no frames", ses)
                continue
            for ch in ("R", "G"):
                n_i = n_ii = 0
                phis, mags, ths = [], [], []
                for f in fr:
                    raw = cv2.imread(f, cv2.IMREAD_COLOR)
                    img = preprocess(raw)
                    m_i, m_ii, phi, mag, th = analyze(img, ch)
                    n_i += int(m_i.sum())
                    n_ii += int(m_ii.sum())
                    phis.append(phi[m_ii])
                    mags.append(mag[m_ii])
                    ths.append(th[m_ii])
                phis = np.concatenate(phis)
                mags = np.concatenate(mags)
                ths = np.concatenate(ths)
                h_cnt, _ = np.histogram(phis, bins)
                h_w, _ = np.histogram(phis, bins, weights=mags)
                cosw = float((np.cos(np.deg2rad(phis)) * mags).sum() / mags.sum())
                cosu = float(np.cos(np.deg2rad(phis)).mean())
                r = dict(group=grp, session=ses, channel=ch, frames=len(fr), n_current=n_i / len(fr),
                         n_any=n_ii / len(fr), ratio_current_over_any=n_i / max(n_ii, 1),
                         frac_phi_gt45=float((phis > 45).mean()), frac_phi_gt60=float((phis > 60).mean()),
                         frac_phi_gt75=float((phis > 75).mean()),
                         wfrac_phi_gt60=float(mags[phis > 60].sum() / mags.sum()),
                         mean_cos_weighted=cosw, mean_cos=cosu)
                rows.append(r)
                for b0, c0, w0 in zip(bins[:-1], h_cnt, h_w):
                    hist_rows.append(dict(group=grp, session=ses, channel=ch, phi_lo=b0, count=int(c0), weight=float(w0)))
                agg.setdefault((grp, ch), [np.zeros(len(bins) - 1), np.zeros(len(bins) - 1), 0, 0])
                agg[(grp, ch)][0] += h_cnt / max(h_cnt.sum(), 1)  # セッションごとに正規化して足す（セッション間で等重み）
                agg[(grp, ch)][1] += h_w / max(h_w.sum(), 1e-9)
                agg[(grp, ch)][2] += 1
                if ch == "R":
                    # セッションごとに等重みにするため 2 万画素に間引いてプール
                    rng = np.random.default_rng(0)
                    pick = rng.choice(ths.size, size=min(20000, ths.size), replace=False)
                    theta_pool.setdefault(grp, []).append((ths[pick], mags[pick]))
                print(f"{grp} | {ses:16s} {ch} 現行の選別 {r['n_current']:8.0f} / 向き不問 {r['n_any']:8.0f} = {r['ratio_current_over_any']:.3f} | "
                      f"φ>45° {r['frac_phi_gt45']:.3f} φ>60° {r['frac_phi_gt60']:.3f} φ>75° {r['frac_phi_gt75']:.3f} | "
                      f"cosφ 平均 {cosu:.3f}（勾配重み {cosw:.3f}）")
    with open(os.path.join(OUT, "a5_edge_stats.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "a5_edge_hist.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hist_rows[0].keys()))
        w.writeheader()
        w.writerows(hist_rows)
    # グループごとの要約（セッション等重み）
    print("\n=== グループの要約（セッション等重みの平均）===")
    for grp in GROUPS:
        for ch in ("R", "G"):
            rr = [r for r in rows if r["group"] == grp and r["channel"] == ch]
            if not rr:
                continue
            f = lambda k: np.mean([r[k] for r in rr])
            print(f"{grp} {ch}: n={len(rr)} 現行/向き不問 {f('ratio_current_over_any'):.3f}（範囲 {min(r['ratio_current_over_any'] for r in rr):.3f}–{max(r['ratio_current_over_any'] for r in rr):.3f}）"
                  f" φ>45° {f('frac_phi_gt45'):.3f} φ>60° {f('frac_phi_gt60'):.3f}（範囲 {min(r['frac_phi_gt60'] for r in rr):.3f}–{max(r['frac_phi_gt60'] for r in rr):.3f}） φ>75° {f('frac_phi_gt75'):.3f}"
                  f" cosφ {f('mean_cos'):.3f}")
    # 図
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e87ba4"]
    centers = 0.5 * (bins[:-1] + bins[1:])
    for ax, ch in zip(axs, ("R", "G")):
        for (grp, col) in zip(GROUPS, pal):
            if (grp, ch) not in agg:
                continue
            hc, hw, n, _ = agg[(grp, ch)]
            ax.plot(centers, hc / n, "-o", color=col, lw=2, ms=4, label=grp)
        ax.set_xlabel("法線と水平のなす角 φ [deg]（0=縦エッジ・90=横エッジ）")
        ax.set_title(f"{ch} チャネル: |∇{ch}|>20 の画素の向きの分布（5° 幅・セッション等重み）", fontsize=9.5)
        ax.set_xlim(0, 90)
        ax.set_xticks(range(0, 91, 15))
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("画素の割合（ビンあたり）")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "a5_edge_orientation.png"), dpi=110)
    plt.close(fig)
    # カタログ形状の基線を、場面のエッジ向きで平均
    prof_path = os.path.join(OUT, "a2_profiles.npy")
    if os.path.exists(prof_path):
        prof = np.load(prof_path, allow_pickle=True).item()
        out = []
        for grp, lst in theta_pool.items():
            th = np.concatenate([t for t, _ in lst])
            mg = np.concatenate([m for _, m in lst])
            for key, d in prof.items():
                vals = []
                for mirror in (False, True):
                    # 画像は y 下向き・開口図は y 上向き → 開口の角度 = −θ_img（mod 180）。表裏の取り違えに備えて鏡像も出す
                    ta = (-th) % 180.0 if not mirror else th % 180.0
                    b = np.interp(ta, np.append(d["thetas"], 180.0), np.append(d["prof"], d["prof"][0]))
                    vals.append((float(b.mean()), float((b < 0.2).mean()), float((b < 0.1).mean())))
                worst = min(vals, key=lambda v: v[0])
                out.append(dict(group=grp, key=key, label=d["label"], mean_baseline=worst[0],
                                mean_baseline_other_mirror=max(vals, key=lambda v: v[0])[0],
                                frac_below_0p2=worst[1], frac_below_0p1=worst[2]))
        with open(os.path.join(OUT, "a5_design_scene_weighted.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print("\n=== 場面のエッジ向きで平均した基線（画素等重み・R の |∇|>20 の画素） ===")
        for grp in theta_pool:
            print(grp)
            for r in sorted([o for o in out if o["group"] == grp], key=lambda o: -o["mean_baseline"]):
                print(f"   {r['label']:22s} 平均基線 {r['mean_baseline']:.3f}（鏡像 {r['mean_baseline_other_mirror']:.3f}） 基線<0.2 の割合 {r['frac_below_0p2']:.3f} <0.1 {r['frac_below_0p1']:.3f}")


if __name__ == "__main__":
    main()
