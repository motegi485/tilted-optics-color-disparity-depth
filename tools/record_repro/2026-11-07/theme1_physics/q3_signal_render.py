"""Q3・Q4: 灯器（3 灯・横型）を PSF で合成し、灯火の分離・色・位置を読めるか、外形と灯火から視差が取れるかを調べる。

条件はすべて [仮定]: 空=0.6（線形・平坦な白）、灯器本体=0.03、消灯=0.05、点灯の最大チャネル=0.9（灯器本体に加算）。
PSF は物体の中心の値で一定（灯器の縦の中でぼけは変わらない近似。誤差は別に出す）。
"""
import csv
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
ROOT = "C:/Users/t5424001/dev/tilted_optics_color_disparity_depth"
sys.path.insert(0, ROOT)
import cv2
import numpy as np

import optics as op
import spectral as sp
import synth as sy
from core.disparity import calcDisparityByZNCC

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
SS = 4
BETA = op.BETA

t_y, t_c = sy.measured_like_filter()
SP_M = sy.Spectra(t_y, t_c, sy.narrow_sensors(), "実測合わせ(白で G=Y0.7:C0.3)")
t_y2, t_c2 = sp.filter_pair(y_edge=520.0, c_edge=565.0, width=6.0)
SP_N = sy.Spectra(t_y2 * 1.0, t_c2 * (0.36 + 0.64 * sp.expit((495.0 - sp.LAM) / 10.0)), sy.narrow_sensors(), "Y端520(青505はC側)")
SP_M.describe()
SP_N.describe()


def flat_w(level_white):
    return {k: (level_white[k][0], level_white[k][1]) for k in level_white}


def build(rig, z, lit_idx, lit_name, spec, sky=0.6, body=0.03, unlit=0.05, lit_peak=0.9, vertical=False, lamp=300.0, W=1200.0, Hh=450.0, pitch=400.0, H_c=5000.0, c_override=None, margin_extra=70):
    c = float(rig.blur_signed(H_c, z)) if c_override is None else c_override
    cy = abs(c) * np.cos(rig.theta)
    w_px, h_px = rig.F * W / z, rig.F * Hh / z
    lamp_px, pitch_px = rig.F * lamp / z, rig.F * pitch / z
    margin = int(max(margin_extra, abs(c) * 1.2 + 62))
    ms = sy.signal_masks(w_px, h_px, lamp_px, pitch_px, SS, margin, vertical=vertical)
    head = ms["head"]
    lamps = ms["lamps"]
    white = spec.white
    comps = []
    comps.append((~head, white, sky))
    body_mask = head.copy()
    for lm in lamps:
        body_mask &= ~lm
    comps.append((body_mask, white, body))
    for i, lm in enumerate(lamps):
        comps.append((lm, white, unlit))
        if i == lit_idx and lit_name is not None:
            comps.append((lm, spec.led[lit_name], lit_peak))
    lin = sy.render(comps, c, cy, spec, ss=SS)
    return lin, ms, c, dict(w_px=w_px, h_px=h_px, lamp_px=lamp_px, pitch_px=pitch_px, margin=margin)


def sample(img, cx, cy, half=1):
    x, y = int(round(cx - 0.5)), int(round(cy - 0.5))
    return float(img[y - half : y + half + 1, x - half : x + half + 1].mean())


def lit_metrics(lin, ms, info, lit_idx, lit_name, spec, vertical=False):
    W = spec.led[lit_name]
    tot = {k: W[k][0] + W[k][1] for k in sp.CH}
    kd = max(tot, key=tot.get)
    cen = ms["centers"]
    # 背景（点灯なしの合成を引かず、消灯の隣の灯火中心の値で評価する）
    lit_c = cen[lit_idx]
    nb_idx = 1 if lit_idx != 1 else 0
    nb_c = cen[nb_idx]
    v_lit = {k: sample(lin[k], lit_c[0], lit_c[1]) for k in sp.CH}
    v_nb = {k: sample(lin[k], nb_c[0], nb_c[1]) for k in sp.CH}
    # コントラスト（線形・支配チャネル）
    contrast = (v_lit[kd] - v_nb[kd]) / max(v_lit[kd] + v_nb[kd], 1e-6)
    # 点灯による信号の隣への漏れ: 点灯なしとの差で見る
    return dict(kd=kd, v_lit=v_lit, v_nb=v_nb, contrast=contrast)


def main():
    rigs = [r for r in op.default_rigs() if not r.name.startswith("対照")]
    ctrl = op.rig_untilted("対照θ=0・無限遠")
    zs = [20e3, 30e3, 50e3, 70e3, 100e3]
    lit_list = [("LED赤630", 0), ("LED黄590", 1), ("LED青505", 2)]
    rows = []
    print()
    print("=== 灯火の読み取り（横型・線形値）===")
    print("配置 | z | 灯色 | c | c/灯間隔 | 隣の灯への漏れ | 点灯/消灯コントラスト | 中心の色(R,G,B/最大) | 色の変化(シャープとのL1) | 判定色 | 位置ずれ/灯間隔 | 判定灯 | 灯器の空に対する暗さ(G)")
    refs = {}
    for lit, idx in lit_list:
        lin0, ms0, _, info0 = build(ctrl, 50e3, idx, lit, SP_M, c_override=0.0)
        c0 = ms0["centers"][idx]
        v = np.array([sample(lin0[k], c0[0], c0[1]) for k in sp.CH])
        refs[lit] = v / v.max()
    print("基準色（シャープ・R,G,B/最大）:", {k: np.round(v, 3).tolist() for k, v in refs.items()})
    for R in rigs + [ctrl]:
        for z in zs:
            for lit, idx in lit_list:
                lin, ms, c, info = build(R, z, idx, lit, SP_M)
                # 点灯なし
                lin_off, _, _, _ = build(R, z, idx, None, SP_M)
                kd = max(sp.CH, key=lambda k: SP_M.led[lit][k][0] + SP_M.led[lit][k][1])
                cen = ms["centers"]
                lc = cen[idx]
                nb = cen[1 if idx != 1 else 0]
                sig = lin[kd] - lin_off[kd]
                vc = sample(sig, lc[0], lc[1])
                vn = sample(sig, nb[0], nb[1])
                leak = vn / vc if vc > 1e-9 else np.nan
                # 灯火の重心のずれ（点灯の差分から）
                H_, W_ = sig.shape
                yy, xx = np.mgrid[0:H_, 0:W_]
                mass = np.clip(sig, 0, None)
                xc = (mass * (xx + 0.5)).sum() / mass.sum()
                yc = (mass * (yy + 0.5)).sum() / mass.sum()
                shift = xc - lc[0]
                # 中心の色
                v = np.array([sample(lin[k], lc[0], lc[1]) for k in sp.CH])
                vnb = sample(lin[kd], nb[0], nb[1])
                contrast = (sample(lin[kd], lc[0], lc[1]) - vnb) / (sample(lin[kd], lc[0], lc[1]) + vnb)
                col = v / v.max()
                dcol = np.abs(col - refs[lit]).sum()
                pred = min(refs, key=lambda k: np.abs(col - refs[k]).sum())
                # どの灯か: 重心に最も近い灯（灯器の中心は G の谷から求める代わりに既知とする）
                pred_i = int(np.argmin([abs(xc - cc[0]) for cc in cen]))
                # 灯器の暗さ（G の最小と空の比）
                G = lin["G"]
                dip = 1 - G[int(ms["cy"]) - 2 : int(ms["cy"]) + 3, int(ms["cx"]) + int(info["pitch_px"] / 2) - 2 : int(ms["cx"]) + int(info["pitch_px"] / 2) + 3].mean() / 0.6
                rows.append(dict(rig=R.name, z=z / 1000, lit=lit, c=c, c_pitch=abs(c) / info["pitch_px"], leak=leak, contrast=contrast,
                                 col_r=col[0], col_g=col[1], col_b=col[2], dcol=dcol, pred=pred, shift=shift, shift_pitch=shift / info["pitch_px"], pred_i=pred_i, true_i=idx, dip=dip))
                if z in (20e3, 50e3, 100e3) or R.name.startswith("(c)"):
                    print(f"{R.name} | {z/1000:.0f} | {lit} | {c:+.1f} | {abs(c)/info['pitch_px']:.2f} | {100*leak:.1f}% | {contrast:.2f} | {col[0]:.2f},{col[1]:.2f},{col[2]:.2f} | {dcol:.2f} | {'○' if pred==lit else '×'+pred} | {shift/info['pitch_px']:+.3f} | {'○' if pred_i==idx else '×'} | {dip:.2f}")
    with open(os.path.join(OUT, "q3_lamp_metrics.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print("=== 縦型の灯器との比較（灯火は縦に並ぶ。R・B の PSF は縦が c、横が c/2。1.0 m 面・50 m・赤） ===")
    for R in [rigs[0], rigs[2], rigs[6]]:
        for vertical in [False, True]:
            lin, ms, c, info = build(R, 50e3, 0, "LED赤630", SP_M, vertical=vertical)
            lin_off, _, _, _ = build(R, 50e3, 0, None, SP_M, vertical=vertical)
            sig = lin["R"] - lin_off["R"]
            cen = ms["centers"]
            vc = sample(sig, cen[0][0], cen[0][1])
            vn = sample(sig, cen[1][0], cen[1][1])
            print(f"{R.name} vertical={vertical}: 隣への漏れ {100*vn/vc:.1f}% c={c:.1f}")

    # ------------------------------------------------------------------
    # 4:2:0 の色への影響（ガンマ後 8bit）
    print()
    print("=== 4:2:0 が灯火の色に与える影響（ガンマ後 8bit・灯火中心 3×3 と縁） ===")
    print("配置 z 灯色 | 灯火径px | 中心 RGB(4:4:4) | 中心 RGB(4:2:0) | ΔRGB 最大 | B チャネルの縁の偽の勾配 |∂x|（点灯あり−なし）p99 4:4:4→4:2:0")
    R = rigs[0]
    for z in [50e3, 100e3, 200e3, 300e3, 500e3]:
        for lit, idx in lit_list:
            lin, ms, c, info = build(R, z, idx, lit, SP_M)
            lin_off, _, _, _ = build(R, z, idx, None, SP_M)
            u = sy.to_u8(lin)
            u4 = sy.chroma420(u)
            uoff = sy.to_u8(lin_off)
            uoff4 = sy.chroma420(uoff)
            cen = ms["centers"][idx]
            x, y = int(cen[0]), int(cen[1])
            c44 = u[y - 1 : y + 2, x - 1 : x + 2].mean(axis=(0, 1))[::-1]
            c42 = u4[y - 1 : y + 2, x - 1 : x + 2].mean(axis=(0, 1))[::-1]
            r_ = int(info["lamp_px"] / 2 + 6)
            def gm(a):
                return np.abs(cv2.Sobel(a[y - r_ : y + r_ + 1, x - r_ : x + r_ + 1, 0], cv2.CV_32F, 1, 0))
            g0 = np.percentile(np.abs(gm(u) - gm(uoff)), 99)
            g1 = np.percentile(np.abs(gm(u4) - gm(uoff4)), 99)
            print(f"(a) {z/1000:.0f}m {lit} | {info['lamp_px']:.1f} | {np.round(c44).astype(int).tolist()} | {np.round(c42).astype(int).tolist()} | {np.abs(c44-c42).max():.0f} | {g0:.0f} → {g1:.0f}")

    # ------------------------------------------------------------------
    # ZNCC 実験
    print()
    print("=== Q4: 実装の ZNCC を灯器に掛ける（ノイズ σ=1.5 階調・10 枚の中央値を使い、枠の外形（両端の短辺）と灯火の縁で分ける） ===")
    print("処理 | 配置 | z | 灯色 | 期待 d | 端の縁: 有効px, 枠内中央値の平均, 偏り, 枚ごとの標準偏差 | 灯火の縁: 有効px, 中央値, 期待との差 | 枠全体の 3 群: 中央値（偽の d=0 付近の割合）")
    zrows = []
    for label, down, K, S, T, use420 in [
        ("1080幅 4:4:4 K25 S83 T20", False, 25, 83, 20, False),
        ("1080幅 4:2:0 K25 S83 T20", False, 25, 83, 20, True),
        ("528幅 4:2:0 K25 S41 T20(現行)", True, 25, 41, 20, True),
    ]:
        for R in [rigs[0], rigs[1], rigs[2], rigs[6]]:
            for z in zs:
                for lit, idx in [("LED赤630", 0), ("LED青505", 2)]:
                    lin, ms, c, info = build(R, z, idx, lit, SP_M)
                    scale = op.R_1080_TO_528 if down else 1.0
                    dtrue = BETA * c / scale
                    res = []
                    for seed in range(10):
                        rng = np.random.default_rng(seed + 100)
                        u = sy.to_u8(lin, noise=1.5, rng=rng)
                        if use420:
                            u = sy.chroma420(u)
                        if down:
                            u = cv2.resize(u, (int(round(u.shape[1] / scale)), int(round(u.shape[0] / scale))), interpolation=cv2.INTER_AREA)
                        dm = calcDisparityByZNCC(u[:, :, 2], u[:, :, 0], K, S, T)
                        H_, W_ = dm.shape
                        yy, xx = np.mgrid[0:H_, 0:W_]
                        cx, cy_ = ms["cx"] / scale, ms["cy"] / scale
                        hw, hh = ms["hw"] / scale, ms["hh"] / scale
                        band = max(6.0, abs(c) / scale)
                        inrow = np.abs(yy + 0.5 - cy_) <= hh
                        end = inrow & (np.abs(np.abs(xx + 0.5 - cx) - hw) <= band)
                        lc = ms["centers"][idx]
                        rr = np.hypot(xx + 0.5 - lc[0] / scale, yy + 0.5 - lc[1] / scale)
                        lamp_edge = (np.abs(rr - info["lamp_px"] / 2 / scale) <= max(3.0, abs(c) / scale / 2)) & ~end
                        dend = dm[end & np.isfinite(dm)]
                        dlamp = dm[lamp_edge & np.isfinite(dm)]
                        res.append((dend.size, np.median(dend) if dend.size else np.nan, dlamp.size, np.median(dlamp) if dlamp.size else np.nan))
                    res = np.array(res, dtype=float)
                    n_end = np.nanmean(res[:, 0])
                    m_end = np.nanmean(res[:, 1]) if np.isfinite(res[:, 1]).any() else np.nan
                    s_end = np.nanstd(res[:, 1]) if np.isfinite(res[:, 1]).sum() > 1 else np.nan
                    n_l = np.nanmean(res[:, 2])
                    m_l = np.nanmean(res[:, 3]) if np.isfinite(res[:, 3]).any() else np.nan
                    print(f"{label} | {R.name} | {z/1000:.0f} | {lit} | {dtrue:.2f} | {n_end:.0f}, {m_end:.2f}, {m_end-dtrue:+.2f}, {s_end:.3f} | {n_l:.0f}, {m_l:.2f}, {m_l-dtrue:+.2f}")
                    zrows.append(dict(proc=label, rig=R.name, z=z / 1000, lit=lit, d_true=dtrue, n_end=n_end, m_end=m_end, bias_end=m_end - dtrue, std_end=s_end, n_lamp=n_l, m_lamp=m_l, bias_lamp=m_l - dtrue))
    with open(os.path.join(OUT, "q4_zncc_signal.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(zrows[0].keys()))
        w.writeheader()
        w.writerows(zrows)

    # 背景が暗い場合（建物や木）: 外形のコントラストが小さい
    print()
    print("=== 背景を暗くしたとき（灯器外形のコントラスト低下）・(a)・50 m・赤・528幅現行 ===")
    R = rigs[0]
    z = 50e3
    for sky in [0.6, 0.2, 0.08, 0.04]:
        lin, ms, c, info = build(R, z, 0, "LED赤630", SP_M, sky=sky)
        scale = op.R_1080_TO_528
        dtrue = BETA * c / scale
        res = []
        for seed in range(10):
            rng = np.random.default_rng(seed + 300)
            u = sy.chroma420(sy.to_u8(lin, noise=1.5, rng=rng))
            u = cv2.resize(u, (int(round(u.shape[1] / scale)), int(round(u.shape[0] / scale))), interpolation=cv2.INTER_AREA)
            dm = calcDisparityByZNCC(u[:, :, 2], u[:, :, 0], 25, 41, 20)
            H_, W_ = dm.shape
            yy, xx = np.mgrid[0:H_, 0:W_]
            cx, cy_ = ms["cx"] / scale, ms["cy"] / scale
            hw, hh = ms["hw"] / scale, ms["hh"] / scale
            end = (np.abs(yy + 0.5 - cy_) <= hh) & (np.abs(np.abs(xx + 0.5 - cx) - hw) <= 6)
            x = dm[end & np.isfinite(dm)]
            res.append((x.size, np.median(x) if x.size else np.nan))
        res = np.array(res, dtype=float)
        print(f"空={sky}: 端の縁の有効画素 {res[:,0].mean():.0f}, 中央値 {np.nanmean(res[:,1]):.2f}（期待 {dtrue:.2f}）")

    # 白飛びした空
    print()
    print("=== 空が白飛びしたとき（空=1.5 → 8bit で 255 に張り付く）・(a)・50 m ===")
    for noise in [1.5]:
        lin, ms, c, info = build(R, z, 0, "LED赤630", SP_M, sky=1.5)
        u = sy.to_u8(lin)  # 白飛びの領域はノイズ後も 255 付近に張り付く
        rng = np.random.default_rng(5)
        arr = sy.enc_gamma(np.stack([lin["B"], lin["G"], lin["R"]], axis=-1))
        sat = arr >= 254.5
        n = rng.normal(0, noise, arr.shape)
        n[sat] = 0.0
        u = np.clip(np.round(arr + n), 0, 255).astype(np.uint8)
        dm = calcDisparityByZNCC(u[:, :, 2], u[:, :, 0], 25, 83, 20)
        H_, W_ = dm.shape
        yy, xx = np.mgrid[0:H_, 0:W_]
        end = (np.abs(yy + 0.5 - ms["cy"]) <= ms["hh"]) & (np.abs(np.abs(xx + 0.5 - ms["cx"]) - ms["hw"]) <= max(6, abs(c)))
        x = dm[end & np.isfinite(dm)]
        print(f"1080幅 4:4:4: 端の縁の候補 {int(end.sum())} 画素, 有効 {x.size}, 中央値 {np.median(x) if x.size else np.nan}（期待 {BETA*c:.2f}）")


if __name__ == "__main__":
    main()
