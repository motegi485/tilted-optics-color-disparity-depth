# -*- coding: utf-8 -*-
"""増えた有効画素は「(i) の有効画素のすぐ近く（同じ窓の中）の埋め足し」か「(i) が取れていなかった場所（新しい構造）」か。
読み取りのみ・非公式の再解析。場面: b5_scene_01（10 フレーム）と 2026-07 のブラインドの 5 セッション（v4 と同じ 4 フレーム）。

近い = (i) の有効画素（同じフレーム）からチェビシェフ距離 12 px 以内（25×25 窓の半幅）。遠い = それより外。
指標: 増えた画素数／うち遠い割合／遠い画素の窓の h の分布（h<0.05 の割合）／時間方向のばらつき（場面内で 5 フレーム以上〈scene〉・
4 フレームとも〈blinds〉有効な画素の d の std の中央値）／scene の紙（v3 と同じ領域・正解 2800 mm）での良好 10 % の割合。
出力: verReal_out/v5_coverage.txt
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
warnings.simplefilter("ignore")
import cv2
import numpy as np

import vr_lib as L

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verReal_out")
log = open(os.path.join(OUT, "v5_coverage.txt"), "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")
    log.flush()


COMBOS = [("i_all", "S_mag", "(i) の照合・S_mag"), ("i_all", "S_h0.1", "(iii) h≥0.1"), ("i_all", "S_J0", "(iii) Txx≥J0"),
          ("cl50", "S_mag", "(ii) color-line・S_mag")]
yy, xx = np.indices((L.H, L.W))
PAPER = (((xx >= 33) & (xx <= 85)) | ((xx >= 216) & (xx < 495))) & (yy >= 12) & (yy < 295)
KER = np.ones((25, 25), np.uint8)


def frames_of(ses):
    if ses == "b5_scene_01":
        return list(range(10, 470, 50))
    idxs = L.list_indices(ses)
    mid = len(idxs) // 2
    return [idxs[min(max(mid - 4 + 3 * k, 0), len(idxs) - 1)] for k in range(4)]


SESSIONS = ["b5_scene_01", "20260715_184146", "20260715_155442", "20260715_180809", "20260715_180845", "20260710_143749"]
t0 = time.time()
summary = {c[2]: dict(gain=[], far_frac=[], far_hlt05=[], near_ts=[], far_ts=[], base_ts=[]) for c in COMBOS}
for ses in SESSIONS:
    frs = frames_of(ses)
    Di, Dm, Fh, Near, Zm = [], {c[2]: [] for c in COMBOS}, [], [], {c[2]: [] for c in COMBOS}
    Zi = []
    for idx in frs:
        proc = L.load_proc(ses, idx)
        f = L.feats(proc)
        di = L.banded(L.rust_disp(proc, L.TH)).astype(np.float64)
        dall = L.rust_disp(proc, 0)
        dcl = L.disp_cl(proc, 0.5)
        Di.append(di)
        Zi.append(L.to_depth(L.rust_disp(proc, L.TH)))
        Fh.append(f["h_R"])
        vi = np.isfinite(di).astype(np.uint8)
        Near.append(cv2.dilate(vi, KER) > 0)
        for (m, sel, lab) in COMBOS:
            src = dall if m == "i_all" else dcl
            dm = np.where(f[sel], src, np.nan).astype(np.float32)
            Dm[lab].append(L.banded(dm).astype(np.float64))
            Zm[lab].append(L.to_depth(dm))
    Di = np.stack(Di)
    Fh = np.stack(Fh)
    Near = np.stack(Near)
    nf = len(frs)
    need = 5 if ses == "b5_scene_01" else nf
    vi = np.isfinite(Di)
    ts_i = np.where(vi.sum(0) >= need, np.nanstd(Di, axis=0), np.nan)
    p(f"\n=== {ses}（{nf} フレーム・{time.time() - t0:.0f}s）: (i) 有効 {vi.sum() / nf:.0f}/フレーム・時間std中央値 {np.nanmedian(ts_i[vi.any(0)]):.3f} px ===")
    for (m, sel, lab) in COMBOS:
        D = np.stack(Dm[lab])
        v = np.isfinite(D)
        add = v & ~vi
        far = add & ~Near
        nadd = add.sum() / nf
        nfar = far.sum() / nf
        hfar = Fh[far]
        hnear = Fh[add & Near]
        ts = np.where(v.sum(0) >= need, np.nanstd(D, axis=0), np.nan)
        any_far = far.any(0)
        any_near = (add & Near).any(0) & ~any_far
        ts_far = np.nanmedian(ts[any_far]) if np.isfinite(ts[any_far]).any() else np.nan
        ts_near = np.nanmedian(ts[any_near]) if np.isfinite(ts[any_near]).any() else np.nan
        line = (f"  {lab:22s} 有効 {v.sum() / nf:6.0f}（×{v.sum() / max(vi.sum(), 1):.2f}） 増えた {nadd:6.0f}/フレーム・うち遠い {nfar:6.0f}（{nfar / max(nadd, 1e-9):.2f}）"
                f" | 窓 h<0.05 の割合: 近い {np.mean(hnear < 0.05) if hnear.size else np.nan:.2f}・遠い {np.mean(hfar < 0.05) if hfar.size else np.nan:.2f}"
                f" | 時間std中央値: 近い {ts_near:.3f}・遠い {ts_far:.3f} px")
        if ses == "b5_scene_01":
            Z = np.stack(Zm[lab])
            gi = []
            for k, (msk) in enumerate([vi & PAPER[None], add & Near & PAPER[None], far & PAPER[None]]):
                zz = (np.stack(Zi) if k == 0 else Z)[msk]
                zz = zz[np.isfinite(zz)]
                gi.append((zz.size / nf, np.mean(np.abs(zz - 2800) <= 280) if zz.size else np.nan, np.median(zz) if zz.size else np.nan))
            line += (f"\n      紙（2800 mm）の良好10%: (i) {gi[0][1]:.2f}（{gi[0][0]:.0f}）・増えた近い {gi[1][1]:.2f}（{gi[1][0]:.0f}・中央値 {gi[1][2]:.0f}）"
                     f"・増えた遠い {gi[2][1]:.2f}（{gi[2][0]:.0f}・中央値 {gi[2][2]:.0f}）")
        p(line)
        s = summary[lab]
        s["gain"].append(v.sum() / max(vi.sum(), 1))
        s["far_frac"].append(nfar / max(nadd, 1e-9))
        s["far_hlt05"].append(np.mean(hfar < 0.05) if hfar.size else np.nan)
        s["near_ts"].append(ts_near)
        s["far_ts"].append(ts_far)
        s["base_ts"].append(np.nanmedian(ts_i[vi.any(0)]))
p("\n=== 6 場面の要約（場面等重みの平均・時間std は NaN を除く）===")
for lab, s in summary.items():
    p(f"  {lab:22s} 倍率 ×{np.mean(s['gain']):.2f}・増えた画素のうち遠い割合 {np.mean(s['far_frac']):.2f}・遠い画素の窓 h<0.05 {np.nanmean(s['far_hlt05']):.2f}"
      f"・時間std (i) {np.nanmean(s['base_ts']):.3f}／近い {np.nanmean(s['near_ts']):.3f}／遠い {np.nanmean(s['far_ts']):.3f} px")
p(f"\n合計 {time.time() - t0:.0f}s")
log.close()
