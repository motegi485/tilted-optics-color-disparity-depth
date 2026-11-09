# -*- coding: utf-8 -*-
"""(c) ブラインド等の横縞が多い屋内の場面（2026-07〜08 のセッション）で推定法を比べる。正解距離はない。読み取りのみ・非公式の再解析。

指標（推論帯 0.45<|d|<5 を通った視差 d [px] で見る。7〜8 月の撮影条件は記録票に見当たらないので、live 係数での深度には直さない）:
  有効画素数／(i) との一致（両方有効な画素の |d − d_i| の中央値）／近傍の (i) との整合（25×25 近傍に (i) の有効画素が 10 以上あるとき
  |d − その中央値|）／時間方向のばらつき（4 フレームとも有効な画素の d の std）／空間の偏差（7×7 近傍の中央値からのずれ）。
  これらを窓の h（R の構造テンソル）の区分別にも出す。
フレーム: 各セッションの中央付近から 3 フレームおきに 4 枚。
出力: verReal_out/v4_blinds.txt・v4_blinds_rows.csv・v4_blinds_<session>.png
"""
import csv
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vr_lib as L
import vr_eval as E
import vr_scene as SC

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verReal_out")
SESSIONS = ["20260715_184146", "20260715_155442", "20260715_180809", "20260715_180845", "20260710_143749"]
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
log = open(os.path.join(OUT, "v4_blinds.txt"), "w", encoding="utf-8")
warnings.simplefilter("ignore")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")
    log.flush()


def neighbor_i_dev(db, di, r=12, min_n=10):
    pad = np.pad(di, r, constant_values=np.nan)
    win = sliding_window_view(pad, (2 * r + 1, 2 * r + 1))
    ys, xs = np.nonzero(np.isfinite(db))
    out = np.full(db.shape, np.nan)
    if ys.size == 0:
        return out
    w = win[ys, xs].reshape(ys.size, -1)
    n = np.isfinite(w).sum(1)
    med = np.nanmedian(w, axis=1)
    v = np.abs(db[ys, xs] - med)
    v[n < min_n] = np.nan
    out[ys, xs] = v
    return out


rows = []
t0 = time.time()
for ses in SESSIONS:
    idxs = L.list_indices(ses)
    mid = len(idxs) // 2
    frames = [idxs[min(max(mid - 4 + 3 * k, 0), len(idxs) - 1)] for k in range(4)]
    store = {c[:2]: [] for c in SC.SCOMBOS}
    fl = []
    procs = []
    for idx in frames:
        proc = L.load_proc(ses, idx)
        procs.append(proc)
        f, d = SC.compute(proc)
        fl.append(f)
        for (m, sel, _) in SC.SCOMBOS:
            store[(m, sel)].append(L.banded(E.combo_disp(f, d, m, sel)).astype(np.float64))
    nf = len(frames)
    DI = np.stack(store[("i", "S_i")])
    hb_all = np.stack([E.bins_of(f)[0] for f in fl])
    Si = np.stack([f["S_i"] for f in fl])
    p(f"\n=== {ses}（フレーム {frames}・{time.time() - t0:.0f}s）===")
    nb = None
    for (m, sel, label) in SC.SCOMBOS:
        D = np.stack(store[(m, sel)])
        valid = np.isfinite(D)
        allv = valid.all(0)
        tstd = np.where(allv, np.nanstd(D, axis=0), np.nan)
        sdev = np.stack([SC.local_dev(D[k]) for k in range(nf)])
        ndev = np.stack([neighbor_i_dev(D[k], DI[k]) for k in range(nf)])
        for sub in ["all", "outSi"] + [f"h:{lab}" for lab in E.HB_LABEL]:
            if sub == "all":
                msk = valid
            elif sub == "outSi":
                msk = valid & ~Si
            else:
                msk = valid & (hb_all == E.HB_LABEL.index(sub[2:]))
            n = int(msk.sum())
            both = msk & np.isfinite(DI)
            tm = msk[0] & np.isfinite(tstd)
            nd = ndev[msk & np.isfinite(ndev)]
            sd = sdev[msk & np.isfinite(sdev)]
            rows.append(dict(session=ses, method=m, sel=sel, label=label, subset=sub, n_per_frame=n / nf,
                             agree_i=float(np.median(np.abs(D[both] - DI[both]))) if both.any() else np.nan,
                             n_nbr=int(nd.size) / nf, nbr_dev_median=float(np.median(nd)) if nd.size else np.nan,
                             nbr_frac_le_0p5=float((nd <= 0.5).mean()) if nd.size else np.nan,
                             tstd_median=float(np.median(tstd[tm])) if tm.any() else np.nan, n_tstd=int(tm.sum()),
                             sdev_median=float(np.median(sd)) if sd.size else np.nan))
        r = [x for x in rows if x["session"] == ses and x["method"] == m and x["sel"] == sel and x["subset"] == "all"][0]
        ro = [x for x in rows if x["session"] == ses and x["method"] == m and x["sel"] == sel and x["subset"] == "outSi"][0]
        if nb is None:
            nb = r["n_per_frame"]
        p(f"  {label:26s} 有効 {r['n_per_frame']:7.0f}（×{r['n_per_frame'] / max(nb, 1e-9):.2f}） (i)との差 {r['agree_i']:.3f} px"
          f" 近傍(i)との差 {r['nbr_dev_median']:.3f} px（≤0.5 px {r['nbr_frac_le_0p5']:.2f}） 時間std {r['tstd_median']:.3f} 空間偏差 {r['sdev_median']:.3f}"
          f" | S_i の外 {ro['n_per_frame']:6.0f} 近傍(i)との差 {ro['nbr_dev_median']:.3f}（≤0.5 {ro['nbr_frac_le_0p5']:.2f}） 時間std {ro['tstd_median']:.3f}")
    # 図（1 枚目のフレーム・視差）
    fig, axs = plt.subplots(2, 5, figsize=(22, 6.8))
    gray = cv2.cvtColor(procs[0], cv2.COLOR_BGR2GRAY).astype(float) / 255.0
    axs[0, 0].imshow(cv2.cvtColor(procs[0], cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title(f"{ses} #{frames[0]}", fontsize=10)
    for ax, (m, sel, label) in zip(axs.ravel()[1:], SC.SCOMBOS):
        D0 = store[(m, sel)][0]
        ax.imshow(0.45 + 0.55 * gray, cmap="gray", vmin=0, vmax=1)
        im = ax.imshow(np.ma.masked_invalid(D0), cmap="coolwarm", vmin=-5, vmax=5, interpolation="nearest")
        ax.set_title(f"{label}（{int(np.isfinite(D0).sum())}）", fontsize=9)
    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.015, label="視差 d [px]（推論帯の後）")
    fig.savefig(os.path.join(OUT, f"v4_blinds_{ses}.png"), dpi=70)
    plt.close(fig)

with open(os.path.join(OUT, "v4_blinds_rows.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

p("\n=== 5 セッションの要約（セッション等重みの平均）===")
for (m, sel, label) in SC.SCOMBOS:
    rr = [x for x in rows if x["method"] == m and x["sel"] == sel and x["subset"] == "all"]
    ro = [x for x in rows if x["method"] == m and x["sel"] == sel and x["subset"] == "outSi"]
    base = [x for x in rows if x["method"] == "i" and x["sel"] == "S_i" and x["subset"] == "all"]
    gain = np.mean([a["n_per_frame"] / b["n_per_frame"] for a, b in zip(rr, base)])
    p(f"  {label:26s} 倍率 ×{gain:.2f} 近傍(i)との差 {np.nanmean([x['nbr_dev_median'] for x in rr]):.3f} px（≤0.5 {np.nanmean([x['nbr_frac_le_0p5'] for x in rr]):.2f}）"
      f" 時間std {np.nanmean([x['tstd_median'] for x in rr]):.3f} | S_i の外: 近傍(i)との差 {np.nanmean([x['nbr_dev_median'] for x in ro]):.3f}（≤0.5 {np.nanmean([x['nbr_frac_le_0p5'] for x in ro]):.2f}） 時間std {np.nanmean([x['tstd_median'] for x in ro]):.3f}")
p("\n=== 窓の h 別（5 セッションをまとめた平均）: 有効/フレーム・近傍(i)との差の ≤0.5 px 割合・時間std ===")
for (m, sel, label) in SC.SCOMBOS:
    parts = []
    for lab in E.HB_LABEL:
        rr = [x for x in rows if x["method"] == m and x["sel"] == sel and x["subset"] == f"h:{lab}"]
        parts.append(f"{lab}: {np.mean([x['n_per_frame'] for x in rr]):5.0f} / {np.nanmean([x['nbr_frac_le_0p5'] for x in rr]):.2f} / {np.nanmean([x['tstd_median'] for x in rr]):.2f}")
    p(f"  {label:26s} " + " | ".join(parts))
p(f"\n合計 {time.time() - t0:.0f}s")
log.close()
