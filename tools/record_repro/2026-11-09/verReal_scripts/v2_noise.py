# -*- coding: utf-8 -*-
"""(a) ノイズ板 B5 の 15 距離で、推定法 × 画素の選び方の組ごとに有効画素数と距離の誤差を出す（読み取りのみ・非公式の再解析）。

正解距離 = 記録票 2026-10-01 §3 の gt_mm（公称値に合わせ込み）。フレームは各距離 10・70・130・190（公式集計は先頭 10 を除く全フレーム）。
出力: verReal_out/v2_noise_rows.csv（距離×組×部分集合）、v2_noise_ratio.csv（d_m/d_i の比）、v2_noise.txt
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

import vr_lib as L
import vr_eval as E

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verReal_out")
DISTS = [1000, 1150, 1250, 1450, 1650, 1850, 2050, 2250, 2450, 2650, 2850, 3050, 3450, 3750, 4000]
FRAMES = [10, 70, 130, 190]
log = open(os.path.join(OUT, "v2_noise.txt"), "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")
    log.flush()


rows, ratio_rows = [], []
t_start = time.time()
for gt in DISTS:
    ses = f"b5_dist_{gt}"
    acc = {c[:2]: dict(err=[], hb=[], pb=[], inSi=[], d=[]) for c in E.COMBOS}
    rat = {m: dict(pos=[], neg=[], absdiff=[]) for m in ("cl50", "cl27", "rb", "gm")}
    nsel = {}
    for idx in FRAMES:
        proc = L.load_proc(ses, idx)
        f, d = E.compute(proc)
        hb, pb = E.bins_of(f)
        for sel in ("S_i", "S_mag", "S_h0.05", "S_h0.1", "S_J0"):
            nsel[sel] = nsel.get(sel, 0) + int(f[sel].sum())
        for (m, sel, _) in E.COMBOS:
            dm = E.combo_disp(f, d, m, sel)
            z = L.to_depth(dm)
            v = np.isfinite(z)
            a = acc[(m, sel)]
            a["err"].append((z[v] - gt).astype(np.float32))
            a["hb"].append(hb[v].astype(np.int8))
            a["pb"].append(pb[v].astype(np.int8))
            a["inSi"].append(f["S_i"][v])
        # 比: (i) の有効画素（S_i・推論帯内）で d_m / d_i
        di = L.banded(d["i"])
        base = np.isfinite(di)
        for m in rat:
            dm = d[m]
            both = base & np.isfinite(dm)
            pos = both & (di > 0)
            neg = both & (di < 0)
            rat[m]["pos"].append(dm[pos] / di[pos])
            rat[m]["neg"].append(dm[neg] / di[neg])
            rat[m]["absdiff"].append(np.abs(dm[both] - di[both]))
    nf = len(FRAMES)
    for (m, sel, label) in E.COMBOS:
        a = acc[(m, sel)]
        err = np.concatenate(a["err"]).astype(np.float64)
        hb = np.concatenate(a["hb"])
        pb = np.concatenate(a["pb"])
        inSi = np.concatenate(a["inSi"])
        subsets = [("all", np.ones(err.size, bool)), ("inSi", inSi), ("outSi", ~inSi)]
        subsets += [(f"hbin:{lab}", hb == k) for k, lab in enumerate(E.HB_LABEL)]
        subsets += [(f"pbin:{lab}", pb == k) for k, lab in enumerate(E.PB_LABEL)]
        for name, msk in subsets:
            st = E.err_stats(err[msk], gt)
            rows.append(dict(gt=gt, method=m, sel=sel, label=label, subset=name, n_per_frame=st["n"] / nf,
                             bias=st["bias"], median=st["median"], mae=st["mae"], rmse=st["rmse"], good10=st["good10"]))
    for m, r in rat.items():
        pos = np.concatenate(r["pos"])
        neg = np.concatenate(r["neg"])
        ad = np.concatenate(r["absdiff"])
        ratio_rows.append(dict(gt=gt, method=m, n_pos=pos.size, ratio_pos_median=float(np.median(pos)) if pos.size else np.nan,
                               n_neg=neg.size, ratio_neg_median=float(np.median(neg)) if neg.size else np.nan,
                               absdiff_median=float(np.median(ad)) if ad.size else np.nan,
                               frac_absdiff_le_0p3=float((ad <= 0.3).mean()) if ad.size else np.nan))
    base = [r for r in rows if r["gt"] == gt and r["method"] == "i" and r["subset"] == "all"][0]
    p(f"{gt} mm ({time.time() - t_start:.0f}s): (i) 有効 {base['n_per_frame']:.0f}/フレーム MAE {base['mae']:.1f} bias {base['bias']:+.1f} 中央値 {base['median']:+.1f}"
      f" | 選別 S_i {nsel['S_i'] / nf:.0f} S_mag {nsel['S_mag'] / nf:.0f} S_h0.05 {nsel['S_h0.05'] / nf:.0f} S_J0 {nsel['S_J0'] / nf:.0f}")
    for (m, sel, label) in E.COMBOS[1:]:
        r = [x for x in rows if x["gt"] == gt and x["method"] == m and x["sel"] == sel and x["subset"] == "all"][0]
        ro = [x for x in rows if x["gt"] == gt and x["method"] == m and x["sel"] == sel and x["subset"] == "outSi"][0]
        p(f"   {label:34s} 有効 {r['n_per_frame']:7.0f}（×{r['n_per_frame'] / base['n_per_frame']:.2f}） MAE {r['mae']:7.1f} bias {r['bias']:+7.1f} 中央値 {r['median']:+6.1f} 良好10% {r['good10']:.3f}"
          f" | S_i の外 {ro['n_per_frame']:6.0f} MAE {ro['mae']:7.1f} 中央値 {ro['median']:+6.1f}")
    for rr in [x for x in ratio_rows if x["gt"] == gt]:
        p(f"   比 d_{rr['method']}/d_i: pos {rr['ratio_pos_median']:.3f}（{rr['n_pos']}）neg {rr['ratio_neg_median']:.3f}（{rr['n_neg']}） |差| 中央値 {rr['absdiff_median']:.3f} px ≤0.3 px {rr['frac_absdiff_le_0p3']:.3f}")

with open(os.path.join(OUT, "v2_noise_rows.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(OUT, "v2_noise_ratio.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(ratio_rows[0].keys()))
    w.writeheader()
    w.writerows(ratio_rows)

# 群の要約（記録票 §6.2 と同じ: 距離ごとの値の単純平均・最大）
GROUPS = {"外挿（近）1000-1250": [1000, 1150, 1250], "校正範囲内 1450-3050": [1450, 1650, 1850, 2050, 2250, 2450, 2650, 2850, 3050],
          "外挿（遠）3450-4000": [3450, 3750, 4000]}
p("\n=== 群の要約（距離ごとの値の単純平均・|bias| と |中央値| は最大）===")
for g, ds in GROUPS.items():
    p(g)
    for (m, sel, label) in E.COMBOS:
        for sub in ("all", "outSi"):
            rr = [x for x in rows if x["gt"] in ds and x["method"] == m and x["sel"] == sel and x["subset"] == sub]
            n = np.mean([x["n_per_frame"] for x in rr])
            p(f"   {label:34s} {sub:5s} 有効 {n:7.0f}/フレーム MAE 平均 {np.nanmean([x['mae'] for x in rr]):7.1f} |bias| 最大 {np.nanmax([abs(x['bias']) for x in rr]):6.1f}"
              f" |中央値| 最大 {np.nanmax([abs(x['median']) for x in rr]):6.1f} 良好10% 平均 {np.nanmean([x['good10'] for x in rr]):.3f}")
p(f"\n合計 {time.time() - t_start:.0f}s")
log.close()
