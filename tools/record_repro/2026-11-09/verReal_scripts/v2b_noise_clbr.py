# -*- coding: utf-8 -*-
"""(a) の補足: 枝ごとに G の位置を変えた color-line（clbr）と、G を使わない明るさの ZNCC（rb）を、ノイズ板で (i) と比べる。
(ii) の G 中央固定が neg 枝で d≈0 に潰れた原因が「G の位置の非対称」かを確かめる。読み取りのみ・非公式の再解析。"""
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
log = open(os.path.join(OUT, "v2b_noise_clbr.txt"), "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")
    log.flush()


rows = []
t0 = time.time()
for gt in DISTS:
    ses = f"b5_dist_{gt}"
    acc = {k: dict(err=[], inSi=[]) for k in ("i", "clbr_Si", "clbr_mag", "clbr_J0", "rb_J0")}
    rat = dict(pos=[], neg=[], ad=[])
    for idx in FRAMES:
        proc = L.load_proc(ses, idx)
        f = L.feats(proc)
        di_raw = L.rust_disp(proc, L.TH)
        dc = L.disp_cl_branch(proc)
        drb = L.disp_rb(proc)
        maps = {"i": di_raw, "clbr_Si": np.where(f["S_i"], dc, np.nan), "clbr_mag": np.where(f["S_mag"], dc, np.nan),
                "clbr_J0": np.where(f["S_J0"], dc, np.nan), "rb_J0": np.where(f["S_J0"], drb, np.nan)}
        for k, dm in maps.items():
            z = L.to_depth(dm.astype(np.float32))
            v = np.isfinite(z)
            acc[k]["err"].append(z[v] - gt)
            acc[k]["inSi"].append(f["S_i"][v])
        di = L.banded(di_raw)
        both = np.isfinite(di) & np.isfinite(dc)
        rat["pos"].append(dc[both & (di > 0)] / di[both & (di > 0)])
        rat["neg"].append(dc[both & (di < 0)] / di[both & (di < 0)])
        rat["ad"].append(np.abs(dc[both] - di[both]))
    nf = len(FRAMES)
    base_n = None
    line = f"{gt} mm ({time.time() - t0:.0f}s):"
    for k, a in acc.items():
        err = np.concatenate(a["err"])
        inSi = np.concatenate(a["inSi"])
        st = E.err_stats(err, gt)
        sto = E.err_stats(err[~inSi], gt)
        if k == "i":
            base_n = st["n"]
        rows.append(dict(gt=gt, combo=k, n_per_frame=st["n"] / nf, gain=st["n"] / base_n, mae=st["mae"], bias=st["bias"],
                         median=st["median"], good10=st["good10"], out_n_per_frame=sto["n"] / nf, out_mae=sto["mae"], out_median=sto["median"]))
        line += f" | {k} {st['n'] / nf:.0f}（×{st['n'] / base_n:.2f}）MAE {st['mae']:.1f} 中央値 {st['median']:+.1f}"
    pos, neg, ad = (np.concatenate(rat[k]) for k in ("pos", "neg", "ad"))
    rows[-1].update()
    p(line)
    p(f"   比 d_clbr/d_i: pos {np.median(pos) if pos.size else np.nan:.3f}（{pos.size}） neg {np.median(neg) if neg.size else np.nan:.3f}（{neg.size}） |差| 中央値 {np.median(ad):.3f} px")
    for r in rows[-len(acc):]:
        r.update(ratio_pos=float(np.median(pos)) if pos.size else np.nan, ratio_neg=float(np.median(neg)) if neg.size else np.nan)

with open(os.path.join(OUT, "v2b_noise_clbr.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
GROUPS = {"近 1000-1250": [1000, 1150, 1250], "校正範囲内 1450-3050": [1450, 1650, 1850, 2050, 2250, 2450, 2650, 2850, 3050],
          "遠 3450-4000": [3450, 3750, 4000]}
p("\n=== 群の要約（距離ごとの単純平均）===")
for g, ds in GROUPS.items():
    for k in acc:
        rr = [r for r in rows if r["gt"] in ds and r["combo"] == k]
        p(f"{g:20s} {k:9s} 有効 {np.mean([r['n_per_frame'] for r in rr]):7.0f}（×{np.mean([r['gain'] for r in rr]):.2f}） MAE {np.mean([r['mae'] for r in rr]):7.1f}"
          f" |中央値| 最大 {np.max([abs(r['median']) for r in rr]):6.1f} 良好10% {np.mean([r['good10'] for r in rr]):.3f}"
          f" | S_i の外 {np.mean([r['out_n_per_frame'] for r in rr]):6.0f} MAE {np.nanmean([r['out_mae'] for r in rr]):7.1f}")
log.close()
