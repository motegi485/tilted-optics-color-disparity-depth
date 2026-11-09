"""verPhys v10: 現行 C/Y＋color-line の良好率が c によって 0〜1 に振れる現象（v8 D）は、強い異方性の 1/f だけか（2026-11-09）

目的: v8 D で、現行 C/Y の color-line は強い異方性の 1/f に対し c=3/6 で良好、c=4/5/10 で 0.00〜0.17 だった。
      physB が評価した c=3/6/12 だけでは見えない振れか、被写体に固有かを、端・曲がりのある横帯と等方 1/f で見る。
      種ごとの中央値の偏りも出す（v8 D は 2 種の中央値の平均で、種で符号が割れると打ち消し合う）。
入力: なし（vp_lib.py を import）
出力: verPhys_out/v10_log.txt（UTF-8）
仮定: vp_lib.py と同じ。校正の比は各 c で等方 1/f（種 901・902）から求める。
"""
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import vp_lib as vp

OUT = os.path.join(os.path.dirname(HERE), "verPhys_out")
LOG = []
t0 = time.time()
H = W = 192
SS = 4
M = 24
CANDS = np.arange(-24, 25, 1.0)


def log(s=""):
    print(s, flush=True)
    LOG.append(str(s))


def make(kind, rng):
    Hm, Wm = H + 2 * M, W + 2 * M
    if kind == "iso":
        return vp.subj_1overf(Hm, Wm, SS, rng)
    if kind == "aniso_strong":
        return vp.subj_1overf(Hm, Wm, SS, rng, aniso_deg=4.0, floor=0.03)
    if kind == "bars_end":
        return vp.subj_bars(Hm, Wm, SS, rng, finite=True, bend=False)
    if kind == "bars_bend":
        return vp.subj_bars(Hm, Wm, SS, rng, finite=False, bend=True)
    if kind == "bars_both":
        return vp.subj_bars(Hm, Wm, SS, rng, finite=True, bend=True)
    raise KeyError(kind)


BASE = {"iso": 100, "aniso_strong": 210, "bars_end": 300, "bars_bend": 310, "bars_both": 320}
log("=== 現行 C/Y・color-line（3 チャネル）。良好率 raw／校正、被覆、種ごとの中央値の偏り [%]")
for shape in ("current", "rgb3"):
    fn = vp.SHAPES[shape]
    q = vp.qvec(fn)
    for c in (4.0, 5.0, 8.0, 10.0):
        acc = []
        for sd in (901, 902):
            subj = make("iso", np.random.default_rng(sd))
            E, _, _ = vp.eval_mask(subj, c, SS, margin=M)
            u8 = vp.to8(vp.render(subj, fn, c, SS, margin=M), 1.5, np.random.default_rng(sd + 7))
            v = vp.est_colorline(u8, q, CANDS)[E]
            acc.append(v[np.isfinite(v)] / c)
        k = float(np.median(np.concatenate(acc)))
        for kind in ("iso", "aniso_strong", "bars_end", "bars_bend", "bars_both"):
            if shape == "rgb3" and kind in ("iso", "bars_end"):
                continue
            raw, cal, med = [], [], []
            for sd in (1, 2):
                subj = make(kind, np.random.default_rng(BASE[kind] + sd))
                E, _, _ = vp.eval_mask(subj, c, SS, margin=M)
                u8 = vp.to8(vp.render(subj, fn, c, SS, margin=M), 1.5, np.random.default_rng(sd + 50 + int(c)))
                e = vp.est_colorline(u8, q, CANDS)
                r = vp.evaluate(e, c, E)
                raw.append(r)
                cal.append(vp.evaluate(e / k, c, E))
                med.append(100 * r["bias"] / c)
            n = sum(x["n"] for x in raw)
            gr = sum(x["good"] * x["n"] for x in raw) / n
            gc = sum(x["good"] * x["n"] for x in cal) / n
            cov = sum(x["cov"] * x["n"] for x in raw) / n
            log(f"  {shape:7s} c={c:4.1f} k={k:.3f} {kind:12s}: 良好 {gr:.2f}／{gc:.2f} 被覆 {cov:.2f} 偏り（種 1, 2）{med[0]:+.0f}%, {med[1]:+.0f}%  ({time.time() - t0:.0f}s)")

with open(os.path.join(OUT, "v10_log.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(LOG) + "\n")
