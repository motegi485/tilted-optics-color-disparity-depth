"""verPhys v5: physB の合成の主要な結論 (a)(b)(c) を、自作の合成で再現できるか（2026-11-09）

目的:
  (a) 現行 C/Y で、強く横向きの筋を持つ被写体（端や曲がりのある横の帯）に対し、(i) の良好率は低く color-line は高いか。
      評価の分母（全開口の |∇G|>20 の画素）と、校正の比での割り算が結論を作っていないか。
      (i) の低さが「被覆（|∂x| の選別）」と「照合の特徴（|∂x|）」のどちらから来るかを、選別と特徴を入れ替えて分ける。
  (b) まっすぐで長い横エッジ（法線 90°）は、現行形状ではチャネル間のどの推定法でも取れないか。
      3 チャネルの縦方向の線広がり関数が同一か（推定法によらない確認）。
  (c) RGB 三扇形・CMY 三扇形・族 B 90° で横エッジが取れるか（color-line と PSF 合わせ）。
入力: なし
出力: verPhys_out/v5_log.txt, verPhys_out/v5_rows.csv
仮定: vp_lib.py 冒頭のとおり。192×192 px（中央 112×112 を評価）・σ=1.5・種 2 つ。
"""
import csv
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
ROWS = []


def log(s=""):
    print(s, flush=True)
    LOG.append(str(s))


H = W = 192
SS = 4
M = 24
CS = [3.0, 6.0, 12.0]
CANDS = np.arange(-24, 25, 1.0)
CANDS4 = np.arange(-20, 21, 1.0)
t0 = time.time()


def make(kind, seed):
    rng = np.random.default_rng(seed)
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
    if kind == "barcode90":
        return vp.subj_barcode(Hm, Wm, SS, rng, 90.0)
    if kind == "barcode0":
        return vp.subj_barcode(Hm, Wm, SS, rng, 0.0)
    if kind == "step90":
        return vp.subj_step(Hm, Wm, SS, 90.0)
    raise KeyError(kind)


def run_methods(u8, shape, q, methods):
    fn = vp.SHAPES[shape]
    out = {}
    for mt in methods:
        if mt == "i_cur":
            out[mt] = vp.est_i(u8, q, feat="dx", sel="dx")
        elif mt == "i_selgrad":
            out[mt] = vp.est_i(u8, q, feat="dx", sel="grad")
        elif mt == "i_lin":
            out[mt] = vp.est_i(u8, q, feat="lin", sel="grad")
        elif mt == "cl3":
            out[mt] = vp.est_colorline(u8, q, CANDS)
        elif mt == "iv":
            out[mt] = vp.est_psfmatch(u8, fn, CANDS4)
    return out


CAL = {}


def calib(shape, c, methods):
    """等方 1/f（種 2 つ）で k = median(ĉ/c)"""
    key = (shape, c)
    if key in CAL:
        return CAL[key]
    q = vp.qvec(vp.SHAPES[shape])
    acc = {m: [] for m in methods}
    for sd in (901, 902):
        subj = make("iso", sd)
        E, _, _ = vp.eval_mask(subj, c, SS, margin=M)
        u8 = vp.to8(vp.render(subj, vp.SHAPES[shape], c, SS, margin=M), 1.5, np.random.default_rng(sd + 7))
        for m, e in run_methods(u8, shape, q, methods).items():
            v = e[E]
            acc[m].append(v[np.isfinite(v)] / c)
    CAL[key] = {m: float(np.median(np.concatenate(v))) for m, v in acc.items()}
    return CAL[key]


def run_block(title, shapes, subjects, methods_for, seeds=(1, 2), hbins=False):
    log()
    log(f"##### {title}")
    for shape in shapes:
        q = vp.qvec(vp.SHAPES[shape])
        methods = methods_for(shape)
        for c in CS:
            k = calib(shape, c, methods)
            for kind in subjects:
                agg = {m: {"raw": [], "cal": []} for m in methods}
                hs = []
                hb = {m: {} for m in methods}
                for sd in seeds:
                    base = {"aniso_strong": 210, "bars_end": 300, "bars_bend": 310, "bars_both": 320,
                            "barcode90": 330, "barcode0": 340, "step90": 0, "iso": 100}[kind]
                    subj = make(kind, base + sd)
                    E, gx, gy = vp.eval_mask(subj, c, SS, margin=M)
                    h = vp.window_h(gx, gy)
                    hs.append(h[E])
                    u8 = vp.to8(vp.render(subj, vp.SHAPES[shape], c, SS, margin=M), 1.5, np.random.default_rng(sd + 50 + int(c)))
                    ests = run_methods(u8, shape, q, methods)
                    for m, e in ests.items():
                        agg[m]["raw"].append((e, E))
                        agg[m]["cal"].append((e / k[m], E))
                        if hbins:
                            for lo, hi in [(0, 0.005), (0.005, 0.02), (0.02, 0.05), (0.05, 1.01)]:
                                sel = E & (h >= lo) & (h < hi)
                                if sel.sum() == 0:
                                    continue
                                ev = vp.evaluate(e / k[m], c, sel)
                                hb[m].setdefault((lo, hi), []).append((ev["good"], ev["n"]))
                hcat = np.concatenate(hs)
                hdesc = f"評価画素 {hcat.size}・窓 h<0.02 {np.mean(hcat < 0.02):.2f}・h<0.05 {np.mean(hcat < 0.05):.2f}・h 中央値 {np.median(hcat):.3f}"
                parts = []
                for m in methods:
                    r = {}
                    for mode in ("raw", "cal"):
                        evs = [vp.evaluate(e, c, E) for e, E in agg[m][mode]]
                        n = sum(x["n"] for x in evs)
                        r[mode] = dict(good=sum(x["good"] * x["n"] for x in evs) / n, cov=sum(x["cov"] * x["n"] for x in evs) / n,
                                       good_own=np.nanmean([x["good_own"] for x in evs]), bias=np.nanmean([x["bias"] for x in evs]),
                                       sd=np.nanmean([x["sd"] for x in evs]))
                        ROWS.append(dict(block=title, shape=shape, subject=kind, c=c, method=m, mode=mode, k=k[m], **r[mode]))
                    parts.append(f"{m}: 良好 raw {r['raw']['good']:.2f}／校正 {r['cal']['good']:.2f}（k={k[m]:.3f}）被覆 {r['raw']['cov']:.2f} "
                                 f"自前の画素での良好 raw {r['raw']['good_own']:.2f}／校正 {r['cal']['good_own']:.2f} 偏り {r['raw']['bias']:+.2f} ばらつき {r['raw']['sd']:.2f}")
                log(f"--- {shape} c={c:g} {kind}（{hdesc}）")
                for p in parts:
                    log("    " + p)
                if hbins:
                    for m in methods:
                        s = []
                        for (lo, hi), lst in sorted(hb[m].items()):
                            n = sum(x[1] for x in lst)
                            gsum = sum(x[0] * x[1] for x in lst)
                            s.append(f"h∈[{lo},{hi}) n={n} 良好 {gsum / n:.2f}")
                        log(f"      窓 h 別（校正後） {m}: " + " ; ".join(s))
            print(f"  [{shape} c={c:g} 済 {time.time() - t0:.0f}s]", flush=True)


# ---------------------------------------------------------------------------
# (b) の推定法によらない確認: 横の段差エッジでの 3 チャネルの縦の断面
# ---------------------------------------------------------------------------
log("##### (b-0) 現行 C/Y・横の段差エッジ（雑音なし）: 3 チャネルの縦の断面は同一か")
for c in [-12.0, -6.0, -3.0, 3.0, 6.0, 12.0]:
    subj = make("step90", 0)
    lin = vp.render(subj, vp.ap_current, c, SS, margin=M)
    col = W // 2
    pr = {k: lin[k][:, col] for k in vp.CH}
    d_rg = np.max(np.abs(pr["R"] - pr["G"]))
    d_bg = np.max(np.abs(pr["B"] - pr["G"]))
    # 横方向にも一様か（どの列でも同じ）
    d_cols = max(np.max(np.abs(lin[k] - lin[k][:, [col]])) for k in vp.CH)
    # 10–90 % の幅（ぼけの大きさの手掛かり）
    g = pr["G"]
    lo, hi = g.min(), g.max()
    yy = np.arange(len(g))
    i10 = np.interp(lo + 0.1 * (hi - lo), g if g[0] < g[-1] else g[::-1], yy if g[0] < g[-1] else yy[::-1])
    i90 = np.interp(lo + 0.9 * (hi - lo), g if g[0] < g[-1] else g[::-1], yy if g[0] < g[-1] else yy[::-1])
    log(f"  c={c:+.0f}: max|R−G|={d_rg:.2e} max|B−G|={d_bg:.2e}（線形の明るさ）列による差 {d_cols:.1e}  G の 10–90% 幅 {abs(i90 - i10):.2f} px")
log("  比較: c=+6 と c=−6 の G の断面の差 " +
    f"{np.max(np.abs(vp.render(make('step90', 0), vp.ap_current, 6.0, SS, margin=M)['G'] - vp.render(make('step90', 0), vp.ap_current, -6.0, SS, margin=M)['G'])):.2e}")

# ---------------------------------------------------------------------------
# (a) 横向きの筋の強い被写体・現行 C/Y
# ---------------------------------------------------------------------------
run_block("(a) 現行 C/Y・横向きの筋の強い被写体", ["current"],
          ["aniso_strong", "bars_end", "bars_bend", "bars_both"],
          lambda s: ["i_cur", "i_selgrad", "i_lin", "cl3", "iv"], hbins=True)

# ---------------------------------------------------------------------------
# (b) まっすぐで長い横エッジ・現行 C/Y
# ---------------------------------------------------------------------------
run_block("(b) 現行 C/Y・まっすぐな横エッジ", ["current"], ["barcode90", "step90"],
          lambda s: ["i_cur", "i_selgrad", "i_lin", "cl3", "iv"])

# ---------------------------------------------------------------------------
# (c) 全方向形状で横エッジ（と縦エッジ）
# ---------------------------------------------------------------------------
run_block("(c) 全方向形状・横エッジと縦エッジ", ["rgb3", "cmy3", "famB90"], ["barcode90", "step90", "barcode0"],
          lambda s: ["cl3", "iv"])

with open(os.path.join(OUT, "v5_log.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(LOG) + "\n")
with open(os.path.join(OUT, "v5_rows.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ROWS[0].keys()))
    w.writeheader()
    w.writerows(ROWS)
print(f"done {time.time() - t0:.0f}s")
