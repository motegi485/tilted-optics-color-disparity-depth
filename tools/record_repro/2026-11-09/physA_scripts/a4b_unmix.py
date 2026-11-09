# -*- coding: utf-8 -*-
"""実フィルタの近似で縮んだ実効重心が、画素ごとの 3×3 の色変換（混色の逆変換）でどこまで戻るかと、その雑音の代償。

背景: [Bando2008] Appendix A は混色行列を測って逆行列で補正し、[Paramonov2016] §2.2 は RGB を CYX 空間へ変換してから視差を求める。
入力: なし（a4_spectral.py の仮定のフィルタとセンサ感度を使う）
出力: physA_out/a4b_unmix.csv、標準出力
方法: 設計に現れるフィルタ種類 f（不透明を除く）の実効透過率 w_f (3,) と理想の透過 t_f (3,) について、
      面積で重み付けした最小二乗 X = argmin Σ_f A_f ||X w_f − t_f||² を解き、補正後の重み X w_f で重心・指標を出す。
      フィルタ種類が 3 以下で w_f が一次独立なら補正は厳密（理想の幾何に戻る）。雑音の増幅は各行のノルム ||X[c,:]||
      （測定チャネルの雑音が独立で等しいと仮定。光子雑音の信号依存は無視）。
仮定: 物体は分光的に平坦（灰色）。色のある物体では画素ごとに実効透過率が変わり、補正は厳密でなくなる。
      幾何は aperture_lib.py の仮定（回折なし・ケラレなし・アオリなし）。濃度勾配の設計は除く。
"""
import csv
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

import aperture_lib as al
import catalog as ct
import a4_spectral as a4

OUT = os.path.join(os.path.dirname(HERE), "physA_out")


def main():
    grid = al.Grid(700)
    rows = []
    skip = {"par_cmy_grad", "par_cy_grad"}
    print("形状 | ケース | 種類数 | 補正後: 縦 / 最悪 / Fisher最小（理想比） | 雑音の増幅 R,G,B")
    for key, func, label, src in ct.DESIGNS:
        if key in skip:
            continue
        T = func(grid.x, grid.y)
        inside = al.in_disk(grid.x, grid.y)
        b = np.rint(T).astype(int)
        codes = b[:, 0] * 4 + b[:, 1] * 2 + b[:, 2]
        present = sorted(set(codes[inside].tolist()) - {0})
        area = {c: (codes[inside] == c).sum() for c in present}
        ideal_e = al.evaluate(func, grid, lsf_step_deg=6.0)
        for cn, case in a4.CASES.items():
            if case is None:
                continue
            Wt = a4.eff_weights(case)
            lut = {}
            for code, idx in a4.CODE2IDX.items():
                lut[code[0] * 4 + code[1] * 2 + code[2]] = idx
            Wf = np.array([Wt[lut[c]] for c in present])  # (k,3) 実効
            Tf = np.array([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in present], float)  # (k,3) 理想
            sw = np.sqrt(np.array([area[c] for c in present], float))
            # X Wf^T ≈ Tf^T → Wf X^T ≈ Tf（面積の重み付き最小二乗）
            Xt, *_ = np.linalg.lstsq(Wf * sw[:, None], Tf * sw[:, None], rcond=None)
            X = Xt.T
            corrected = Wt @ X.T  # 全種類の補正後の重み (8,3)
            gain = np.linalg.norm(X, axis=1)
            resid = np.abs(Wf @ X.T - Tf).max()

            def fcorr(x, y, func=func, corrected=corrected):
                TT = func(x, y)
                ins = al.in_disk(x, y)
                bb = np.rint(TT).astype(int)
                idx = np.array([lut[v] for v in (bb[:, 0] * 4 + bb[:, 1] * 2 + bb[:, 2])])
                out = corrected[idx]
                out[~ins] = 0.0
                return out

            e = al.evaluate(fcorr, grid, lsf_step_deg=6.0)
            raw = al.evaluate(a4.realize(func, Wt), grid, lsf_step_deg=6.0)
            r = dict(key=key, label=label, case=cn, n_types=len(present), exact=bool(resid < 1e-6), resid=resid,
                     vert_raw=raw["vert"], worst_raw=raw["worst"], vert_corr=e["vert"], worst_corr=e["worst"],
                     fisher_corr=e["fisher_min"], vert_ideal=ideal_e["vert"], worst_ideal=ideal_e["worst"],
                     gain_R=gain[0], gain_G=gain[1], gain_B=gain[2])
            rows.append(r)
            if cn in ("基準(Y500/C580,幅7)", "重なり広(Y480/C610)"):
                print(f"{label} | {cn} | {len(present)}{'（厳密）' if r['exact'] else ''} | 縦 {raw['vert']:.3f}→{e['vert']:.3f}（理想 {ideal_e['vert']:.3f}）"
                      f" 最悪 {raw['worst']:.3f}→{e['worst']:.3f}（理想 {ideal_e['worst']:.3f}） | 増幅 {gain[0]:.2f},{gain[1]:.2f},{gain[2]:.2f}")
    with open(os.path.join(OUT, "a4b_unmix.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
