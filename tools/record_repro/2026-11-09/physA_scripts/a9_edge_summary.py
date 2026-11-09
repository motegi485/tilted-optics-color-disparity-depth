# -*- coding: utf-8 -*-
"""a5_edge_orientation.py の出力 CSV から、メモ用のグループ別要約を作る（計算のやり直しはしない）。

目的: 既存の撮影画像で、現行のエッジ選別（R の |∂R/∂x|>20）が、向きを問わない |∇R|>20 の画素のうち何割を残すか、
      エッジの法線と水平のなす角 φ の分布（左右分割の基線は cosφ に比例）を、場面のグループごとにまとめる。
入力: physA_out/a5_edge_stats.csv、a5_edge_hist.csv、a5_design_scene_weighted.csv（argv[1] で physA_out を指定可）
出力: physA_out/a9_edge_summary.csv、標準出力
仮定: a5 と同じ（528×352 に縮小・参照実装の走査範囲・現行リグ θ=30° の C/Y 左右を通した画像）。
      グループ内の平均はセッション等重み。
"""
import csv
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "physA_out")


def read(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    st = read(os.path.join(OUT, "a5_edge_stats.csv"))
    hist = read(os.path.join(OUT, "a5_edge_hist.csv"))
    groups = []
    for r in st:
        if r["group"] not in groups:
            groups.append(r["group"])
    rows = []
    print("グループ | ch | セッション数 | 現行の選別/向き不問（平均・範囲） | φ>45° | φ>60°（範囲） | φ>75° | cosφ 平均 | 1 セッションあたり画素数（現行・向き不問）")
    for g in groups:
        for ch in ("R", "G"):
            rr = [r for r in st if r["group"] == g and r["channel"] == ch]
            if not rr:
                continue
            f = lambda k: np.array([float(r[k]) for r in rr])
            ratio = f("ratio_current_over_any")
            p60 = f("frac_phi_gt60")
            row = dict(group=g, channel=ch, n_sessions=len(rr), ratio_mean=ratio.mean(), ratio_min=ratio.min(),
                       ratio_max=ratio.max(), phi45=f("frac_phi_gt45").mean(), phi60=p60.mean(), phi60_min=p60.min(),
                       phi60_max=p60.max(), phi75=f("frac_phi_gt75").mean(), cos_mean=f("mean_cos").mean(),
                       cos_w=f("mean_cos_weighted").mean(), n_current=f("n_current").mean(), n_any=f("n_any").mean())
            rows.append(row)
            print(f"{g} | {ch} | {len(rr)} | {ratio.mean():.3f}（{ratio.min():.3f}–{ratio.max():.3f}） | {row['phi45']:.3f} | "
                  f"{p60.mean():.3f}（{p60.min():.3f}–{p60.max():.3f}） | {row['phi75']:.3f} | {row['cos_mean']:.3f} | "
                  f"{row['n_current']:.0f}・{row['n_any']:.0f}")
    with open(os.path.join(OUT, "a9_edge_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # 角度ヒストグラム（R・セッション等重み）を 15° 幅にまとめる
    print("\nR の |∇R|>20 の画素の φ 分布（15° 幅・セッション等重み、0°=縦エッジ・90°=横エッジ）")
    for g in groups:
        sess = sorted(set(r["session"] for r in hist if r["group"] == g and r["channel"] == "R"))
        acc = np.zeros(6)
        for s in sess:
            h = np.array([float(r["count"]) for r in hist if r["group"] == g and r["channel"] == "R" and r["session"] == s])
            h = h / max(h.sum(), 1)
            acc += h.reshape(6, 3).sum(1)
        acc /= max(len(sess), 1)
        print(f"  {g}: " + " ".join(f"{15 * i:2d}–{15 * (i + 1):2d}°:{v:.3f}" for i, v in enumerate(acc)))
    # 場面のエッジ向きで平均した基線
    p = os.path.join(OUT, "a5_design_scene_weighted.csv")
    if os.path.exists(p):
        dw = read(p)
        keys = ["cy_lr", "cy_lr_45", "rgb3", "cmy3", "ycg3", "quad", "wheel6", "ywcg4_v", "cy_m90", "kaneko_g", "bando", "mca"]
        print("\n場面のエッジ向きで平均した基線（R の |∇R|>20 の画素で等重み・鏡像のうち低い方）と、基線<0.1 の画素の割合")
        for g in groups:
            sub = {r["key"]: r for r in dw if r["group"] == g}
            if not sub:
                continue
            print(f"  {g}: " + "; ".join(f"{sub[k]['label']} {float(sub[k]['mean_baseline']):.3f}/{float(sub[k]['frac_below_0p1']):.2f}"
                                       for k in keys if k in sub))


if __name__ == "__main__":
    main()
