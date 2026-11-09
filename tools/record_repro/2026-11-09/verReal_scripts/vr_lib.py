# -*- coding: utf-8 -*-
"""verReal: 既存の実撮影データで推定法を比べるための共通ライブラリ（読み取りのみ・開発リポのファイルは変更しない）。

処理経路（開発リポで確認 2026-11-09）:
  raw 1280×720 PNG → tools.analysis_common.preprocess_to（中央 1080×720 → INTER_AREA で 528×352）
  → (i) core.disparity.calcDisparityByZNCC（Rust・K=25・S=41・T=20）→ apply_gates（gate1 |d|>20・推論帯 0.45<|d|<5）
  → depth_of（参照式 z = v/(y − a·d − u)・live 係数 params/fit_result.csv・pos/neg 枝）。

推定法（視差 d はすべて (i) と同じ「R の位置 − B の位置」[px]・同じ候補 d = −20..20・端 2 ビンを除いた最良・厳密な局所極値・放物線）:
  i      : Rust 実装そのもの（R の |∂x|>20 で選ぶ・R/B の |∂x| 画像の ZNCC）
  i_all  : Rust 実装を T=0 で呼んだもの（照合は (i) と同じ。画素の選び方だけを後で変えるため。|∂x|=0 の画素は NaN）
  gx_box : (i) の照合を box filter で書き直したもの（自作の照合の機構が Rust と一致するかの確認用）
  gm     : 照合の特徴を |∇|（Sobel の大きさ）に変えた ZNCC（横方向の探索）
  rb     : 照合の特徴を明るさ（線形化した R と B）に変えた ZNCC（2 チャネル・横方向の探索）
  cl50   : color-line。R を x + d/2、B を x − d/2 から取り G は固定、線形の明るさの窓内相関行列の det を最小化（課題の (ii)）
  cl27   : 同じで G を R から B へ 0.27 の位置に置いたもの（実測 d_RG/d_RB≈0.27・調査メモ_2026-11-07 earlier1_phys.md Q3 の pos 枝）
  補間: color-line は 3 チャネルとも 1/4 画素ずらした位置を三次補間で取る（偶奇の候補で補間のぼけ方が変わる偏りを避ける）。
画素の選び方:
  S_i    : 現行（R の convertScaleAbs(Sobel_x) > 20）
  S_mag  : 向き不問（R の |∇R| を四捨五入・飽和した値 > 20。physA a5 と同じ）
  S_h<τ> : S_mag かつ 窓（25×25）の構造テンソルの h = Txx/(Txx+Tyy) ≥ τ（R）
  S_J0   : S_mag かつ 窓の Txx ≥ 現行の選別が受け入れた窓の Txx の 10 パーセンタイル（physA a11 の J ≥ J0 と同じ。フレームごと）
走査範囲は参照実装と同じ y∈[12, 340)・x∈[33, 495)。
"""
import os
import sys

sys.dont_write_bytecode = True
ROOT = "C:/Users/t5424001/dev/tilted_optics_color_disparity_depth"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import numpy as np

import config
from core.disparity import calcDisparityByZNCC
from core.params import load_params
from tools.analysis_common import apply_gates, depth_of, preprocess_to

K = config.KERNEL_SIZE          # 25
S = config.SEARCH_RANGE_X       # 41
TH = config.EDGE_TH             # 20
KH = (K - 1) // 2               # 12
SH = (S - 1) // 2               # 20
W, H = config.IMAGE_SIZE        # 528, 352
BAND = (config.INFER_BAND_LOW, config.INFER_BAND_HIGH)
PARAMS = load_params(config.PARAMS_PATH)
CAP = os.path.join(ROOT, "data", "captures")
OUTP = os.path.join(ROOT, "outputs")
DS = np.arange(-SH, SH + 1)     # 候補 d（index m ↔ d = m − 20）

ROI = np.zeros((H, W), bool)
ROI[KH:H - KH, SH + KH + 1:W - SH - KH - 1] = True

_MAPY = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
_X = np.tile(np.arange(W, dtype=np.float32)[None, :], (H, 1))


def frame_path(session, idx):
    return os.path.join(CAP, session, f"{idx:06d}.png")


def load_proc(session, idx):
    raw = cv2.imread(frame_path(session, idx), cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(frame_path(session, idx))
    return preprocess_to(raw, config.IMAGE_SIZE)


def list_indices(session):
    return sorted(int(f[:-4]) for f in os.listdir(os.path.join(CAP, session)) if f.endswith(".png") and f[:-4].isdigit())


def rust_disp(proc, th=TH):
    blue, _, red = cv2.split(proc)
    return calcDisparityByZNCC(red, blue, K, S, th)


def to_depth(disp):
    """gate1＋推論帯 → 参照式（live 係数）→ gate2（0<z<50000）。推論帯を通った後の深度（NaN は無効）。"""
    z = depth_of(apply_gates(disp, disp_max=config.DISP_MAX, band=BAND), PARAMS)
    z = z.astype(np.float64)
    z[~((z > config.DEPTH_MIN) & (z < config.DEPTH_MAX))] = np.nan
    return z


def banded(disp):
    return apply_gates(disp, disp_max=config.DISP_MAX, band=BAND)


def srgb_lin(u8):
    x = u8.astype(np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def box(a):
    return cv2.boxFilter(a, cv2.CV_64F, (K, K), normalize=True, borderType=cv2.BORDER_REFLECT)


def feats(proc):
    """選別用の勾配・窓の向き。"""
    out = {}
    for name, ch in (("R", 2), ("G", 1)):
        c = proc[:, :, ch]
        gx = cv2.Sobel(c, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(c, cv2.CV_32F, 0, 1)
        Txx = box((gx * gx).astype(np.float64))
        Tyy = box((gy * gy).astype(np.float64))
        tr = Txx + Tyy
        out[f"h_{name}"] = np.where(tr > 0, Txx / np.maximum(tr, 1e-12), np.nan)
        out[f"Txx_{name}"] = Txx
        out[f"phi_{name}"] = np.degrees(np.arctan2(np.abs(gy), np.abs(gx)))
        out[f"gx_{name}"] = gx
        out[f"gy_{name}"] = gy
    gxR, gyR = out["gx_R"], out["gy_R"]
    ex = cv2.convertScaleAbs(gxR)
    mag_u8 = np.clip(np.rint(np.hypot(gxR, gyR)), 0, 255)
    out["S_i"] = (ex > TH) & ROI
    out["S_mag"] = (mag_u8 > TH) & ROI
    for tau in (0.02, 0.05, 0.1, 0.2):
        out[f"S_h{tau:g}"] = out["S_mag"] & (out["h_R"] >= tau)
    if out["S_i"].any():
        j0 = np.percentile(out["Txx_R"][out["S_i"]], 10)
        out["S_J0"] = out["S_mag"] & (out["Txx_R"] >= j0)
    else:
        out["S_J0"] = np.zeros_like(out["S_mag"])
    return out


def _peak_min(C):
    """C: (41,H,W) コスト（小さいほど良い・NaN 可）。端 2 ビンを除いた最小 → 厳密な局所最小 → 放物線。"""
    inner = C[1:-1]
    allnan = np.all(np.isnan(inner), axis=0)
    filled = np.where(np.isnan(inner), np.inf, inner)
    m = np.argmin(filled, axis=0) + 1
    yy, xx = np.indices(m.shape)
    c0 = C[m, yy, xx]
    cm = C[m - 1, yy, xx]
    cp = C[m + 1, yy, xx]
    ok = (~allnan) & np.isfinite(c0) & np.isfinite(cm) & np.isfinite(cp) & (c0 < cm) & (c0 < cp)
    den = cm - 2 * c0 + cp
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where(den > 0, (cm - cp) / (2 * den), 0.0)
    delta = np.where(np.abs(delta) <= 1.0, delta, 0.0)
    d = (m - SH) + delta
    out = np.full(m.shape, np.nan, np.float32)
    out[ok] = d[ok]
    out[~ROI] = np.nan
    return out, ok


def _shift_int(a, d):
    """b(x) = a(x − d)（整数・はみ出しは端の値）。"""
    if d == 0:
        return a.copy()
    out = np.empty_like(a)
    if d > 0:
        out[:, d:] = a[:, :-d]
        out[:, :d] = a[:, :1]
    else:
        out[:, :d] = a[:, -d:]
        out[:, d:] = a[:, -1:]
    return out


def zncc_hshift(Fr, Fb, eps=1e-9):
    """R の窓（中心 x）と B の窓（中心 x − d）の ZNCC を全候補で。コスト = −ZNCC。"""
    Fr = Fr.astype(np.float64)
    Fb = Fb.astype(np.float64)
    mR = box(Fr)
    vR = box(Fr * Fr) - mR * mR
    mB0 = box(Fb)
    vB0 = box(Fb * Fb) - mB0 * mB0
    C = np.full((S, H, W), np.nan, np.float32)
    for m, d in enumerate(DS):
        Bs = _shift_int(Fb, int(d))
        mB = _shift_int(mB0, int(d))
        vB = _shift_int(vB0, int(d))
        cov = box(Fr * Bs) - mR * mB
        den = np.sqrt(np.maximum(vR, 0) * np.maximum(vB, 0))
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where((vR > eps) & (vB > eps), cov / den, np.nan)
        C[m] = (-z).astype(np.float32)
    return C


def disp_gx_box(proc):
    blue, _, red = cv2.split(proc)
    er = cv2.convertScaleAbs(cv2.Sobel(red, cv2.CV_32F, 1, 0))
    eb = cv2.convertScaleAbs(cv2.Sobel(blue, cv2.CV_32F, 1, 0))
    return _peak_min(zncc_hshift(er, eb))[0]


def disp_gm(proc):
    blue, _, red = cv2.split(proc)

    def mag(c):
        gx = cv2.Sobel(c, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(c, cv2.CV_32F, 0, 1)
        return np.clip(np.rint(np.hypot(gx, gy)), 0, 255)
    return _peak_min(zncc_hshift(mag(red), mag(blue)))[0]


def disp_rb(proc):
    return _peak_min(zncc_hshift(srgb_lin(proc[:, :, 2]), srgb_lin(proc[:, :, 0]), eps=1e-10))[0]


def _remap(a, s):
    return cv2.remap(a, _X + np.float32(s), _MAPY, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def colorline_cost(proc, g=0.5, eps=1e-10):
    """R を x + g·d、B を x − (1−g)·d、G を x から取り（いずれも +1/4 画素）、窓内の 3×3 相関行列の det を返す（41,H,W）。"""
    R = srgb_lin(proc[:, :, 2]).astype(np.float32)
    Gc = srgb_lin(proc[:, :, 1]).astype(np.float32)
    B = srgb_lin(proc[:, :, 0]).astype(np.float32)
    off = 0.25
    Gs = _remap(Gc, off).astype(np.float64)
    mG = box(Gs)
    vG = box(Gs * Gs) - mG * mG
    C = np.full((S, H, W), np.nan, np.float32)
    for m, d in enumerate(DS):
        Rs = _remap(R, g * d + off).astype(np.float64)
        Bs = _remap(B, -(1.0 - g) * d + off).astype(np.float64)
        mR = box(Rs)
        mB = box(Bs)
        vR = box(Rs * Rs) - mR * mR
        vB = box(Bs * Bs) - mB * mB
        cRG = box(Rs * Gs) - mR * mG
        cBG = box(Bs * Gs) - mB * mG
        cRB = box(Rs * Bs) - mR * mB
        ok = (vR > eps) & (vG > eps) & (vB > eps)
        with np.errstate(invalid="ignore", divide="ignore"):
            sR, sG, sB = np.sqrt(np.maximum(vR, 0)), np.sqrt(np.maximum(vG, 0)), np.sqrt(np.maximum(vB, 0))
            rg = cRG / (sR * sG)
            rb = cRB / (sR * sB)
            gb = cBG / (sG * sB)
            det = 1 - rg * rg - rb * rb - gb * gb + 2 * rg * rb * gb
        C[m] = np.where(ok, det, np.nan).astype(np.float32)
    return C


def disp_cl(proc, g=0.5):
    return _peak_min(colorline_cost(proc, g))[0]


def disp_cl_branch(proc, g_pos=0.27, g_neg=0.10):
    """枝ごとに G の位置を変えた color-line（d>0 の候補は g_pos、d<0 は g_neg、d=0 は共通）。
    g は調査メモ_2026-11-07 earlier1_phys.md Q3 の実測 d_RG/d_RB（pos 0.26〜0.28・neg 0.05〜0.14・校正 raw で測ったもの）。"""
    Cp = colorline_cost(proc, g_pos)
    Cn = colorline_cost(proc, g_neg)
    C = np.where((DS > 0)[:, None, None], Cp, Cn)
    return _peak_min(C)[0]
