"""分割カラーフィルタ開口の PSF で物体を合成する道具。

チャネル k の画像 I_k = Σ_成分 [ W_kY (mask ⊛ PSF_Y) + W_kC (mask ⊛ PSF_C) ]
PSF_Y, PSF_C はそれぞれ半円（面積比 0.5 ずつ）。合焦面より手前（c>0）では Y 側の像が右へずれる。
大気・レンズ収差・回折・ケラレは入れない。
"""
import sys

sys.dont_write_bytecode = True
import cv2
import numpy as np
from scipy.signal import fftconvolve

import spectral as sp


def half_kernels(c, cy=None, ss=4):
    """横幅 |c| px・縦幅 cy px の楕円を左右に分けた核（ss 倍の格子）。合計は各 0.5。"""
    cy = abs(c) if cy is None else abs(cy)
    rx = abs(c) * ss / 2.0
    ry = cy * ss / 2.0
    if rx < 0.5 or ry < 0.5:
        k = np.array([[0.5]])
        return k, k.copy()
    nx, ny = int(np.ceil(rx)) + 1, int(np.ceil(ry)) + 1
    yy, xx = np.mgrid[-ny : ny + 1, -nx : nx + 1].astype(float)
    disc = (xx / rx) ** 2 + (yy / ry) ** 2 <= 1.0
    right = disc & (xx > 0)
    left = disc & (xx < 0)
    mid = disc & (xx == 0)
    kr = right.astype(float) + 0.5 * mid
    kl = left.astype(float) + 0.5 * mid
    s = kr.sum() + kl.sum()
    kr /= s
    kl /= s
    # c>0: Y の像は右へ、C の像は左へ
    return (kr, kl) if c >= 0 else (kl, kr)


def conv(mask, k):
    if k.size == 1:
        return mask * k[0, 0]
    return fftconvolve(mask, k, mode="same")


def box_down(img, ss):
    if ss == 1:
        return img
    h, w = img.shape
    h2, w2 = h // ss, w // ss
    return img[: h2 * ss, : w2 * ss].reshape(h2, ss, w2, ss).mean(axis=(1, 3))


def enc_gamma(x):
    x = np.clip(x, 0, 1)
    return 255.0 * np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 1e-9), 1 / 2.4) - 0.055)


def to_u8(lin_rgb, noise=0.0, rng=None):
    """線形 RGB（dict R,G,B）→ ガンマ後 8bit の BGR。noise はガンマ後の階調単位"""
    bgr = np.stack([enc_gamma(lin_rgb["B"]), enc_gamma(lin_rgb["G"]), enc_gamma(lin_rgb["R"])], axis=-1)
    if noise > 0:
        bgr = bgr + rng.normal(0, noise, bgr.shape)
    return np.clip(np.round(bgr), 0, 255).astype(np.uint8)


def chroma420(bgr):
    """4:2:0 の模擬: クロマを 2×2 平均して 1/2 にし、線形補間で戻す（BT.601 の cv2 変換）"""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    h, w = ycc.shape[:2]
    h2, w2 = h // 2, w // 2
    y = ycc[:, :, 0]
    cr = cv2.resize(ycc[:, :, 1], (w2, h2), interpolation=cv2.INTER_AREA)
    cb = cv2.resize(ycc[:, :, 2], (w2, h2), interpolation=cv2.INTER_AREA)
    cr = cv2.resize(cr, (w, h), interpolation=cv2.INTER_LINEAR)
    cb = cv2.resize(cb, (w, h), interpolation=cv2.INTER_LINEAR)
    out = np.stack([y, cr, cb], axis=-1)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


def measured_like_filter():
    """実データに合わせた [仮定] のフィルタとセンサ: 白色光で R≈Y 半開口のみ、B≈C 半開口のみ、G は Y:C≈0.7:0.3。

    分光の形そのものは未測定。白色光の重み（実測の d_RG/d_RB≈0.3）に合わせただけで、灯火の波長での値は仮定。
    """
    lam = sp.LAM
    t_y = sp.expit((lam - 505.0) / 6.0)
    # シアン側の緑の透過を下げる
    t_c = sp.expit((565.0 - lam) / 6.0) * (0.36 + 0.64 * sp.expit((495.0 - lam) / 10.0))
    return t_y, t_c


def narrow_sensors():
    sens = dict(R=sp.gauss(612.0, 55.0) * sp.IRCUT, G=sp.gauss(535.0, 70.0) * sp.IRCUT, B=sp.gauss(458.0, 50.0) * sp.IRCUT)
    return sens


def weights_for(E, t_y, t_c, sens):
    gains = {k: 1.0 / (np.sum(sens[k] * (t_y + t_c) * 0.5) * sp.DL) for k in sp.CH}
    out = {}
    for k in sp.CH:
        wy = float(np.sum(E * sens[k] * t_y * 0.5) * sp.DL * gains[k])
        wc = float(np.sum(E * sens[k] * t_c * 0.5) * sp.DL * gains[k])
        out[k] = (wy, wc)
    return out


class Spectra:
    """白色光での重みと灯火の重みをまとめて持つ"""

    def __init__(self, t_y, t_c, sens, name=""):
        self.name = name
        self.t_y, self.t_c, self.sens = t_y, t_c, sens
        flat = np.ones_like(sp.LAM)
        self.white = weights_for(flat, t_y, t_c, sens)
        src = sp.sources()
        self.led = {}
        for nm in ["LED赤630", "LED黄590", "LED青505"]:
            W = weights_for(src[nm], t_y, t_c, sens)
            m = max(W[k][0] + W[k][1] for k in sp.CH)
            self.led[nm] = {k: (W[k][0] / m, W[k][1] / m) for k in sp.CH}
        self.src = {k: weights_for(v, t_y, t_c, sens) for k, v in src.items()}

    def describe(self):
        print(f"[{self.name}] 白色光: " + ", ".join(f"{k}(Y,C)=({self.white[k][0]:.3f},{self.white[k][1]:.3f}) e={sp.centroid_e(*self.white[k]):+.2f}" for k in sp.CH))
        for nm, W in self.led.items():
            print(f"  {nm}: " + ", ".join(f"{k}=({W[k][0]:.3f},{W[k][1]:.3f})" for k in sp.CH))


def ideal_spectra():
    """理想: R は Y のみ、B は C のみ、G は Y・C 半々。灯火は単色で、狭帯域の 1 チャネル（黄は R,G）"""
    return None


def signal_masks(w_px, h_px, lamp_px, pitch_px, ss, margin, vertical=False):
    W = int(np.ceil((w_px + 2 * margin))) * ss
    H = int(np.ceil((h_px + 2 * margin))) * ss
    if vertical:
        W = int(np.ceil((h_px + 2 * margin))) * ss
        H = int(np.ceil((w_px + 2 * margin))) * ss
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W / 2.0, H / 2.0
    hw = (w_px if not vertical else h_px) * ss / 2.0
    hh = (h_px if not vertical else w_px) * ss / 2.0
    head = (np.abs(xx - cx) <= hw) & (np.abs(yy - cy) <= hh)
    lamps = []
    centers = []
    for i in (-1, 0, 1):
        lx = cx + (i * pitch_px * ss if not vertical else 0.0)
        ly = cy + (i * pitch_px * ss if vertical else 0.0)
        lamps.append((xx - lx) ** 2 + (yy - ly) ** 2 <= (lamp_px * ss / 2.0) ** 2)
        centers.append((lx / ss, ly / ss))
    return dict(head=head, lamps=lamps, centers=centers, cx=cx / ss, cy=cy / ss, hw=hw / ss, hh=hh / ss, shape=(H, W))


def render(components, c, cy, spec, ss=4):
    """components: list of (mask, W, L)。W は {ch:(wy,wc)}、L は明るさ係数。線形 RGB の dict を返す。"""
    kY, kC = half_kernels(c, cy, ss)
    out = {k: 0.0 for k in sp.CH}
    for mask, W, L in components:
        m = mask.astype(np.float32)
        a = conv(m, kY)
        b = conv(m, kC)
        for k in sp.CH:
            # 一様な面では a=b=0.5。W は wy+wc=1（白）なので 2 倍して I=L にそろえる
            out[k] = out[k] + L * (W[k][0] * a + W[k][1] * b) * 2.0
    return {k: box_down(np.asarray(out[k]), ss) for k in sp.CH}
