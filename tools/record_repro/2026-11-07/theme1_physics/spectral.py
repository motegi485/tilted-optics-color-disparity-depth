"""分光モデル: 光源・センサ・開口フィルタを合わせて、各センサチャネルがどちらの半開口の光を受けるかを出す。

開発リポにフィルタの分光資料は無かった（Grep で確認）ので、すべて [仮定]。
ただし「単色光ならどのチャネルも同じ開口形状を見る」は仮定によらない（T(x,y;λ) は 1 つの関数）。
"""
import numpy as np

LAM = np.arange(380.0, 731.0, 1.0)
DL = LAM[1] - LAM[0]


def gauss(mu, fwhm):
    s = fwhm / 2.3548
    return np.exp(-0.5 * ((LAM - mu) / s) ** 2)


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


# センサ感度 [仮定]（IR カットで 680 nm 以降を落とす）
IRCUT = expit((680.0 - LAM) / 8.0)
SENS = dict(
    R=gauss(600.0, 75.0) * IRCUT,
    G=gauss(535.0, 85.0) * IRCUT,
    B=gauss(462.0, 65.0) * IRCUT,
)
CH = ["R", "G", "B"]


def filter_pair(y_edge=500.0, c_edge=580.0, width=7.0, leak=0.0):
    """Y（黄: 長波長を通す）、C（シアン: 短波長を通す）の透過率 [仮定]"""
    t_y = leak + (1 - leak) * expit((LAM - y_edge) / width)
    t_c = leak + (1 - leak) * expit((c_edge - LAM) / width)
    return t_y, t_c


FILTER_CASES = {
    "基準(Y500/C580)": dict(y_edge=500.0, c_edge=580.0, leak=0.0),
    "重なり狭(Y520/C560)": dict(y_edge=520.0, c_edge=560.0, leak=0.0),
    "重なり広(Y480/C610)": dict(y_edge=480.0, c_edge=610.0, leak=0.0),
    "基準+漏れ3%": dict(y_edge=500.0, c_edge=580.0, leak=0.03),
}

# 光源・物体 [仮定]。LED のピーク・半値幅は一般知識の範囲の仮定
def sources():
    flat = np.ones_like(LAM)
    def hp(edge, floor=0.05, w=12.0):
        return floor + (0.85 - floor) * expit((LAM - edge) / w)
    def bp(mu, fwhm, floor=0.05, peak=0.7):
        return floor + (peak - floor) * gauss(mu, fwhm)
    return dict(
        LED赤630=gauss(630.0, 20.0),
        LED黄590=gauss(590.0, 15.0),
        LED青505=gauss(505.0, 30.0),
        白=0.85 * flat,
        黒=0.04 * flat,
        赤塗料=hp(600.0),
        黄塗料=hp(510.0),
        青塗料=bp(465.0, 70.0),
        緑塗料=bp(525.0, 70.0),
        印刷赤=0.06 + 0.72 * expit((LAM - 595.0) / 12.0),
        印刷黄=0.06 + 0.78 * expit((LAM - 505.0) / 12.0),
        印刷青緑=0.06 + 0.55 * gauss(500.0, 80.0),
    )


def wb_gains(t_y, t_c):
    """白色の平坦な光で R=G=B=1（両半開口の合計）になるゲイン"""
    g = {}
    for k in CH:
        g[k] = 1.0 / np.sum(SENS[k] * (t_y + t_c) * 0.5) / DL
    return g


def weights(E, t_y, t_c, gains=None):
    """W[k] = (Y半開口の寄与, C半開口の寄与)。ホワイトバランス後の値（平坦な白 1.0 で合計 1）。"""
    gains = gains or wb_gains(t_y, t_c)
    out = {}
    for k in CH:
        wy = np.sum(E * SENS[k] * t_y * 0.5) * DL * gains[k]
        wc = np.sum(E * SENS[k] * t_c * 0.5) * DL * gains[k]
        out[k] = (float(wy), float(wc))
    return out


def centroid_e(wy, wc):
    """チャネルの実効重心。+1: すべて Y 半開口、-1: すべて C 半開口、0: 全開口。

    合計が 0 のときは未定義（nan）。
    """
    s = wy + wc
    return (wy - wc) / s if s > 1e-12 else np.nan
