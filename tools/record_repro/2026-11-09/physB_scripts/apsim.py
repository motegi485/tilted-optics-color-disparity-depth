"""開口形状ごとの色視差の合成と推定の共通ライブラリ（physB・2026-11-09）

目的:
  任意のカラーフィルタ開口（チャネルごとの透過マスクを返す関数）でぼけた RGB 画像を合成し、
  視差（＝ぼけ量 c）の推定誤差を、エッジの向き・ぼけ量・雑音ごとに比べるための道具。
  開発リポ tools/record_repro/2026-11-07/theme1_physics/synth.py（左右半円 PSF）を任意の開口へ一般化したもの
  （synth.py は読んだだけで import はしない。符号の約束「c>0 で Y 側＝R の像が右」は synth.py に合わせた）。
入力: なし（ライブラリ）。run_*.py から import する。
出力: なし。
開口形状の差し替え:
  形状は shape(U, V) -> {"R": T_R, "G": T_G, "B": T_B} の関数。U は右・V は上の開口座標（単位円の内側だけが使われる）、
  T は 0〜1 の透過率。SHAPES に (関数, 表示名) を登録すれば、どの run_*.py にもそのまま流し込める。
  開口座標の「左右」は 研究方針.md §7 の表記（現行は R=左半分）に合わせた。像面では c>0 のとき点対称に反転して写る。
仮定（すべて [仮定]。実機では未確認）:
  - 幾何光学のみ。回折・収差・ケラレ・色収差・センサのクロストークは入れない。
  - チャネル k の PSF は、直径 |c| px のデフォーカス円（アオリの縦の縮みは sy=cosθ 倍の楕円）をチャネル k の
    透過領域で切り取ったもの。c>0（合焦面より手前）は開口マスクを点対称に反転、c<0 はそのまま。
  - 理想フィルタ（Y=R+G, C=G+B, M=R+B, W=素通し を 0/1 で透過）。被写体は無彩色（3 チャネルで同じ反射率）。
  - ホワイトバランス後（各チャネルの PSF の総和を 1）。sRGB ガンマ→ガウス雑音（8bit 階調・既定 σ=1.5）→丸め→0..255。
  - 画素の積分は ss 倍（既定 4）の格子で描いて箱型平均で縮小。
"""
import sys

sys.dont_write_bytecode = True
import cv2
import numpy as np
import scipy.fft as sfft

ROOT = r"C:\Users\t5424001\dev\tilted_optics_color_disparity_depth"
CH = ("R", "G", "B")
LO, HI = 0.05, 0.70  # 被写体の線形の明るさの範囲 [仮定]（q3_beta_zncc.py と同じ）
BETA = 4.0 / (3.0 * np.pi)


# ---------------------------------------------------------------------------
# 開口形状（U 右・V 上・単位円）
# ---------------------------------------------------------------------------
def _ang(U, V):
    return np.degrees(np.arctan2(V, U)) % 360.0


def _sector(a, center, width=120.0):
    d = (a - center + 180.0) % 360.0 - 180.0
    return np.abs(d) <= width / 2.0


def _f(x):
    return x.astype(np.float64)


def shape_full(U, V):
    o = np.ones_like(U)
    return dict(R=o, G=o, B=o)


def shape_cy_lr(U, V):
    """現行: 左 Y（R+G）・右 C（G+B）。R=左半分・G=全開口・B=右半分"""
    left = U < 0
    return dict(R=_f(left), G=np.ones_like(U), B=_f(~left))


def rotated(fn, deg):
    """形状 fn を開口面内で deg 度（反時計回り）回した形状を返す"""
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)

    def g(U, V):
        # 回した後の点 (U,V) は、回す前の点 (U',V') = R(-t)(U,V)
        return fn(c * U + s * V, -s * U + c * V)

    return g


def shape_rgb3(U, V):
    a = _ang(U, V)
    return dict(R=_f(_sector(a, 210)), G=_f(_sector(a, 90)), B=_f(_sector(a, 330)))


def shape_cmy3(U, V):
    a = _ang(U, V)
    m, c, y = _sector(a, 90), _sector(a, 210), _sector(a, 330)
    return dict(R=_f(m | y), G=_f(c | y), B=_f(m | c))


def shape_ycg3(U, V):
    """Y／C／G 三扇形: R=Y の扇形（150°）・B=C の扇形（30°）・G=全開口（270° は G フィルタ）"""
    a = _ang(U, V)
    return dict(R=_f(_sector(a, 150)), G=np.ones_like(U), B=_f(_sector(a, 30)))


def make_kaneko(rho0=5.6 / 22.0, center_rb=True):
    """Kaneko2025 型: 左右の R/B 半円（外側の輪）＋中央の小開口（G 用・半径比 rho0）。

    中央は穴（素通し）で R・G・B とも通すと仮定（center_rb=True）[仮定: 論文に中央の R/B 透過の記載なし]。
    """

    def g(U, V):
        rho = np.hypot(U, V)
        cen = rho <= rho0
        left = U < 0
        if center_rb:
            R = (left & ~cen) | cen
            B = (~left & ~cen) | cen
        else:
            R = left & ~cen
            B = ~left & ~cen
        return dict(R=_f(R), G=_f(cen), B=_f(B))

    return g


def make_cy_msector(width=120.0):
    """改変例 1: 現行の C/Y 左右分割の下側の扇形（中心 270°・幅 width）を M（R+B・G を遮る）に置き換える"""

    def g(U, V):
        a = _ang(U, V)
        m = _sector(a, 270.0, width)
        left = U < 0
        return dict(R=_f((left & ~m) | m), G=_f(~m), B=_f((~left & ~m) | m))

    return g


def shape_cy_topclear(U, V):
    """改変例 2: 上半分を素通し（W）、下半分を左 Y・右 C"""
    top = V > 0
    left = U < 0
    return dict(R=_f(top | left), G=np.ones_like(U), B=_f(top | ~left))


SHAPES = {
    "cy_lr": (shape_cy_lr, "現行 C/Y 左右"),
    "cy_rot45": (rotated(shape_cy_lr, 45.0), "C/Y 45°回転"),
    "rgb3": (shape_rgb3, "RGB 三扇形"),
    "cmy3": (shape_cmy3, "CMY 三扇形"),
    "ycg3": (shape_ycg3, "Y/C/G 三扇形"),
    "kaneko": (make_kaneko(), "Kaneko2025 型"),
    "cy_msec": (make_cy_msector(120.0), "C/Y＋下 M 扇形"),
    "cy_topW": (shape_cy_topclear, "C/Y＋上半分素通し"),
    "cy_tb": (rotated(shape_cy_lr, 90.0), "C/Y 上下（90°回転）"),
    "full": (shape_full, "全開口（参照）"),
}


# ---------------------------------------------------------------------------
# 形状の統計（重心・光量）
# ---------------------------------------------------------------------------
def aperture_grid(n=801):
    g = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    U, V = np.meshgrid(g, -g)  # 行 0 が上（V=+1）
    inside = U ** 2 + V ** 2 <= 1.0
    return U, V, inside


def shape_stats(fn, n=801):
    """各チャネルの平均透過（開口全体比）と開口座標の重心 (cU, cV)（半径 1 単位）"""
    U, V, inside = aperture_grid(n)
    T = fn(U, V)
    out = {}
    for k in CH:
        t = np.where(inside, T[k], 0.0)
        s = t.sum()
        out[k] = dict(area=s / inside.sum(), cU=(t * U).sum() / s, cV=(t * V).sum() / s)
    return out


def centroid_img(stats, sy=1.0):
    """像面座標（x 右・y 下）での重心。c>0 で点対称に反転するので p=(-cU, sy*cV)。
    チャネル k の像のずれ（オフセット）は o_k = (c/2)·p_k（c は符号付き）。"""
    return {k: np.array([-stats[k]["cU"], sy * stats[k]["cV"]]) for k in CH}


def baseline_table(p, angles_deg=np.arange(0, 180, 1.0)):
    """エッジの法線 n=(cosφ, sinφ)（像面・y 下）ごとの、(p_i−p_j)·n の最大（ぼけ c=2 あたり）。
    研究方針.md §7 の「法線 n のエッジの基線 = max|(p_i−p_j)·n|」と同じ量。あわせて全ペアの二乗和の平方根も返す。"""
    pairs = [("R", "G"), ("G", "B"), ("R", "B")]
    mx, rs = [], []
    for a in angles_deg:
        n = np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        v = [abs((p[i] - p[j]) @ n) for i, j in pairs]
        mx.append(max(v))
        rs.append(np.sqrt(np.sum(np.square(v))))
    return np.array(mx), np.array(rs)


# ---------------------------------------------------------------------------
# PSF と描画
# ---------------------------------------------------------------------------
def psf_kernels(fn, c, ss=4, sy=1.0, sub=4):
    """チャネルごとの PSF（ss 倍の格子・総和 1）。c は符号付きぼけ直径 [1× の px]。sy は縦の倍率（アオリで cosθ）"""
    r = abs(c) / 2.0 * ss
    ry = r * sy
    if r < 0.5 or ry < 0.5:
        d = np.ones((1, 1))
        return {k: d.copy() for k in CH}
    nx, ny = int(np.ceil(r)) + 1, int(np.ceil(ry)) + 1
    yy, xx = np.mgrid[-ny : ny + 1, -nx : nx + 1].astype(np.float64)
    off = (np.arange(sub) + 0.5) / sub - 0.5
    acc = {k: np.zeros_like(xx) for k in CH}
    for oy in off:
        for ox in off:
            x = xx + ox
            y = yy + oy
            if c > 0:
                U, V = -x / r, y / ry
            else:
                U, V = x / r, -y / ry
            inside = U ** 2 + V ** 2 <= 1.0
            T = fn(U, V)
            for k in CH:
                acc[k] += np.where(inside, T[k], 0.0)
    out = {}
    for k in CH:
        s = acc[k].sum()
        if s <= 0:
            raise ValueError(f"チャネル {k} の透過がゼロ")
        out[k] = acc[k] / s
    return out


def box_down(img, ss):
    if ss == 1:
        return img
    h, w = img.shape
    return img[: h // ss * ss, : w // ss * ss].reshape(h // ss, ss, w // ss, ss).mean(axis=(1, 3))


class Scene:
    """ss 倍の格子の線形の被写体。FFT を一度だけ計算して使い回す（周期境界。評価は中央の ROI だけ）"""

    def __init__(self, img_ss, ss, name=""):
        self.img = img_ss.astype(np.float64)
        self.ss = ss
        self.name = name
        self.shape = img_ss.shape
        self.F = sfft.rfft2(self.img, workers=-1)


def kernel_fft(k, shape):
    pad = np.zeros(shape)
    kh, kw = k.shape
    pad[:kh, :kw] = k
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return sfft.rfft2(pad, workers=-1)


def render(scene, fn, c, sy=1.0, kfft=None):
    """一定のぼけ c で描く。線形 RGB の dict（1× の解像度）を返す"""
    if kfft is None:
        ks = psf_kernels(fn, c, scene.ss, sy)
        kfft = {k: kernel_fft(ks[k], scene.shape) for k in CH}
    out = {}
    for k in CH:
        img = sfft.irfft2(scene.F * kfft[k], s=scene.shape, workers=-1)
        out[k] = box_down(img, scene.ss)
    return out


def kernel_ffts(fn, c, shape, ss, sy=1.0):
    ks = psf_kernels(fn, c, ss, sy)
    return {k: kernel_fft(ks[k], shape) for k in CH}


def render_rowvarying(scene, fn, c_rows, sy=1.0, step=0.25):
    """行ごとにぼけ c が変わる画像（アオリ）。c_rows は 1× の各行の符号付き c。
    c を step 刻みで描いた画像を行ごとに線形補間する [仮定: 行内一定のぼけで近似。PSF の広がりの中での c の変化は無視]"""
    lo = np.floor(c_rows.min() / step) * step
    hi = np.ceil(c_rows.max() / step) * step
    levels = np.arange(lo, hi + step * 0.5, step)
    imgs = []
    for cl in levels:
        imgs.append(render(scene, fn, float(cl), sy))
    H = len(c_rows)
    out = {k: np.zeros((H, imgs[0]["R"].shape[1])) for k in CH}
    for y in range(H):
        t = (c_rows[y] - lo) / step
        i0 = int(np.clip(np.floor(t), 0, len(levels) - 1))
        i1 = min(i0 + 1, len(levels) - 1)
        w = t - i0
        for k in CH:
            out[k][y] = (1 - w) * imgs[i0][k][y] + w * imgs[i1][k][y]
    return out


def enc_gamma(x):
    x = np.clip(x, 0, 1)
    return 255.0 * np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 1e-9), 1 / 2.4) - 0.055)


def dec_gamma(v8):
    x = np.asarray(v8, dtype=np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def to_u8(lin, sigma=1.5, rng=None):
    """線形 RGB → ガンマ → 雑音（ガンマ後の階調・チャネルごとに σ を変えられる）→ uint8 の dict"""
    out = {}
    for k in CH:
        v = enc_gamma(lin[k])
        s = sigma[k] if isinstance(sigma, dict) else sigma
        if s > 0:
            v = v + rng.normal(0, s, v.shape)
        out[k] = np.clip(np.round(v), 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# 被写体（ss 倍の格子・線形）
# ---------------------------------------------------------------------------
def _coords(H, W, ss):
    """ss 格子の画素中心の 1× 座標"""
    y = (np.arange(H * ss) + 0.5) / ss
    x = (np.arange(W * ss) + 0.5) / ss
    return np.meshgrid(x, y)


def scene_barcode(H, W, ss, phi_deg, rng, wmin=2.0, wmax=10.0):
    """1 次元のランダムなバーコード（幅 wmin〜wmax px・明るさ一様乱数）。
    φ はエッジの法線の角度（像面・x 右・y 下）: φ=0 で縦エッジ（縦縞）、φ=90 で横エッジ（横縞）。"""
    X, Y = _coords(H, W, ss)
    n = np.array([np.cos(np.deg2rad(phi_deg)), np.sin(np.deg2rad(phi_deg))])
    t = (X - W / 2) * n[0] + (Y - H / 2) * n[1]
    tmin, tmax = t.min() - 1, t.max() + 1
    edges = [tmin]
    while edges[-1] < tmax:
        edges.append(edges[-1] + rng.uniform(wmin, wmax))
    edges = np.array(edges)
    vals = rng.uniform(LO, HI, len(edges))
    idx = np.searchsorted(edges, t) - 1
    return vals[np.clip(idx, 0, len(vals) - 1)]


def scene_step(H, W, ss, phi_deg, lo=0.12, hi=0.55):
    """画像中央を通る 1 本の段差エッジ（法線の角度 φ）"""
    X, Y = _coords(H, W, ss)
    n = np.array([np.cos(np.deg2rad(phi_deg)), np.sin(np.deg2rad(phi_deg))])
    t = (X - W / 2) * n[0] + (Y - H / 2) * n[1]
    return np.where(t > 0, hi, lo).astype(np.float64)


def scene_1overf(H, W, ss, rng, aniso_sigma_deg=None, floor=0.15, mean=0.30, std=0.12, clip=True):
    """1/f 雑音（振幅 ∝ 1/f・1× のナイキストで打ち切り）。
    aniso_sigma_deg を与えると、周波数ベクトルが縦軸（fy）に近い成分＝横向きの構造を強める:
    振幅の重み = floor + exp(-ψ²/(2σ²))（ψ: 周波数ベクトルと fy 軸のなす角）。"""
    h, w = H * ss, W * ss
    fy = np.fft.fftfreq(h)[:, None] * ss  # 1× の cycles/px
    fx = np.fft.fftfreq(w)[None, :] * ss
    f = np.hypot(fx, fy)
    amp = np.where((f > 0) & (f <= 0.5), 1.0 / np.maximum(f, 1e-9), 0.0)
    if aniso_sigma_deg is not None:
        psi = np.degrees(np.arctan2(np.abs(fx), np.abs(fy)))  # 0 が fy 軸
        amp = amp * (floor + np.exp(-(psi ** 2) / (2 * aniso_sigma_deg ** 2)))
    ph = rng.uniform(0, 2 * np.pi, (h, w))
    img = np.real(np.fft.ifft2(amp * np.exp(1j * ph)))
    img = (img - img.mean()) / img.std() * std + mean
    return np.clip(img, 0.02, 0.85) if clip else img


# ---------------------------------------------------------------------------
# 推定法
# ---------------------------------------------------------------------------
def box(a, K):
    return cv2.boxFilter(a, cv2.CV_64F, (K, K), normalize=True, borderType=cv2.BORDER_REFLECT)


def est_i_rust(img8, a="R", b="B", K=25, S=41, T=20):
    """(i) 現行の実装そのもの（core.disparity.calcDisparityByZNCC・Rust）。戻り値 d = x_a − x_b [px]"""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from core.disparity import calcDisparityByZNCC

    return calcDisparityByZNCC(np.ascontiguousarray(img8[a]), np.ascontiguousarray(img8[b]), K, S, T)


def _shift_x_int(a, o):
    """out(x) = a(x+o)（端は複製）"""
    if o == 0:
        return a
    out = np.empty_like(a)
    if o > 0:
        out[:, :-o] = a[:, o:]
        out[:, -o:] = a[:, -1:]
    else:
        out[:, -o:] = a[:, :o]
        out[:, :-o] = a[:, :1]
    return out


def est_i_py(img8, a="R", b="B", K=25, S=41, T=20):
    """(i) の Python 再実装（ベクトル化）。_disparity_reference.py と同じ規則:
    |Sobel x|（uint8 に飽和）、a 側で T 超の画素を選ぶ、整数シフトの ZNCC、端を除く最大、厳密な局所最大、
    放物線のサブピクセル。ZNCC の分母が 0 のときは 0（OpenCV の TM_CCOEFF_NORMED と同じ扱い）。
    ピークの値と曲率も返す（ぼけ形状の不一致の解析用）。"""
    ga = cv2.convertScaleAbs(cv2.Sobel(img8[a], cv2.CV_32F, 1, 0)).astype(np.float64)
    gb = cv2.convertScaleAbs(cv2.Sobel(img8[b], cv2.CV_32F, 1, 0)).astype(np.float64)
    H, W = ga.shape
    sr = (S - 1) // 2
    k = (K - 1) // 2
    mA = box(ga, K)
    vA = np.maximum(box(ga * ga, K) - mA * mA, 0)
    sims = np.empty((S, H, W))
    for j in range(S):
        o = sr - j
        gbs = _shift_x_int(gb, o)
        mB = box(gbs, K)
        vB = np.maximum(box(gbs * gbs, K) - mB * mB, 0)
        cab = box(ga * gbs, K) - mA * mB
        den = np.sqrt(vA * vB)
        sims[j] = np.where(den > 1e-9, cab / np.maximum(den, 1e-12), 0.0)
    d, peak, curv = _peak_subpix(sims, np.arange(S) - sr, 1.0)
    sel = ga > T
    valid_xy = np.zeros_like(sel)
    valid_xy[k : H - k, sr + k + 1 : W - sr - k - 1] = True
    m = sel & valid_xy
    d[~m] = np.nan
    return d, peak, curv


def _peak_subpix(sims, cands, step):
    """候補軸 0 の最大（端を除く）→ 厳密な局所最大 → 放物線。cands は等間隔。値・ピーク値・曲率を返す"""
    N = sims.shape[0]
    inner = sims[1 : N - 1]
    j = np.argmax(inner, axis=0) + 1
    H, W = j.shape
    yy, xx = np.mgrid[0:H, 0:W]
    p0 = sims[j, yy, xx]
    pn = sims[j - 1, yy, xx]
    pp = sims[j + 1, yy, xx]
    ok = (p0 > pn) & (p0 > pp)
    den = 4 * p0 - 2 * pn - 2 * pp
    use = (den > 0) & (np.abs(pp - pn) <= den)
    sub = np.where(use, (pp - pn) / np.where(use, den, 1.0), 0.0)
    val = cands[0] + (j + sub) * step
    val = np.where(ok, val, np.nan)
    curv = np.where(ok, den / 2.0, np.nan)  # 2p0 − pn − pp（候補 1 刻みあたり）
    return val, np.where(ok, p0, np.nan), curv


def grad_mag(img8):
    out, gxs, gys = {}, {}, {}
    for k in CH:
        f = img8[k].astype(np.float32)
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1)
        out[k] = np.sqrt(gx * gx + gy * gy).astype(np.float64)
        gxs[k], gys[k] = gx.astype(np.float64), gy.astype(np.float64)
    return out, gxs, gys


PAIRS = (("R", "G"), ("G", "B"), ("R", "B"))


def est_scalar2d(img8, p, cands, K=25, T=20, use_bando=True, pairs=PAIRS, metrics=("zsum", "zw", "bando")):
    """(ii)/(iii) スカラーのぼけ c を仮定し、各チャネルを o_k=(c/2)·p_k だけ 2 次元に戻して（三次補間）照合する。

    zsum : |∇I|（Sobel の勾配の大きさ）のチャネル対 ZNCC の単純平均（(ii)）
    bando: 線形の明るさの 3×3 共分散から 1 − det(相関行列)＝1 − det(Σ)/(σr²σg²σb²)（Bando 2008 の color-line 指標・(ii)）
    zw   : チャネル対の ZNCC を、窓内の構造テンソル T で w_ij = b_ijᵀ T b_ij / tr T（b_ij=p_i−p_j）重み付け（(iii)）
    エッジの選択: 3 チャネルの |∇I| の最大 > T。戻り値: {metric: (ĉ, peak, curv)}・選択マスク
    """
    mag, gx, gy = grad_mag(img8)
    H, W = img8["R"].shape
    sel = np.maximum.reduce([mag[k] for k in CH]) > T
    lin = {k: dec_gamma(img8[k]).astype(np.float64) for k in CH}
    Gx = sum(gx[k] for k in CH)
    Gy = sum(gy[k] for k in CH)
    Jxx, Jxy, Jyy = box(Gx * Gx, K), box(Gx * Gy, K), box(Gy * Gy, K)
    tr = Jxx + Jyy + 1e-9
    wts = {}
    for i, j in pairs:
        b = p[i] - p[j]
        wts[(i, j)] = (b[0] ** 2 * Jxx + 2 * b[0] * b[1] * Jxy + b[1] ** 2 * Jyy) / tr
    wsum = sum(wts.values()) + 1e-12
    Yg, Xg = np.mgrid[0:H, 0:W].astype(np.float32)
    N = len(cands)
    S = {m: np.empty((N, H, W)) for m in metrics}
    chans_needed = sorted(set([c for pr in pairs for c in pr]))
    for n, c in enumerate(cands):
        sm, sl = {}, {}
        for k in chans_needed:
            mx = (Xg + np.float32(c / 2.0 * p[k][0])).astype(np.float32)
            my = (Yg + np.float32(c / 2.0 * p[k][1])).astype(np.float32)
            sm[k] = cv2.remap(mag[k].astype(np.float32), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT).astype(np.float64)
            if use_bando and "bando" in metrics:
                sl[k] = cv2.remap(lin[k].astype(np.float32), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT).astype(np.float64)
        mu = {k: box(sm[k], K) for k in chans_needed}
        var = {k: np.maximum(box(sm[k] * sm[k], K) - mu[k] ** 2, 0) for k in chans_needed}
        z = {}
        for i, j in pairs:
            cov = box(sm[i] * sm[j], K) - mu[i] * mu[j]
            den = np.sqrt(var[i] * var[j])
            z[(i, j)] = np.where(den > 1e-9, cov / np.maximum(den, 1e-12), 0.0)
        if "zsum" in metrics:
            S["zsum"][n] = sum(z.values()) / len(z)
        if "zw" in metrics:
            S["zw"][n] = sum(wts[pr] * z[pr] for pr in pairs) / wsum
        if "bando" in metrics:
            ml = {k: box(sl[k], K) for k in CH}
            cv = {}
            for a_ in range(3):
                for b_ in range(a_, 3):
                    ka, kb = CH[a_], CH[b_]
                    cv[(a_, b_)] = box(sl[ka] * sl[kb], K) - ml[ka] * ml[kb]
            vr, vg, vb = [np.maximum(cv[(i, i)], 1e-12) for i in range(3)]
            rrg = cv[(0, 1)] / np.sqrt(vr * vg)
            rgb = cv[(1, 2)] / np.sqrt(vg * vb)
            rrb = cv[(0, 2)] / np.sqrt(vr * vb)
            det = 1 - rrg ** 2 - rgb ** 2 - rrb ** 2 + 2 * rrg * rgb * rrb
            S["bando"][n] = 1.0 - det
    step = float(cands[1] - cands[0])
    res = {}
    for m in metrics:
        v, pk, cu = _peak_subpix(S[m], np.asarray(cands, dtype=np.float64), step)
        v = np.where(sel, v, np.nan)
        res[m] = (v, pk, cu)
    return res, sel


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------
def eval_mask_from_full(scene, c, sy=1.0, T=20, roi=None, rng_seed=0):
    """形状によらない評価画素: 全開口（通常のレンズ）で描いた画像の |∇I| が T を超える画素（ROI 内）"""
    lin = render(scene, shape_full, c, sy)
    u8 = to_u8(lin, sigma=0.0)
    mag, gx, gy = grad_mag(u8)
    m = mag["G"] > T
    if roi is not None:
        r = np.zeros_like(m)
        r[roi] = True
        m &= r
    return m, gx["G"], gy["G"]


def local_orientation(gx, gy, K=25):
    """窓内の構造テンソルから、支配的なエッジ法線の角度（0〜180°・0=縦エッジ）とコヒーレンスを返す"""
    Jxx, Jxy, Jyy = box(gx * gx, K), box(gx * gy, K), box(gy * gy, K)
    ang = 0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy)) % 180.0
    coh = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-9)
    return ang, coh


def est_crossblur(img8, fn, cands, K=25, T=20, sy=1.0, pairs=PAIRS):
    """(iv) PSF 合わせ（[Mishima2018] の「チャネル別 PSF を畳み込んでぼけを揃える」考え方を、ここでは 2 次元・任意形状で試す）。
    候補 c ごとに A = I_i ⊛ K_j(c), B = I_j ⊛ K_i(c)（線形の明るさ）を作り、|∇A| と |∇B| の ZNCC の対平均を最大化する。
    c が真値なら A=B（畳み込みの可換性）なので、ぼけの形の不一致による偏りが原理的に出ない。
    注意: Mishima2018 の実装（水平方向・凸最適化）そのものではない。"""
    lin = {k: dec_gamma(img8[k]).astype(np.float64) for k in CH}
    mag0, _, _ = grad_mag(img8)
    sel = np.maximum.reduce([mag0[k] for k in CH]) > T
    H, W = img8["R"].shape
    N = len(cands)
    S = np.empty((N, H, W))
    for n, c in enumerate(cands):
        ks = psf_kernels(fn, float(c), ss=1, sy=sy, sub=8)
        zs = []
        for i, j in pairs:
            # filter2D は相関なので核を反転して畳み込みにする
            A = cv2.filter2D(lin[i], cv2.CV_64F, ks[j][::-1, ::-1].copy(), borderType=cv2.BORDER_REFLECT)
            B = cv2.filter2D(lin[j], cv2.CV_64F, ks[i][::-1, ::-1].copy(), borderType=cv2.BORDER_REFLECT)
            ga = np.hypot(cv2.Sobel(A, cv2.CV_64F, 1, 0), cv2.Sobel(A, cv2.CV_64F, 0, 1))
            gb = np.hypot(cv2.Sobel(B, cv2.CV_64F, 1, 0), cv2.Sobel(B, cv2.CV_64F, 0, 1))
            ma, mb = box(ga, K), box(gb, K)
            va = np.maximum(box(ga * ga, K) - ma * ma, 0)
            vb = np.maximum(box(gb * gb, K) - mb * mb, 0)
            cab = box(ga * gb, K) - ma * mb
            den = np.sqrt(va * vb)
            zs.append(np.where(den > 1e-18, cab / np.maximum(den, 1e-30), 0.0))
        S[n] = sum(zs) / len(zs)
    v, pk, cu = _peak_subpix(S, np.asarray(cands, dtype=np.float64), float(cands[1] - cands[0]))
    return np.where(sel, v, np.nan), pk, cu


LR_SHAPES = {"cy_lr", "kaneko", "cy_msec", "cy_topW"}


def run_methods(u8, p, cands, use_i):
    """(i)（use_i のとき）と (ii)/(iii) をまとめて当て、{method: ĉ} を返す"""
    out = {}
    if use_i:
        d = est_i_rust(u8)
        out["i_hzncc"] = 2.0 * d / (p["R"][0] - p["B"][0])
    res, _ = est_scalar2d(u8, p, cands)
    out["ii_zsum"], out["ii_bando"], out["iii_zw"] = res["zsum"][0], res["bando"][0], res["zw"][0]
    return out


def iso_gain(fn, p, c, cands, use_i, seeds=(9001, 9002), H=192, W=192, ss=4, sigma=1.5,
             roi=(slice(40, 152), slice(40, 152))):
    """校正の模擬: 等方 1/f 雑音（実機の校正板＝ランダム模様の代わり）で、推定法ごとの比 k = median(ĉ/c) を求める。
    実機の校正は距離ごとに d と z の関係を経験的に合わせるので、向きによらない一定の比は吸収される、とみなす [仮定]。"""
    ratios = {}
    for sd in seeds:
        rng = np.random.default_rng(sd)
        sc = Scene(scene_1overf(H, W, ss, rng), ss)
        lin = render(sc, fn, c)
        u8 = to_u8(lin, sigma, np.random.default_rng(sd + 1))
        m, _, _ = eval_mask_from_full(sc, c, roi=roi)
        for mt, e in run_methods(u8, p, cands, use_i).items():
            v = e[m]
            v = v[np.isfinite(v)]
            ratios.setdefault(mt, []).append(v / c)
    return {mt: float(np.median(np.concatenate(v))) if sum(len(x) for x in v) else np.nan for mt, v in ratios.items()}


def stats(est, truth, mask, tol=1.0):
    """est, truth: ぼけ c の推定と真値 [px]（truth はスカラーか配列）。mask: 評価画素"""
    n = int(mask.sum())
    if n == 0:
        return dict(n=0, valid=np.nan, good=np.nan, bias=np.nan, sd=np.nan, mae=np.nan)
    t = truth if np.isscalar(truth) else truth[mask]
    e = est[mask] - t
    ok = np.isfinite(e)
    ev = e[ok]
    if ev.size == 0:
        return dict(n=n, valid=0.0, good=0.0, bias=np.nan, sd=np.nan, mae=np.nan)
    med = float(np.median(ev))
    mad = float(1.4826 * np.median(np.abs(ev - med)))
    return dict(n=n, valid=float(ok.mean()), good=float((np.abs(ev) <= tol).sum() / n), bias=med, sd=mad,
                mae=float(np.mean(np.abs(ev))))
