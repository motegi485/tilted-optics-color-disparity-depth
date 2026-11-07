"""テーマ1の物理検算: 薄レンズのアオリ光学系モデルと、式に依らない光線追跡。

座標系（単位 mm）: 原点はレンズ中心。Z は前方（センサ面の法線）、Y は上、X は横。
センサ面は Z = -v。レンズ面は X 軸まわりに θ 回転（θ>0 はレンズ上端が被写体側へ倒れる）。
  レンズ面の接線 t = (Y,Z) = (cosθ, sinθ)、法線 n = (Y,Z) = (-sinθ, cosθ)
行 r は下向きに増える。r = c0 - F*Y/Z、F = v/p。

ヒンジ則・式 c=(A/J)(u-r)+AF/z は使わずに raytrace() が結果を出す。式との照合に使う。
"""
import sys

sys.dont_write_bytecode = True
import numpy as np

BETA = 4.0 / (3.0 * np.pi)

# F の扱い（研究方針_検出と測距.md §4.2 の 2026-10-30 注記）:
# 1080 幅・v=109.3 mm で F=5398 px → F は v に比例する（画素ピッチ・crop は同じ）
F_REF_1080 = 5398.0
V_REF = 109.3
P_1080 = V_REF / F_REF_1080  # mm/px（校正からの逆算）
P_1080_SENSOR = 22.3 / 1080.0  # mm/px（センサ幅 22.3 mm を 1080 px に割った値・照合用）
C0 = 360.0  # 720 行の中央
R_1080_TO_528 = 1080.0 / 528.0


class Rig:
    def __init__(self, name, theta_deg, v, Hc=1300.0, f=90.0, N=5.6, p=P_1080, phi_deg=0.0, note=""):
        self.name = name
        self.theta = np.deg2rad(theta_deg)
        self.theta_deg = theta_deg
        self.v = float(v)
        self.Hc = float(Hc)
        self.f = float(f)
        self.N = float(N)
        self.p = float(p)
        self.phi = np.deg2rad(phi_deg)
        self.note = note

    @property
    def A(self):
        return self.f / self.N

    @property
    def F(self):
        return self.v / self.p

    @property
    def J(self):
        """符号付きヒンジ距離（レンズ中心の下が正）。θ→0 では inf。"""
        s = np.sin(self.theta)
        return np.inf if abs(s) < 1e-12 else self.f / s

    @property
    def row_coef(self):
        """A/J = A sinθ / f（θ→0 で 0 に連続）"""
        return self.A * np.sin(self.theta) / self.f

    @property
    def const_term(self):
        """A (f - v cosθ)/(f p)。θ→0 でも特異にならない形で持つ"""
        return self.A * (self.f - self.v * np.cos(self.theta)) / (self.f * self.p)

    @property
    def tan_psi(self):
        s = np.sin(self.theta)
        return np.nan if abs(s) < 1e-12 else (self.v * np.cos(self.theta) - self.f) / (self.v * s)

    @property
    def u(self):
        """合焦面の消失行。θ→0 では発散するので nan を返す"""
        t = self.tan_psi
        return np.nan if not np.isfinite(t) else C0 - self.F * t

    def to_camera(self, H, z):
        """路面からの高さ H・レンズからの水平距離 z を、カメラ座標 (Y_c, Z_c) へ"""
        Yw = np.asarray(H, dtype=float) - self.Hc
        Zw = np.asarray(z, dtype=float)
        c, s = np.cos(self.phi), np.sin(self.phi)
        return Yw * c - Zw * s, Yw * s + Zw * c

    def row(self, H, z):
        Yc, Zc = self.to_camera(H, z)
        return C0 - self.F * Yc / Zc

    def blur_signed_rowz(self, r, Zc):
        """式: c = (A/J)(c0 - r) + A(f - v cosθ)/(f p) + A F / Z（= (A/J)(u-r) + AF/Z）

        θ→0 でも使える非特異な形。Z→∞ では AF/Z → 0。c>0 は合焦面より手前。
        """
        return self.row_coef * (C0 - np.asarray(r, dtype=float)) + self.const_term + self.A * self.F / np.asarray(Zc, dtype=float)

    def blur_signed(self, H, z):
        Yc, Zc = self.to_camera(H, z)
        return self.blur_signed_rowz(C0 - self.F * Yc / Zc, Zc)

    def pof_height(self, z):
        """カメラ水平（φ=0）の場合の、距離 z で合焦面が通る路面からの高さ"""
        if abs(np.sin(self.theta)) < 1e-12:
            return np.nan
        return self.Hc - self.J + np.asarray(z, dtype=float) * self.tan_psi


def rig_parallel(name, p_mm, Hc=1300.0, N=5.6, f=90.0, p=P_1080, note=""):
    """合焦面を路面と平行に高さ p_mm へ置く（カメラ水平）。"""
    J = Hc - p_mm
    if abs(J) <= f:
        raise ValueError(f"|J|={abs(J)} <= f: 合焦面の高さがレンズ高に近すぎて実現できない")
    th = np.arcsin(f / J)
    v = f / np.cos(th)
    return Rig(name, np.rad2deg(th), v, Hc=Hc, f=f, N=N, p=p, note=note)


def rig_hinge_slope(name, J, tan_psi, Hc=1300.0, N=5.6, f=90.0, p=P_1080, note=""):
    th = np.arcsin(f / J)
    v = f / (np.cos(th) - np.sin(th) * tan_psi)
    return Rig(name, np.rad2deg(th), v, Hc=Hc, f=f, N=N, p=p, note=note)


def rig_road_pitched(name, phi_deg, Hc=1300.0, N=5.6, f=90.0, p=P_1080, note=""):
    """カメラを上へ φ 向けたまま合焦面を路面に置く。J=Hc/cosφ、カメラ座標の tanψ=-tanφ"""
    phi = np.deg2rad(phi_deg)
    J = Hc / np.cos(phi)
    th = np.arcsin(f / J)
    v = f / (np.cos(th) + np.sin(th) * np.tan(phi))
    return Rig(name, np.rad2deg(th), v, Hc=Hc, f=f, N=N, p=p, phi_deg=phi_deg, note=note)


def rig_untilted(name, z_focus=np.inf, Hc=1300.0, N=5.6, f=90.0, p=P_1080, note=""):
    v = f if not np.isfinite(z_focus) else f * z_focus / (z_focus - f)
    return Rig(name, 0.0, v, Hc=Hc, f=f, N=N, p=p, note=note)


# ---------------------------------------------------------------------------
# 光線追跡（ヒンジ則を使わない）
# ---------------------------------------------------------------------------

def _aperture_grid(A, n=161):
    g = np.linspace(-A / 2, A / 2, n)
    ax, at = np.meshgrid(g, g)
    m = ax**2 + at**2 <= (A / 2) ** 2
    return ax[m], at[m]


def _aperture_edge(A, n=1440):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return A / 2 * np.cos(t), A / 2 * np.sin(t)


def _trace(rig, X, Y, Z, ax, at):
    """物点 (X,Y,Z)[カメラ座標] から開口点 (ax, at) を通る光線がセンサ面に着く位置を返す。

    薄レンズの結像（1/s + 1/s' = 1/f、s は法線方向の物体距離）だけを使う。
    物点が無限遠に近いときも共役点は有限に収束するので、そのまま計算する。
    """
    th = rig.theta
    f = rig.f
    # レンズ法線方向の物体距離 s = P・n
    s = -Y * np.sin(th) + Z * np.cos(th)
    if not s > f:
        raise ValueError("物点がレンズの前側焦点より手前")
    m = f / (s - f)  # P' = -m P
    Xp, Yp, Zp = -m * X, -m * Y, -m * Z
    # 開口上の点（レンズ面内）
    a_x = ax
    a_y = at * np.cos(th)
    a_z = at * np.sin(th)
    lam = (-rig.v - a_z) / (Zp - a_z)
    Sx = a_x + lam * (Xp - a_x)
    Sy = a_y + lam * (Yp - a_y)
    col = -Sx / rig.p
    row = C0 + Sy / rig.p
    return col, row


def raytrace(rig, X, Y, Z, n_grid=161):
    """物点の像の足跡（footprint）を返す。

    c_x: 横幅 [px]、c_y: 縦幅 [px]、d: 左右の半開口の像の重心の差 [px]（col_右半開口 - col_左半開口）
    row0: 主光線の行、sign: 符号付きの c（左右の順序から判定。d と同符号）
    """
    ex, et = _aperture_edge(rig.A)
    col_e, row_e = _trace(rig, X, Y, Z, ex, et)
    c_x = col_e.max() - col_e.min()
    c_y = row_e.max() - row_e.min()
    ax, at = _aperture_grid(rig.A, n_grid)
    col, row = _trace(rig, X, Y, Z, ax, at)
    left = ax < 0
    right = ax > 0
    d = col[right].mean() - col[left].mean()
    col0, row0 = _trace(rig, X, Y, Z, np.array([0.0]), np.array([0.0]))
    # 物理的に、開口の +X 側を通る光線が像の col で -側へ来るのが「手前」でない場合など、向きは d の符号で表す
    return dict(c_x=c_x, c_y=c_y, d=d, row0=float(row0[0]), col0=float(col0[0]),
                col_full=col.mean(), col_right=col[right].mean(), col_left=col[left].mean())


def raytrace_world(rig, H, z, x_lat=0.0):
    Yc, Zc = rig.to_camera(H, z)
    return raytrace(rig, float(x_lat), float(Yc), float(Zc))


# ---------------------------------------------------------------------------
# 対象（寸法は [仮定]）
# ---------------------------------------------------------------------------
OBJECTS = {
    "signal": dict(label="信号灯器", H_c=5000.0, h=450.0, w=1200.0, lamp=300.0),
    "sign60": dict(label="路側標識0.6m", H_c=2100.0, h=600.0, w=600.0, lamp=np.nan),
    "sign90": dict(label="片持標識0.9m", H_c=5450.0, h=900.0, w=900.0, lamp=np.nan),
    "head": dict(label="人の頭(1.7m)", H_c=1700.0, h=1700.0, w=500.0, lamp=np.nan, top=1700.0, bot=0.0),
    "car_top": dict(label="車の上端(1.5m)", H_c=1500.0, h=1500.0, w=1800.0, lamp=np.nan, top=1500.0, bot=0.0),
    "car_tail": dict(label="車の尾灯(0.9m)", H_c=900.0, h=1500.0, w=1800.0, lamp=150.0, top=1500.0, bot=0.0),
}


def obj_top_bot(o):
    top = o.get("top", o["H_c"] + o["h"] / 2)
    bot = o.get("bot", o["H_c"] - o["h"] / 2)
    return top, bot


def default_rigs():
    a = rig_parallel("(a) 路面合焦", 0.0, note="Hc=1.3m,F5.6")
    b05 = rig_parallel("(b) 面を0.5m", 500.0, note="路面と平行")
    b10 = rig_parallel("(b) 面を1.0m", 1000.0, note="路面と平行")
    c = Rig("(c) 現行θ=30°", 30.0, 109.3, Hc=1330.0, note="v=109.3,Hc=1.33m")
    d1 = rig_parallel("(d1) 面を5.0m", 5000.0, note="路面と平行・θ<0")
    d2 = rig_hinge_slope("(d2) 斜面50mで5m", 1300.0, 0.1, note="路面ヒンジ・勾配0.1")
    e = rig_parallel("(e) (a)をF2.8", 0.0, N=2.8)
    e2 = rig_parallel("(e') 0.5m面をF2.8", 500.0, N=2.8)
    ctrl = rig_untilted("対照 θ=0・無限遠")
    return [a, b05, b10, c, d1, d2, e, e2, ctrl]
