"""合焦付近の「不感域」は左右分割開口でも理論上の限界か: 1 次元の Fisher 情報で確かめる。
模様は未知のガウス過程（パワー 1/(f²+f0²)）、観測は R,B（または対称開口 1 チャネル）に白色雑音 σ。
Trouvé 流の共分散の微分から I(c) = Σ_f tr(Q⁻¹ Q' Q⁻¹ Q') を出し、σ_c=1/√I、σ_d=βσ_c を比べる。
理想の半円 PSF（1 次元射影）。収差・色の混ざりは別に入れて比べる。
"""
import sys

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

BETA = 4 / (3 * np.pi)
N = 25  # 窓の画素数（1 次元で窓の幅に当たる周波数格子）
NH = 25  # 縦方向の平均の本数（ZNCC 窓の縦 25 行を独立と数える粗い近似）
f = np.arange(1, N // 2 + 1) / N  # cycles/px
PIX = np.sinc(f)  # 画素の開口


def lsf_half(c, side, ss=4001):
    """幅 |c| の円の左右半分の 1 次元射影。side=+1 で x>0 側。c<0 なら左右が入れ替わる。合計 1。"""
    r = abs(c) / 2
    if r < 1e-9:
        return np.array([0.0]), np.array([1.0])
    x = np.linspace(0, r, ss)
    w = np.sqrt(np.maximum(r * r - x * x, 0))
    w /= w.sum()
    s = side * (1 if c >= 0 else -1)
    return s * x, w


def H_half(c, side, sigma_base=0.0):
    x, w = lsf_half(c, side)
    h = (w[None, :] * np.exp(-2j * np.pi * f[:, None] * x[None, :])).sum(axis=1)
    return h * PIX * np.exp(-2 * (np.pi * f * sigma_base) ** 2)


def H_disc(c, sigma_base=0.0):
    x, w = lsf_half(c, 1)
    # 対称開口: 両側の平均
    h = (w[None, :] * np.cos(2 * np.pi * f[:, None] * x[None, :])).sum(axis=1)
    return h * PIX * np.exp(-2 * (np.pi * f * sigma_base) ** 2)


def fisher_split(c, S, sigma, leak=0.0, base=0.0, dc=1e-3):
    def Q(cc):
        Hy = H_half(cc, +1, base)
        Hc = H_half(cc, -1, base)
        Hr = (1 - leak) * Hy + leak * Hc
        Hb = (1 - leak) * Hc + leak * Hy
        out = np.zeros((f.size, 2, 2), dtype=complex)
        out[:, 0, 0] = S * Hr * np.conj(Hr)
        out[:, 0, 1] = S * Hr * np.conj(Hb)
        out[:, 1, 0] = S * Hb * np.conj(Hr)
        out[:, 1, 1] = S * Hb * np.conj(Hb)
        out[:, 0, 0] += sigma**2
        out[:, 1, 1] += sigma**2
        return out

    Qp, Qm = Q(c + dc), Q(c - dc)
    Q0 = Q(c)
    dQ = (Qp - Qm) / (2 * dc)
    I = 0.0
    for k in range(f.size):
        Qi = np.linalg.inv(Q0[k])
        M = Qi @ dQ[k]
        I += np.real(np.trace(M @ M))
    return I * NH


def fisher_disc(c, S, sigma, base=0.0, dc=1e-3):
    def Q(cc):
        return S * np.abs(H_disc(cc, base)) ** 2 + sigma**2

    c0 = max(abs(c), 1e-6)
    Qp, Qm = Q(c0 + dc), Q(max(c0 - dc, 0))
    Q0 = Q(c0)
    dQ = (Qp - Qm) / (dc + min(dc, c0))
    return np.sum((dQ / Q0) ** 2) * NH * 2  # 2 チャネル分（同じ開口を R,B で観測）


# パワースペクトル: 校正板に近い 1/f² 型（粒の幅に対応する f0）。コントラストを 1 に規格化
f0 = 1.0 / 15
S = 1.0 / (f**2 + f0**2)
S = S / S.sum() * (0.3**2) * N  # 分散 0.3²（8bit の 76 階調相当）
for snr_name, sigma in [("σ=1.5/255", 1.5 / 255), ("σ=5/255", 5 / 255)]:
    print(f"=== {snr_name}（画素の雑音） ===")
    print("c[px] | 対称開口 σ_c[px] | 左右分割 σ_c | 左右分割 σ_d=βσ_c | 分割(色の混ざり 15 %) σ_d | 分割(基底ぼけ σ=2 px) σ_d | 分割(混ざり 15 %＋基底 σ=2) σ_d")
    for c in [0.05, 0.1, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
        Id = fisher_disc(c, S, sigma)
        Is = fisher_split(c, S, sigma)
        Il = fisher_split(c, S, sigma, leak=0.15)
        Ib = fisher_split(c, S, sigma, base=2.0)
        Ilb = fisher_split(c, S, sigma, leak=0.15, base=2.0)
        print(f"{c} | {1/np.sqrt(Id):.3f} | {1/np.sqrt(Is):.3f} | {BETA/np.sqrt(Is):.4f} | {BETA/np.sqrt(Il):.4f} | {BETA/np.sqrt(Ib):.4f} | {BETA/np.sqrt(Ilb):.4f}")
    print()
print("注: 窓 25×25 を独立なスペクトル標本とみなす粗い近似で、絶対値は目安。比較に使うのは c に対する形（c→0 で発散するか）。")
