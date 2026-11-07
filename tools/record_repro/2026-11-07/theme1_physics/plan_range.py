"""傾きで広がる 1/z の幅、同じ量のぼけ、要求距離と視差の関係（528×352・薄レンズの計算。実測なし）。"""
import math

BETA = 4.0 / (3.0 * math.pi)
A = 90.0 / 5.6
ROWS = 352  # architecture.md の IMAGE_SIZE (528, 352)
PX_PER_MM = 2178.0 / 90.2  # 既知の組（v=90.2 で 2178, v=109.3 で 2639）

print("検算: v=109.3 の F =", round(PX_PER_MM * 109.3, 1), "（既知 2639）")
print()
print("配置ごとの幅（F5.6）。帯の仮定: ぼけ量のみ 片側 c 1〜10 px、色視差 |d|≤5 px")
print("配置 | J[mm] | F | 行で広がる幅[/m] | 画面内ぼけ差[px] | ぼけ差/幅[px・m] | ぼけのみ計[/m] | 色視差計[/m] | 色視差(0.75≤|d|≤5)計[/m]")
for name, th, v in [
    ("傾けない", 0.0, 90.0),
    ("路面合焦", 3.97, 90.2),
    ("1.0 m 面", 17.46, 94.35),
    ("θ=30°", 30.0, 109.3),
]:
    F = PX_PER_MM * v
    J = 90.0 / math.sin(math.radians(th)) if th > 0 else math.inf
    span = ROWS / (J * F) * 1000.0
    blur = A / J * ROWS
    ratio = blur / span if span > 0 else float("nan")
    AF = A * F / 1000.0
    K = BETA * AF
    b = (10 - 1) / AF
    d = 10 / K
    d2 = (10 - 1.5) / K
    print(f"{name} | {J:.0f} | {F:.0f} | {span:.3f} | {blur:.1f} | {ratio:.1f} | {b + span:.3f} | {d + span:.3f} | {d2 + span:.3f}")

print()
print("必要な 1/z の幅")
for near, far in [(1, 4), (1, 40), (3, 40), (5, 30), (10, 40), (5, 100)]:
    print(f"{near}〜{far} m: {1 / near - 1 / far:.3f} /m")

K = BETA * A * 2178 / 1000
print()
print(f"校正のばらつき相当の δd: 4000 mm で ±80 mm → {80 * BETA * A * 2639 / 4000 ** 2:.3f} px")
print()
print("停止距離（仮定: 反応 0.75 s, 減速 0.7 g）と距離の信号（K=14.86 px・m, 528 幅, 路面合焦）")
print("速度 | 停止距離 | K/z [px] | δd=0.15 | δd=0.09 | δd=0.05 | 人の高さ [px]")
for kmh in [30, 40, 60, 80, 100]:
    vms = kmh / 3.6
    stop = vms * 0.75 + vms * vms / (2 * 0.7 * 9.80665)
    dz = K / stop
    res = []
    for dd in [0.15, 0.09, 0.05]:
        lo = K / (dz + dd)
        hi = K / (dz - dd) if dz > dd else math.inf
        res.append(f"{(lo / stop - 1) * 100:+.0f}/{(hi / stop - 1) * 100:+.0f} %")
    print(f"{kmh} km/h | {stop:.1f} m | {dz:.2f} | {res[0]} | {res[1]} | {res[2]} | {2178 * 1.7 / stop:.0f}")

print()
print("固定距離で同じ表")
for z in [5, 10, 15, 20, 30, 40, 50]:
    dz = K / z
    res = []
    for dd in [0.15, 0.09, 0.05]:
        lo = K / (dz + dd)
        hi = K / (dz - dd) if dz > dd else math.inf
        res.append(f"{(lo / z - 1) * 100:+.0f}/{(hi / z - 1) * 100:+.0f} %")
    print(f"{z} m | {dz:.2f} px | {res[0]} | {res[1]} | {res[2]} | 人 {2178 * 1.7 / z:.0f} px | 灯 0.3 m {2178 * 0.3 / z:.0f} px")

print()
print("人（1.7 m）の全身が 352 行に収まる最短距離と視野")
for name, F in [("路面合焦 F=2178", 2178.0), ("θ=30° F=2639", 2639.0)]:
    print(
        f"{name}: 全身 {F * 1.7 / ROWS:.1f} m 以遠 / 水平画角 {2 * math.degrees(math.atan(264 / F)):.1f}° / "
        f"4 m での視野 幅 {528 / F * 4:.2f} m × 高さ {ROWS / F * 4:.2f} m / 20 m での視野 幅 {528 / F * 20:.1f} m"
    )
    print(f"  4 m の視野に収まる物体の高さの上限 {ROWS / F * 4:.2f} m")

print()
print("θ=30° の合焦行が画面内にある距離（B5 の当てはめ r_c = 428477/z − 45.39）")
for r in [0, 351]:
    print(f"行 {r}: z = {428477 / (r + 45.39) / 1000:.2f} m")
print(f"2.8 m の合焦行: {428477 / 2800 - 45.39:.1f}")
r_c = 428477 / 2800 - 45.39
print(f"2.8 m 上端 c（0.0893/行）: {r_c * 0.0893:.1f} px, 下端 c（0.0893/行）: {(351 - r_c) * 0.0893:.1f} px, 下端 c（0.1055/行＝neg 側）: {(351 - r_c) * 0.1055:.1f} px")
