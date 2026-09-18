"""calibration/capture_calib.py — FR-8 webcam 校正画像の取得（バースト平均 → preprocess → PNG。--save-raw で raw も保存）。

校正パイプライン「取得 → calibrate → fit_params」の最初の工程。`data/calib_imgs/{距離}.png`
を出力し、後段の `calibration.calibrate` → `calibration.fit_params` が live 推論 params
（root `params/fit_result.csv`）を生成する。本スクリプトは `params/` には一切触れない。

最重要要件＝校正と推論の取得条件一致（docs/module-calibration.md）。そのため取得経路は推論と完全同一にする:
`realtime.camera.WebcamSource`（取得）と共有 `preprocess`（3:2 中央クロップ 1280→1080 →
INTER_AREA → 528×352）を **再利用** し、取得・前処理を独自に再実装しない。視差チェック（任意）
も `core.disparity.calcDisparityByZNCC` を **読むだけ** で純関数呼び出しする。

書き込み先は `data/calib_imgs/` のみ（`params/`・`outputs/` には触れない・docs/architecture.md）。プロンプト/
メッセージは日本語、技術用語・パス・関数名・定数は英語のまま。入力検証は raise（assert は -O で
無効化されるため使わない・リスク#10）。
"""
from __future__ import annotations

# --- サブパッケージから root の config / core / realtime を import するためのブートストラップ ---
# 推奨実行は `uv run python -m calibration.capture_calib`（root を cwd に）。
# 直接実行 `python calibration/capture_calib.py` でも動くよう root を sys.path に通す
# （calibration/calibrate.py と同方式）。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time

import argparse

import cv2
import numpy as np

import config
from realtime.camera import WebcamSource, preprocess
from core.disparity import calcDisparityByZNCC


# ===== 本スクリプト固有の定数（config に対応値が無いもの） =====
DEFAULT_BURST_N = 32          # バースト枚数の既定値（CLI --n で可変）
# バースト1回あたりの経過時間上限（秒）。フィードが固着（同一フレーム連発）した場合に
# 無限ループせず「異常」として打ち切るための安全弁。source.read() のブロッキング挙動や
# np.array_equal の比較コスト・実 fps に依存しない time.monotonic() ベースの時間予算とする。
# 目安: ~4fps 相当の控えめな前提でも DEFAULT_BURST_N(32) 枚は 8s 以内に揃う
# （実 30fps なら約1.1s）。--n を増やしても破綻しない十分な安全弁として 8.0 を採用。
# config.LIVE_WAIT_TIMEOUT_S（open 時のライブ待ち）とは別概念・独立。
BURST_TIMEOUT_S = 8.0


def scan_calib_dir() -> tuple[list[Path], list[Path]]:
    """起動時の散布ガード用。calib_imgs/ の中身を「距離名 PNG」と「想定外」に分類して返す。

    距離名 PNG = 拡張子 .png かつ stem が正の整数（本キャンペーンが書く形式 {dist}.png）。
    それ以外（非 png・非整数 stem・サブディレクトリ等）は想定外として別枠で返す。
    ディレクトリ未作成なら両方空リスト。
    """
    d = config.CALIB_IMGS_DIR
    if not d.is_dir():
        return [], []
    dist_pngs: list[Path] = []
    others: list[Path] = []
    for p in sorted(d.iterdir()):
        stem_is_int = p.is_file() and p.suffix.lower() == ".png" and _is_positive_int(p.stem)
        (dist_pngs if stem_is_int else others).append(p)
    return dist_pngs, others


def _is_positive_int(s: str) -> bool:
    """文字列が正の整数 mm を表すか（距離ファイル名の判定）。"""
    try:
        return int(s) > 0
    except ValueError:
        return False


def prompt_distance() -> int | None:
    """距離[mm]（正の整数）を対話入力する。q/quit で終了センチネル None を返す。

    空入力・非正整数はクラッシュさせず再入力させる（docs/requirements-and-status.md の制約条件）。
    """
    while True:
        raw = input("距離[mm] を入力（正の整数）。終了は q: ").strip()
        if raw.lower() in {"q", "quit"}:
            return None
        if _is_positive_int(raw):
            return int(raw)
        print("  → 正の整数で入力してください（例: 1400）。終了は q。")


def _ask_yes_no(message: str) -> bool:
    """y/n を受け取る。y/yes→True、n/no→False。それ以外は再入力。"""
    while True:
        ans = input(f"{message} (y/n): ").strip().lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("  → y か n で答えてください。")


def capture_burst(source: WebcamSource, n: int) -> np.ndarray | None:
    """N 枚の **異なる** raw フレーム（1280×720 BGR）を取得し、raw 領域で平均した uint8 raw を返す。

    平均は raw float64 累積 → /N → uint8（preprocess は呼び出し側で平均後に1回だけ適用する）。
    crop と INTER_AREA は線形なので「各フレーム preprocess 後に平均」と概ね等価だが、丸めを1回に
    抑え計算も軽い raw 平均 → 単一 preprocess を採る（docs/module-calibration.md）。

    フレーム同一性: WebcamSource は frame_id を持たないため、直前に採用したフレームとの完全一致
    （np.array_equal）で重複を検出し skip する。実フィードはセンサノイズで毎フレーム必ず変動する
    （_wait_until_live が時間 std>0 で確認済み）ため、固着していない限り N 枚は速やかに揃う。
    バースト開始からの経過時間が BURST_TIMEOUT_S（time.monotonic() 基準）を超えたらフィード固着
    とみなし None を返す（呼び出し側で撮り直し可能に）。
    read 失敗時も None を返し、半端・破損した平均は書き込ませない（docs/module-calibration.md）。
    """
    if n < 1:
        raise ValueError(f"capture_burst: n は 1 以上が必要です（渡された値: {n}）")

    acc = np.zeros((*config.WEBCAM_NATIVE_SIZE[::-1], 3), dtype=np.float64)  # (H,W,3)
    last: np.ndarray | None = None
    accepted = 0
    deadline = time.monotonic() + BURST_TIMEOUT_S  # 壁時計 time.time() はシステム時刻補正の影響を受けるため不使用

    while accepted < n:
        if time.monotonic() >= deadline:
            print(f"  → 異なるフレームが {n} 枚揃いません（{accepted}/{n}）。"
                  "フィードが固着している可能性があります。撮り直してください。")
            return None
        ok, frame = source.read()
        if not ok or frame is None:
            print("  → フレーム read に失敗しました。撮り直してください。")
            return None
        if last is not None and np.array_equal(frame, last):
            time.sleep(0.003) # 直前と同一の重複フレーム → 一瞬待つ
            continue
        acc += frame.astype(np.float64)
        last = frame
        accepted += 1

    avg = np.rint(acc / n).astype(np.uint8)  # 四捨五入で uint8 raw に戻す
    return avg


def save_calib_png(proc_bgr: np.ndarray, dist: int, out_dir: Path,
                   allow_overwrite: bool) -> bool:
    """preprocess 済み (352,528,3) BGR を PNG（可逆・lossless）で out_dir/{dist}.png に保存する。

    webcam は既に 4:2:0 で R/B 横解像が落ちているため、JPEG 保存による二重クロマ間引きを避ける
    （PNG 必須・docs/module-calibration.md）。保存前に shape=(352,528,3) を検証（IMAGE_SIZE 整合）。
    既存ファイルがあり allow_overwrite=False のときは上書き確認を行い、否なら保存せず False を返す。
    """
    expected = (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0], 3)  # (H,W,3)=(352,528,3)
    if proc_bgr.shape != expected:
        raise ValueError(
            f"save_calib_png: 保存対象は {expected} を想定。実際は {proc_bgr.shape}（IMAGE_SIZE 不整合）")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dist}.png"
    if path.exists() and not allow_overwrite:
        if not _ask_yes_no(f"  {path.name} は既に存在します。上書きしますか"):
            print("  → 保存を中止しました。")
            return False

    if not cv2.imwrite(str(path), proc_bgr):
        raise RuntimeError(f"save_calib_png: PNG 書き込みに失敗しました: {path}")
    print(f"  → 保存しました: {path}")
    return True


def save_calib_raw(raw_bgr: np.ndarray, dist: int, out_dir: Path, allow_overwrite: bool) -> bool:
    """バースト平均の raw (720,1280,3) BGR を PNG（可逆）で out_dir/{dist}.png に保存する（--save-raw 用・2026-09-16）。

    処理解像度（config.IMAGE_SIZE）を変える実験では、校正画像を別解像度で作り直す必要がある。
    preprocess 前の raw を残しておけば、再撮影せずに `preprocess_to(raw, size)`（tools/analysis_common.py）で
    任意の解像度の校正画像を再生成できる。保存前に shape=(720,1280,3)（WEBCAM_NATIVE_SIZE 整合）を検証する。
    上書きの扱いは save_calib_png と同じ。
    """
    w, h = config.WEBCAM_NATIVE_SIZE
    expected = (h, w, 3)
    if raw_bgr.shape != expected or raw_bgr.dtype != np.uint8:
        raise ValueError(
            f"save_calib_raw: 保存対象は {expected} uint8 を想定。実際は {raw_bgr.shape} {raw_bgr.dtype}")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dist}.png"
    if path.exists() and not allow_overwrite:
        if not _ask_yes_no(f"  raw {path.name} は既に存在します。上書きしますか"):
            print("  → raw の保存を中止しました。")
            return False

    if not cv2.imwrite(str(path), raw_bgr):
        raise RuntimeError(f"save_calib_raw: PNG 書き込みに失敗しました: {path}")
    print(f"  → raw を保存しました: {path}")
    return True


def run_subfunc_check(proc_bgr: np.ndarray) -> None:
    """保存した (528,352) で視差を試算し、撮影品質を即時に診断する（docs/module-calibration.md・任意）。

    pipeline/evaluate と同一のチャンネル分割（cv2.split の BGR 順で G を破棄し blue/red を割当）と
    同一 gate1（|d|>DISP_MAX を NaN）を当て、有効点数・pos/neg 点数・|d| 最大・clip 警告・
    calibrate 採用帯（DISP_PLOT_LOW<|d|<DISP_PLOT_HEIGHT）の予測点数を表示する。
    推論帯（config.INFER_BAND_LOW/HIGH・2026-09-18）は掛けない: これは校正撮影の診断で、校正は校正帯を使うため。
    core は読むだけ・純関数呼び出し・無改変。
    """
    # pipeline.process_frame / evaluate.evaluate と一致させる: blue, _, red = cv2.split(...)
    blue, _, red = cv2.split(proc_bgr)
    disp = calcDisparityByZNCC(
        red, blue, config.KERNEL_SIZE, config.SEARCH_RANGE_X, config.EDGE_TH)

    # gate1: 探索半幅超の視差を無効化（pipeline/evaluate と同一）
    disp[np.abs(disp) > config.DISP_MAX] = np.nan

    valid = ~np.isnan(disp)
    n_valid = int(valid.sum())
    n_pos = int(np.count_nonzero(disp > 0))   # NaN は比較で False になり自然に除外
    n_neg = int(np.count_nonzero(disp < 0))

    print(f"  [サブフィット] 有効視差画素数 = {n_valid}（pos={n_pos} / neg={n_neg}）")
    if n_valid == 0:
        print("  → 有効視差が 0 です。グレイン不足・パターン不良・露出を確認してください。")
        return

    abs_d = np.abs(disp)
    d_max = float(np.nanmax(abs_d))
    print(f"  [サブフィット] |d| 最大 = {d_max:.2f}px（DISP_MAX={config.DISP_MAX}）")
    # 探索端の張り付き検知【2026-09 レビューで閾値修正】。
    # 旧条件 `d_max >= DISP_MAX - 0.5`（=14.5）は到達不能な死にゲートだった: 視差は
    # d = max_x - s_range + δ、max_x は端2点を除いた [1, S-2]、かつ厳密局所最大の3点放物線
    # 補間では |δ| < 0.5 が代数的に保証される（A=p-n>0, B=p-q>0 として δ=(A-B)/(2(A+B))）。
    # よって |d| < DISP_MAX - 0.5 が常に成り立ち、警告は原理的に発火しない。
    # 実際に意味があるのは「ZNCC ピークが探索できる最外シフト（max_x = S-2、すなわち
    # |d| > DISP_MAX - 1.5）に載った」ことなので、そこを閾値にし、端ビンの画素率も併記する。
    edge_bin_th = config.DISP_MAX - 1.5
    n_edge_bin = int(np.count_nonzero(abs_d > edge_bin_th))
    if n_edge_bin:
        print(f"  → 警告: |d| が探索最外ビン（>{edge_bin_th:.1f}px）に載った画素が "
              f"{n_edge_bin} 個（有効の {n_edge_bin / n_valid * 100:.4f}%、最大 {d_max:.4f}px）。"
              f"真の視差が探索範囲 ±{config.DISP_MAX}px を超えている可能性があります。")

    low, height = config.DISP_PLOT_LOW, config.DISP_PLOT_HEIGHT
    band_pos = int(np.count_nonzero((disp > low) & (disp < height)))
    band_neg = int(np.count_nonzero((disp < -low) & (disp > -height)))
    print(f"  [サブフィット] calibrate 採用帯 "
          f"({low}<|d|<{height}) 予測点数: pos={band_pos} / neg={band_neg}")
    if n_pos == 0 or n_neg == 0:
        print("  → 警告: pos/neg のどちらかが 0 です。校正には両符号の点が必要です。")


def _startup_guard() -> bool:
    """起動時に calib_imgs/ の現状を表示し、想定外/既存 PNG があれば続行可否を確認する（docs/module-calibration.md）。

    続行してよければ True、中止なら False を返す。空ディレクトリ（または未作成）はそのまま True。
    """
    dist_pngs, others = scan_calib_dir()
    print(f"=== 起動時チェック: {config.CALIB_IMGS_DIR} ===")
    if not dist_pngs and not others:
        print("  （空です。新規撮影フローに進みます）")
        return True

    if dist_pngs:
        print(f"  既存の距離 PNG（{len(dist_pngs)} 件）:")
        for p in dist_pngs:
            print(f"    - {p.name}")
    if others:
        print(f"  ★想定外のファイル/ディレクトリ（{len(others)} 件）:")
        for p in others:
            print(f"    - {p.name}")
        print("  （非 png・非距離名・サブディレクトリ等。混入や誤上書きを避けるため確認します）")

    return _ask_yes_no("このまま続行しますか")


def _capture_one_distance(source: WebcamSource, dist: int, n: int, save_raw: bool = False) -> None:
    """1 距離分の撮影フロー: バースト → preprocess → 保存 → サブフィット(任意) → 撮り直し(任意)。

    撮り直し=n になるまでループする。撮り直し時は明示的上書きを許可する（docs/module-calibration.md）。
    """
    is_reshoot = False
    while True:
        print(f"--- {dist}mm: バースト撮影（{n} 枚平均）。ターゲット/カメラを静止させてください ---")
        print("    （WB に加え露出も手動固定であること。露出変動は平均を狂わせます）")
        raw_avg = capture_burst(source, n)
        if raw_avg is None:
            # read 失敗・固着。半端な平均は書かない。撮り直し可否のみ尋ねる
            if _ask_yes_no(f"{dist}mm を撮り直しますか"):
                is_reshoot = True
                continue
            return

        proc = preprocess(raw_avg)  # (1280,720) raw → (352,528,3)。平均後に1回だけ適用
        saved = save_calib_png(proc, dist, config.CALIB_IMGS_DIR, allow_overwrite=is_reshoot)
        if saved and save_raw:      # 処理解像度の実験用に preprocess 前の raw も残す（--save-raw）
            save_calib_raw(raw_avg, dist, config.CALIB_RAW_DIR, allow_overwrite=is_reshoot)

        if saved and _ask_yes_no(f"{dist}mm の撮影フレームをチェックしますか"):
            run_subfunc_check(proc)

        if _ask_yes_no(f"{dist}mm を撮り直しますか"):
            is_reshoot = True
            continue
        return


def run(n: int, save_raw: bool = False) -> None:
    """対話ループ本体。webcam を起動時に1度だけ open し、全距離でその1接続を使い回す（docs/module-calibration.md）。"""
    if not _startup_guard():
        print("中止しました（散布ガード）。")
        return

    source = WebcamSource()
    try:
        source.open()
    except RuntimeError as e:
        # open 失敗（タイムアウト/実フィード未検出）等は原因を示してクリーンに終了。
        # 撮影ループ内で起きる RuntimeError（imwrite 失敗等）はここでは捕まえない（誤吸収防止）。
        print(f"webcam を開けませんでした: {e}")
        return

    try:
        while True:
            dist = prompt_distance()
            if dist is None:
                break
            try:
                _capture_one_distance(source, dist, n, save_raw=save_raw)
            except Exception as e:
                # 1 距離分の処理で起きた例外（save_calib_png の imwrite 失敗 RuntimeError・
                # cv2.error・ValueError 等）は、この距離だけスキップして原因を表示し、
                # セッションは継続する（10 距離キャンペーン中の 1 失敗で全滅させない・docs/requirements-and-status.md）。
                # ここでは release しない（webcam は後続距離で使い回す。release は外側 finally のみ）。
                # 捕捉は Exception まで。KeyboardInterrupt / SystemExit は Exception 派生でないため
                # 捕まえず外側 finally の release → main() の Ctrl-C 処理へ素通しする（誤吸収防止）。
                # 「webcam を開けませんでした」は open 失敗時専用であり、こことは別事象。
                print(f"  → {dist}mm の処理でエラーが発生しました: {type(e).__name__}: {e}")
                print("  → この距離をスキップします（webcam は解放しません）。次の距離入力に戻ります。")
                continue
    finally:
        source.release()  # 正常終了・ループ内例外・Ctrl-C のいずれでも確実に release

    # 終了サマリ: 保存済み距離 PNG の一覧
    dist_pngs, _ = scan_calib_dir()
    print(f"=== 終了: {config.CALIB_IMGS_DIR} の距離 PNG（{len(dist_pngs)} 件） ===")
    for p in dist_pngs:
        print(f"  - {p.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FR-8 webcam 校正画像の取得（バースト平均 → preprocess → PNG）")
    parser.add_argument("--n", type=int, default=DEFAULT_BURST_N,
                        help=f"バースト枚数（既定 {DEFAULT_BURST_N}）")
    parser.add_argument("--save-raw", action="store_true",
                        help=f"バースト平均の raw (1280x720) も {config.CALIB_RAW_DIR.name}/<距離>.png に保存する"
                             "（処理解像度の実験用・2026-09-16）")
    args = parser.parse_args()
    if args.n < 1:
        raise ValueError(f"--n は 1 以上が必要です（渡された値: {args.n}）")

    try:
        run(args.n, save_raw=args.save_raw)
    except KeyboardInterrupt:
        # run() の finally で release 済み。メッセージのみ出して終了
        print("\n中断しました（Ctrl-C）。webcam は解放済みです。")


if __name__ == "__main__":
    main()
