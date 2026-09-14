"""evaluate_distance_accuracy.py — 距離別精度評価（ライブ経路・校正と独立な距離）の集計。

realtime.runner が保存した outputs/{session}/{i:06d}.npy（深度マップ・float32・NaN 保持・
唯一の正本）を、セッションごとの正解距離（レーザ距離計の実測 [mm]）と比較し、画素単位の
誤差統計を出す。深度の再計算はしない（保存済み npy をそのまま集計する。再現一致の確認は
checks.verify_clip_replay の役割）。

集計の定義（checks.evaluate の回帰アンカーと同じ流儀・docs/verification-and-tools.md）:
  - 有効画素: gate2 と同じ config.DEPTH_MIN < z < config.DEPTH_MAX を満たす画素
    （NaN は比較で False となり自然に除外される）
  - 誤差 e = z_est − z_gt [mm]（画素ごと・float64 で集計）
  - 全フレーム統合（pooled）: 有効画素数・有効率・平均推定距離・平均誤差（符号付き＝bias）・
    MAE・誤差の標準偏差（母標準偏差）・RMSE・誤差の中央値
  - フレーム間: 各フレームの平均推定距離の平均と標準偏差・有効画素数の min/max
  - 任意: ROI（処理解像度の画素座標 x0,y0,x1,y1・右下は排他的。既定は全画面）、
    先頭フレームのスキップ、集計する最大フレーム数
  - 任意: runner の実行ログ（--log-dir/{session}.txt）から、保存・表示込みの
    E2E スループット [fps] と取りこぼし率（camera frame id に対する未処理フレームの割合）を推定

正解距離の指定は2通り:
  - --sessions NAME [NAME ...]  セッション名末尾の `_<数字>` を正解距離 [mm] と解釈（例: dist_1500）
  - --manifest CSV              列 session,gt_mm[,note]（数字以外のセッション名を使うとき）

実行例:
  uv run python -m checks.evaluate_distance_accuracy --sessions dist_1500 dist_1700 dist_1900
  uv run python -m checks.evaluate_distance_accuracy --manifest outputs\\distance_manifest.csv `
      --skip-first 10 --log-dir outputs\\logs --out outputs\\distance_accuracy.csv
"""
from __future__ import annotations

# --- checks/ から root の config を import するためのブートストラップ ---
# 推奨実行は `uv run python -m checks.<name>`（root を cwd に）。
# 直接実行 `python checks/<name>.py` でも動くよう root を sys.path に通す（他の checks と同方式）。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import re
from datetime import datetime

import numpy as np

import config


# 出力 CSV の列順（print の表もこの順）。docs/verification-and-tools.md に列の定義を記載。
COLUMNS = [
    "session", "gt_mm", "frames_used", "frames_total", "roi",
    "n_valid_total", "n_roi_px_per_frame", "valid_rate",
    "mean_est_mm", "mean_err_mm", "mae_mm", "std_err_mm", "rmse_mm", "median_err_mm",
    "frame_mean_est_avg_mm", "frame_mean_est_std_mm", "n_valid_min", "n_valid_max",
    "e2e_fps", "camera_fps_est", "drop_ratio", "log_span_s",
    "params_path", "note",
]

_GT_SUFFIX = re.compile(r"_(\d+)$")
# runner.py の logging.basicConfig 形式: "YYYY-MM-DD HH:MM:SS,mmm name LEVEL processed=N (frame id=M): ..."
# 行頭を固定しない（PowerShell の 2>&1 リダイレクトが行頭に接頭辞を付けることがあるため）。
_LOG_LINE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) .*?processed=(?P<processed>\d+) "
    r"\(frame id=(?P<fid>\d+)\)")


def _check_gt(gt: int, where: str) -> int:
    """正解距離 [mm] が gate2 のレンジ内（DEPTH_MIN < gt < DEPTH_MAX）であることを検証する。

    runner 既定のセッション名（YYYYmmdd_HHMMSS）を誤って渡すと末尾の時刻が距離として
    解釈されてしまうため、レンジ外は拒否する。
    """
    if not (config.DEPTH_MIN < gt < config.DEPTH_MAX):
        raise ValueError(f"{where}: 正解距離 {gt} mm が評価レンジ "
                         f"({config.DEPTH_MIN} < gt < {config.DEPTH_MAX}) の外です。"
                         f"セッション名の末尾 `_<数字>` が距離 [mm] になっているか確認してください")
    return gt


def parse_gt_from_session_name(name: str) -> int:
    """セッション名末尾の `_<数字>` を正解距離 [mm] として返す（例: dist_1500 -> 1500）。"""
    m = _GT_SUFFIX.search(name)
    if m is None:
        raise ValueError(f"セッション名 '{name}' の末尾に `_<数字>`（正解距離 mm）がありません。"
                         f"--manifest で正解距離を指定してください")
    return _check_gt(int(m.group(1)), f"セッション名 '{name}'")


def load_manifest(path: Path) -> list[tuple[str, int, str]]:
    """CSV（列: session,gt_mm[,note]）を読み、(session, gt_mm, note) のリストを返す。"""
    if not path.is_file():
        raise FileNotFoundError(f"manifest が見つかりません: {path}")
    rows: list[tuple[str, int, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "session" not in reader.fieldnames or "gt_mm" not in reader.fieldnames:
            raise ValueError(f"manifest には session,gt_mm 列が必要です: {path} (列={reader.fieldnames})")
        for i, row in enumerate(reader, start=2):
            session = (row.get("session") or "").strip()
            gt_txt = (row.get("gt_mm") or "").strip()
            if not session or not gt_txt:
                raise ValueError(f"manifest {path} の {i} 行目: session / gt_mm が空です")
            try:
                gt = int(gt_txt)
            except ValueError as e:
                raise ValueError(f"manifest {path} の {i} 行目: gt_mm が整数ではありません: {gt_txt!r}") from e
            _check_gt(gt, f"manifest {path} の {i} 行目")
            rows.append((session, gt, (row.get("note") or "").strip()))
    if not rows:
        raise ValueError(f"manifest にデータ行がありません: {path}")
    return rows


def parse_roi(text: str | None, width: int, height: int) -> tuple[int, int, int, int]:
    """'x0,y0,x1,y1'（右下排他）を検証して返す。None は全画面。"""
    if text is None:
        return (0, 0, width, height)
    parts = text.split(",")
    if len(parts) != 4:
        raise ValueError(f"--roi は x0,y0,x1,y1 の4整数: {text!r}")
    try:
        x0, y0, x1, y1 = (int(p) for p in parts)
    except ValueError as e:
        raise ValueError(f"--roi は整数のみ: {text!r}") from e
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"--roi が処理解像度 (W={width}, H={height}) の範囲外か空です: {text!r}")
    return (x0, y0, x1, y1)


def list_depth_files(out_dir: Path) -> list[Path]:
    """runner が保存した連番 npy（??????.npy・数字 stem）を昇順で返す。"""
    return sorted(p for p in out_dir.glob("??????.npy") if p.stem.isdigit())


def summarize_errors(errors: np.ndarray) -> dict[str, float]:
    """有効画素の誤差列（float64・1次元・空でない）から pooled 統計を返す。"""
    if errors.ndim != 1 or errors.size == 0:
        raise ValueError("errors は空でない1次元配列である必要があります")
    return {
        "mean_err_mm": float(np.mean(errors)),
        "mae_mm": float(np.mean(np.abs(errors))),
        "std_err_mm": float(np.std(errors)),           # 母標準偏差（ddof=0）
        "rmse_mm": float(np.sqrt(np.mean(errors ** 2))),
        "median_err_mm": float(np.median(errors)),
    }


def summarize_session(out_dir: Path, gt_mm: int, *, roi: tuple[int, int, int, int],
                      skip_first: int = 0, max_frames: int | None = None,
                      depth_min: float = config.DEPTH_MIN, depth_max: float = config.DEPTH_MAX) -> dict:
    """1セッションの深度 npy 群を集計して COLUMNS のサブセット dict を返す。

    有効画素が0件のときは n_valid_total=0 とし、誤差統計は NaN で埋める（呼び出し側が判定）。
    """
    if skip_first < 0:
        raise ValueError(f"--skip-first は0以上: {skip_first}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"--max-frames は1以上: {max_frames}")
    files = list_depth_files(out_dir)
    if not files:
        raise FileNotFoundError(f"{out_dir} に連番 npy（??????.npy）がありません")
    used = files[skip_first:]
    if max_frames is not None:
        used = used[:max_frames]
    if not used:
        raise ValueError(f"{out_dir}: skip/max の指定で集計フレームが0件になりました "
                         f"(total={len(files)}, skip_first={skip_first}, max_frames={max_frames})")

    x0, y0, x1, y1 = roi
    n_roi = (x1 - x0) * (y1 - y0)
    per_frame_mean: list[float] = []
    per_frame_valid: list[int] = []
    error_chunks: list[np.ndarray] = []
    shape_ref: tuple[int, ...] | None = None
    for p in used:
        depth = np.load(str(p))
        if depth.ndim != 2:
            raise ValueError(f"{p}: 深度マップは2次元配列のはずです (shape={depth.shape})")
        if shape_ref is None:
            shape_ref = depth.shape
            h, w = shape_ref
            if not (x1 <= w and y1 <= h):
                raise ValueError(f"{p}: ROI {roi} が深度マップの形状 (H={h}, W={w}) を超えています")
        elif depth.shape != shape_ref:
            raise ValueError(f"{p}: フレーム間で形状が不一致 ({depth.shape} != {shape_ref})")
        sub = depth[y0:y1, x0:x1].astype(np.float64)
        valid = sub[(sub > depth_min) & (sub < depth_max)]   # gate2 と同じ。NaN は自動的に落ちる
        per_frame_valid.append(int(valid.size))
        if valid.size:
            per_frame_mean.append(float(np.mean(valid)))
            error_chunks.append(valid - float(gt_mm))

    result: dict = {
        "gt_mm": gt_mm,
        "frames_used": len(used),
        "frames_total": len(files),
        "roi": f"{x0},{y0},{x1},{y1}",
        "n_roi_px_per_frame": n_roi,
        "n_valid_min": int(min(per_frame_valid)),
        "n_valid_max": int(max(per_frame_valid)),
    }
    n_valid_total = int(sum(per_frame_valid))
    result["n_valid_total"] = n_valid_total
    result["valid_rate"] = n_valid_total / (n_roi * len(used))
    if n_valid_total == 0:
        result.update({k: float("nan") for k in
                       ("mean_est_mm", "mean_err_mm", "mae_mm", "std_err_mm", "rmse_mm", "median_err_mm",
                        "frame_mean_est_avg_mm", "frame_mean_est_std_mm")})
        return result
    errors = np.concatenate(error_chunks)
    result["mean_est_mm"] = float(np.mean(errors) + gt_mm)
    result.update(summarize_errors(errors))
    fm = np.asarray(per_frame_mean, dtype=np.float64)
    result["frame_mean_est_avg_mm"] = float(np.mean(fm))
    # 有効画素のあるフレームが1枚だけなら ddof=1 は定義できないので NaN
    result["frame_mean_est_std_mm"] = float(np.std(fm, ddof=1)) if fm.size >= 2 else float("nan")
    return result


def parse_runner_log(path: Path) -> dict[str, float] | None:
    """runner の実行ログから E2E スループットを推定する。

    ログは log_every（既定30）フレームごとに `processed=N (frame id=M)` を時刻付きで出す。
    先頭と末尾の該当行の間で
      e2e_fps        = Δprocessed / Δt   （保存・表示・取得待ちを含む実効処理レート）
      camera_fps_est = Δfid / Δt         （取得スレッドが受け取ったフレームのレート）
      drop_ratio     = 1 − Δprocessed/Δfid（処理が追いつかず捨てられたフレームの割合）
    を返す。該当行が2本未満なら None。
    """
    if not path.is_file():
        return None
    first: tuple[datetime, int, int] | None = None
    last: tuple[datetime, int, int] | None = None
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _LOG_LINE.search(line)
            if m is None:
                continue
            ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
            rec = (ts, int(m.group("processed")), int(m.group("fid")))
            if first is None:
                first = rec
            last = rec
    if first is None or last is None or last is first:
        return None
    span = (last[0] - first[0]).total_seconds()
    d_proc = last[1] - first[1]
    d_fid = last[2] - first[2]
    if span <= 0 or d_proc <= 0:
        return None
    return {
        "e2e_fps": d_proc / span,
        "camera_fps_est": d_fid / span if d_fid > 0 else float("nan"),
        "drop_ratio": (1.0 - d_proc / d_fid) if d_fid > 0 else float("nan"),
        "log_span_s": span,
    }


def _read_params_path(out_dir: Path) -> str:
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        return ""
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(meta.get("params_path", ""))


def _fmt(v) -> str:
    if isinstance(v, float):
        return "nan" if np.isnan(v) else f"{v:.4g}" if abs(v) < 1e-2 else f"{v:.2f}"
    return str(v)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in COLUMNS})


def main(argv: list[str] | None = None) -> int:
    # Windows の日本語コンソール(cp932)は非対応文字の print で UnicodeEncodeError を出すため
    # UTF-8 に固定する（realtime/runner.py の main() 冒頭と同じ対処）。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="距離別精度評価（保存済み深度 npy と正解距離の誤差統計）")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sessions", nargs="+", metavar="NAME",
                     help="outputs/{NAME} を集計。末尾の `_<数字>` を正解距離 [mm] と解釈（例: dist_1500）")
    src.add_argument("--manifest", metavar="CSV",
                     help="列 session,gt_mm[,note] の CSV でセッションと正解距離を指定")
    parser.add_argument("--roi", default=None,
                        help="評価領域 x0,y0,x1,y1（処理解像度の画素座標・右下排他）。既定: 全画面")
    parser.add_argument("--skip-first", type=int, default=0,
                        help="各セッションの先頭からこの枚数を集計から除く（起動直後の暖機用。既定0）")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="各セッションで集計する最大フレーム数（既定: 全部）")
    parser.add_argument("--log-dir", default=None,
                        help="runner の実行ログ {session}.txt を置いたディレクトリ（E2E スループット推定用・任意）")
    parser.add_argument("--out", default=None,
                        help="集計結果 CSV の出力先（既定: outputs/distance_accuracy_<日時>.csv）")
    args = parser.parse_args(argv)

    width, height = config.IMAGE_SIZE
    try:
        roi = parse_roi(args.roi, width, height)
        if args.manifest is not None:
            targets = load_manifest(Path(args.manifest))
        else:
            targets = [(name, parse_gt_from_session_name(name), "") for name in args.sessions]
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}")
        return 2

    out_root = Path(config.OUTPUTS_DIR)
    log_dir = Path(args.log_dir) if args.log_dir else None
    rows: list[dict] = []
    any_empty = False
    for session, gt, note in targets:
        out_dir = out_root / session
        if not out_dir.is_dir():
            print(f"[ERROR] outputs セッションが見つかりません: {out_dir}")
            return 2
        try:
            r = summarize_session(out_dir, gt, roi=roi, skip_first=args.skip_first, max_frames=args.max_frames)
        except (ValueError, FileNotFoundError) as e:
            print(f"[ERROR] {session}: {e}")
            return 2
        r["session"] = session
        r["note"] = note
        r["params_path"] = _read_params_path(out_dir)
        if log_dir is not None:
            stats = parse_runner_log(log_dir / f"{session}.txt")
            if stats is None:
                print(f"  [INFO] {session}: 実行ログ {log_dir / (session + '.txt')} が無いか該当行が2本未満のため "
                      f"E2E スループットは未算出")
            else:
                r.update(stats)
        if r["n_valid_total"] == 0:
            any_empty = True
            print(f"[WARN] {session}: 有効画素が0件（gate2 {config.DEPTH_MIN}<z<{config.DEPTH_MAX} と ROI {r['roi']}）")
        rows.append(r)

    # 表示（正解距離順ではなく指定順のまま。列は主要なものに絞る）
    show = ["session", "gt_mm", "frames_used", "n_valid_total", "valid_rate",
            "mean_est_mm", "mean_err_mm", "mae_mm", "std_err_mm", "rmse_mm",
            "frame_mean_est_std_mm", "e2e_fps", "drop_ratio"]
    print(f"gate2: {config.DEPTH_MIN} < z < {config.DEPTH_MAX} [mm]  roi={rows[0]['roi']}  "
          f"skip_first={args.skip_first}  max_frames={args.max_frames}")
    print("  ".join(show))
    for r in rows:
        print("  ".join(_fmt(r.get(k, "")) for k in show))

    out_path = Path(args.out) if args.out else out_root / f"distance_accuracy_{datetime.now():%Y%m%d_%H%M%S}.csv"
    write_csv(rows, out_path)
    print(f"\nCSV: {out_path}")
    return 1 if any_empty else 0


if __name__ == "__main__":
    sys.exit(main())
