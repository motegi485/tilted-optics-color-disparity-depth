"""B5: runner ログから E2E（checks.evaluate_distance_accuracy.parse_runner_log の定義）と、ログ行の計算時間の要約を出す。

E2E: 先頭と末尾の `processed=N (frame id=M)` 行の差分で e2e_fps=ΔN/Δt・camera_fps_est=ΔM/Δt・drop_ratio=1−ΔN/ΔM。
計算時間: 各ログ行の `X ms`（runner.py の dt＝preprocess＋process_frame・30 処理ごとに 1 行の抜き取り。p95 ではない）の
min／median／max（processed=0 の行は除く）。`最終 fps`（計算時間だけの EMA）は E2E ではないので出さない。
ログは UTF-8（変換後）なら parse_runner_log を呼び、in-memory の計算と一致することを確かめる。UTF-16 のままなら
in-memory だけで計算し、その旨を列 source に書く。
"""
from __future__ import annotations

import csv
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from checks.evaluate_distance_accuracy import _LOG_LINE, parse_runner_log

_MS = re.compile(r"processed=(\d+) \(frame id=\d+\): ([\d.]+) ms")
CONDITION = {"b5_scene_01": "record (PNG+npy+viz), typical scene",
             "b5_e2e_norecord_scene": "no-record (display on), typical scene",
             "b5_e2e_norecord_noise": "no-record (display on), noise board 2850 mm full frame",
             "b5_e2e_norecord_noise_1150_unintended": "REFERENCE ONLY: no-record, noise board 1150 mm (unintended run)"}
COLUMNS = ["log", "condition", "source", "e2e_fps", "camera_fps_est", "drop_ratio", "log_span_s", "n_log_lines",
           "compute_ms_min", "compute_ms_median", "compute_ms_max"]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    logs_dir, out_csv, names = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    rows = []
    for n in names:
        p = logs_dir / f"{n}.txt"
        raw = p.read_bytes()
        utf16 = raw[:2] == b"\xff\xfe"
        text = raw.decode("utf-16") if utf16 else raw.decode("utf-8")
        recs, ms = [], []
        for line in text.splitlines():
            m = _LOG_LINE.search(line)
            if m:
                recs.append((datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S,%f"),
                             int(m.group("processed")), int(m.group("fid"))))
            k = _MS.search(line)
            if k and int(k.group(1)) > 0:
                ms.append(float(k.group(2)))
        (t0, p0, f0), (t1, p1, f1) = recs[0], recs[-1]
        span = (t1 - t0).total_seconds()
        mem = {"e2e_fps": (p1 - p0) / span, "camera_fps_est": (f1 - f0) / span,
               "drop_ratio": 1.0 - (p1 - p0) / (f1 - f0), "log_span_s": span}
        if utf16:
            source = "in-memory (UTF-16 log; parse_runner_log would return None)"
        else:
            official = parse_runner_log(p)
            if official != mem:
                print(f"[ERROR] {n}: parse_runner_log {official} != in-memory {mem}")
                return 1
            source = "parse_runner_log (== in-memory)"
        rows.append({"log": n, "condition": CONDITION.get(n, "record (PNG+npy+viz), noise board full frame"),
                     "source": source, **mem, "n_log_lines": len(recs),
                     "compute_ms_min": min(ms), "compute_ms_median": statistics.median(ms), "compute_ms_max": max(ms)})
        r = rows[-1]
        print(f"{n:40s} e2e={r['e2e_fps']:.4f} cam={r['camera_fps_est']:.4f} drop={r['drop_ratio']:.4f} "
              f"span={r['log_span_s']:.3f} lines={r['n_log_lines']} ms={r['compute_ms_min']}/{r['compute_ms_median']}/"
              f"{r['compute_ms_max']}")
    with open(out_csv, "x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
