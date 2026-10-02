"""B5: 距離別精度の manifest（列 session,gt_mm,note）を BOM なし UTF-8・CRLF で作る（列は既存 outputs/distance_manifest_*.csv と同じ。既存の outputs 側は LF、記録フォルダの写しは CRLF。load_manifest はどちらも読める）。

gt_mm = 公称値（全 15 距離でレーザを公称値に合わせ込み）。
note は距離の区分（ASCII にそろえる）。b5_scene_01 は入れない
（単一の正解距離が無い。--sessions に渡すと末尾 _01 が gt=1 mm と解釈される）。既存ファイルは上書きしない。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from checks.evaluate_distance_accuracy import load_manifest

NEAR, FAR = (1000, 1150, 1250), (3450, 3750, 4000)
INTERIOR = (1450, 1650, 1850, 2050, 2250, 2450, 2650, 2850, 3050)
NOTE = {"near": "extrapolation (near; outside calibrated range 1400-3200)",
        "interior": "held-out (B5 independent; inside calibrated range 1400-3200)",
        "far": "extrapolation (far; outside calibrated range 1400-3200)"}


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/distance_manifest_b5.csv")
    rows = [(f"b5_dist_{mm}", mm, NOTE["near"]) for mm in NEAR]
    rows += [(f"b5_dist_{mm}", mm, NOTE["interior"]) for mm in INTERIOR]
    rows += [(f"b5_dist_{mm}", mm, NOTE["far"]) for mm in FAR]
    with out.open("x", encoding="utf-8", newline="") as f:     # 排他作成・BOM なし・csv は \r\n
        w = csv.writer(f)
        w.writerow(["session", "gt_mm", "note"])
        w.writerows(rows)
    back = load_manifest(out)                                   # 集計ツールと同じ読み方で検証
    assert [(s, g, n) for s, g, n in back] == rows, "manifest の読み戻しが一致しません"
    assert len(back) == 15 and out.read_bytes()[:3] != b"\xef\xbb\xbf"
    for s, g, n in back:
        print(s, g, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
