"""B5: runner の実行ログ（PowerShell Tee-Object の UTF-16 LE）を UTF-8（BOM なし）・CRLF の写しにする。

原本（outputs/logs/<name>.txt）は改名も変更もしない。写しは記録フォルダの logs/ に排他作成し、集計
（checks.evaluate_distance_accuracy --log-dir）はこの写しを読む。
各ファイルで、(1) 原本の先頭が UTF-16 LE BOM であること、(2) 写しを UTF-8 で読み戻した文字列が原本の
UTF-16 デコード結果と改行以外で一致すること、を確かめる。

実行（開発リポの root）:
  uv run python tools/record_repro/2026-10-01/b5_logs_to_utf8.py --src outputs/logs --dst project_knowledge/records/2026-10-01/logs \
      --names b5_dist_1000 ... [--rename OLD=NEW ...]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原本のフォルダ（UTF-16 の <name>.txt）")
    ap.add_argument("--dst", required=True, help="写しを置くフォルダ")
    ap.add_argument("--names", nargs="+", required=True, help="拡張子なしのログ名")
    ap.add_argument("--rename", nargs="*", default=[], help="写しの名前を変える場合 OLD=NEW（拡張子なし）")
    args = ap.parse_args()
    ren = dict(r.split("=", 1) for r in args.rename)
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        p = src / f"{name}.txt"
        raw = p.read_bytes()
        if raw[:2] != b"\xff\xfe":
            print(f"[ERROR] {p}: UTF-16 LE BOM で始まっていません（先頭 {raw[:4]!r}）")
            return 1
        text = raw.decode("utf-16")
        lines = text.splitlines()
        out = ("\r\n".join(lines) + "\r\n").encode("utf-8")
        q = dst / f"{ren.get(name, name)}.txt"
        with q.open("xb") as f:  # 排他作成（既存なら FileExistsError で止まる）
            f.write(out)
        back = q.read_bytes().decode("utf-8").splitlines()
        if back != lines:
            print(f"[ERROR] {q}: 読み戻しが原本と一致しません")
            return 1
        print(f"{p} (utf-16, {len(raw)} B, sha256 {hashlib.sha256(raw).hexdigest()[:16]}) -> {q} "
              f"(utf-8 crlf, {len(out)} B, {len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
