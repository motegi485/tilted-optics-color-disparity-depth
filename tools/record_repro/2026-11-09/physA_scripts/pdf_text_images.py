# -*- coding: utf-8 -*-
"""PDF の指定ページのテキストと埋め込み画像を取り出す（読み取りのみ）。
入力: argv[1]=PDF, argv[2]=出力フォルダ, argv[3]=ページ番号（1 始まり, カンマ区切り）
出力: 標準出力にテキスト、出力フォルダに画像
"""
import sys
from pathlib import Path
from pypdf import PdfReader

pdf, outdir, pages = sys.argv[1], Path(sys.argv[2]), [int(p) for p in sys.argv[3].split(",")]
outdir.mkdir(parents=True, exist_ok=True)
r = PdfReader(pdf)
print("pages:", len(r.pages))
for p in pages:
    pg = r.pages[p - 1]
    print(f"===== page {p} =====")
    t = pg.extract_text() or ""
    sys.stdout.buffer.write(t.encode("utf-8", "replace"))
    print()
    try:
        for i, im in enumerate(pg.images):
            fn = outdir / f"p{p}_{i}_{im.name}"
            fn.write_bytes(im.data)
            print("image:", fn, len(im.data))
    except Exception as e:
        print("image extraction error:", e)
