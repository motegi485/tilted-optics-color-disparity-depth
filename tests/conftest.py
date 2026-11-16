"""tests/ から root の config / core / calibration / realtime を import するための
ブートストラップ（他モジュールの sys.path.insert 方式と同一）。テストはカメラなどのハードウェアに依存しない。

視差計算（core/disparity.py）は Rust 拡張 `_rt_disparity_native` を使うため、拡張が未導入のままだと
大半のテストが import の時点で失敗する。その場合は、個々のエラーを大量に出す代わりに、導入手順を示して
テスト全体を止める（README の「実行手順」を参照）。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import _rt_disparity_native  # noqa: F401
except ImportError as _e:
    pytest.exit(
        "Rust 拡張 `_rt_disparity_native` が導入されていないため、テストを実行できません: "
        f"{_e}\n"
        "rust/disparity/ で `maturin build --release --interpreter ..\\..\\.venv\\Scripts\\python.exe` を実行し、"
        "`uv pip install (Get-ChildItem target\\wheels\\*.whl).FullName` で導入してください"
        "（`uv sync` を実行すると導入済みの拡張は消えるので、その後もやり直しが必要です）。",
        returncode=2,
    )
