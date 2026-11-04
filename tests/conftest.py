"""tests/ から root の config / core / calibration / realtime を import するための
ブートストラップ（他モジュールの sys.path.insert 方式と同一。project_knowledge/architecture/architecture.md の HW 非依存テスト方針）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
