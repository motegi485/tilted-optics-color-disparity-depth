# テーマ 1 の計算スクリプト（2026-11-06〜07）

論文リポの `研究計画_テーマ1.md` と `研究方針_アオリの意義.md` の計算を再現する読み取り専用のスクリプト。研究コードと校正値は変更しない。

## 実行

リポジトリ直下から PowerShell で実行する。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONUTF8='1'
uv run --no-sync python -u tools/record_repro/2026-11-07/theme1_physics/plan_window3.py
```

## 内容

| フォルダ | 内容 |
|---|---|
| `theme1_physics/` | 薄レンズの光線追跡（`optics.py`）、配置の比較（`plan_*.py`）、計画の前半の検証（`q*.py`） |
| `theme1_physics_verify/` | 傾きと K の関係（`v1`）、場面ごとの比較（`v2`）、帯と誤差（`v3`）、感度の圧縮（`v4`〜`v9`）、文献の 1/z 幅（`v10`） |
| `theme1_plan_check/` | `b5_scene_01` の印刷紙の再集計（`refute_data3.py` が彩度別と行帯別の MAE を出す）。`refute_data2.py` は確認用の画像を一時フォルダに書く |

計算は薄レンズの範囲で、実機では未確認。実データを読むスクリプトは `data/` の校正画像や記録を読むので、手元にデータがないと動かない。
