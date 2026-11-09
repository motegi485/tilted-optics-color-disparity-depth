# テーマ 2 の計算・再解析スクリプト（2026-11-09）

論文リポの `研究計画_テーマ2.md` と `調査メモ_2026-11-09/` の計算を再現する読み取り専用のスクリプト。研究コード、校正値、撮影データは変更しない。

## 実行

リポジトリ直下から PowerShell で実行する。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONUTF8='1'
uv run --frozen python -u tools/record_repro/2026-11-09/physA_scripts/a1_table_check.py
```

出力は各フォルダの隣の `*_out/`（例: `physA_scripts` → `physA_out/`）に書かれる。出力は再生成できるので、このフォルダには置いていない。フォルダ名は調査時の担当名のままにしてある（スクリプトどうしが名前で参照するため）。

## 内容

| フォルダ | 内容 | 対応するメモ |
|---|---|---|
| `physA_scripts/` | 開口形状の幾何（重心・基線・光量・Fisher）、光量と全方向の基線の上限、設計空間の全探索、実フィルタの近似（分光は仮定）、既存の撮影画像のエッジの向きの集計。`aperture_lib.py`・`catalog.py` が共通部品。`a11` は `a2`・`a10` の後に実行 | `design_space_analysis.md` |
| `physB_scripts/` | 任意の開口形状で RGB 画像を合成し、推定法（現行の R–B 横 ZNCC・color-line・PSF 合わせなど）ごとにエッジの向きとぼけ量に対する誤差を出す。`apsim.py` が共通部品 | `synthesis_simulation.md` |
| `critic_scripts/` | フィルタの取り付け位置・貼りずれと基線（`c1`）、校正板画像の左右・上下の色比（`c2`、`data/calib_raw/` を読む） | `critique_and_gaps.md` |
| `verPhys_scripts/` | 原理と上限の独立な再導出、2 種類のフィルタで重心が一直線に並ぶことの確認（`v9`）、color-line のぼけ量依存（`v10`・`v11`。`v11` は `physB_scripts/apsim.py` を読む）、アオリの合成（`v6`） | `verify_physics.md` |
| `verReal_scripts/` | 既存の実撮影データ（`data/captures/b5_dist_*`・`b5_scene_01`・2026-07 の屋内の場面）の再解析。現行の Rust 実装と、画素の選び方・照合を変えた推定法を比べる。`vr_lib.py` が共通部品 | `verify_real_data.md` |
| `litB_scripts/` | エッジの向きに対する実効基線と誤差倍率（`edge_angle_gain.py`） | `lit_adjacent_and_lineage.md` |

数値はすべて理想化（幾何光学、回折なし）か、既存画像の非公式な再集計。実機の新しい形状では何も確かめていない。実データを読むスクリプトは `data/` の撮影画像と `params/` の live 係数を読むので、手元にデータがないと動かない。
