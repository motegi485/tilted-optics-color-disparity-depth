# tilted-optics-color-disparity-depth

アオリ光学系と分割型カラーフィルタ開口を組み合わせた単眼距離推定のリアルタイム実装。入出力・表示は Python（uv）、視差計算は Rust 拡張（PyO3 / maturin）。`core/disparity.py` は `rust/disparity/` の拡張への薄いラッパーである。過去の実装例として公開しており、基本的に保守・更新はしない。

## 文書

- 利用者向けの入口は `README.md`（原理・処理の流れ・コードの読み順・実行手順）。
- `LICENSE`（BSD-3-Clause）、`NOTICE`（元にした MIT コードの表示）、`CITATION.cff`（引用のお願い）。
- 図は `assets/readme/`、データ無しで試すサンプルは `assets/sample/`。
- **`CLAUDE.md` と `AGENTS.md` は同一内容を保つ。** 片方を変えたらもう片方も変える。

## コマンド（Windows 11 / PowerShell）

- Python 実行（uv）: `uv run python <script>.py` ／ パッケージ追加: `uv add <pkg>` ／ 依存同期: `uv sync`
- 校正 → 推論（リポジトリ root を cwd に）: `uv run python -m calibration.calibrate` → `uv run python -m calibration.fit_params` → `uv run python -m checks.evaluate`
- 検証ハーネス（`checks/`）: `uv run python -m checks.evaluate` ／ `uv run python -m checks.verify_pipeline_equivalence` ／ `uv run python -m checks.verify_clip_replay --session <name>`
- 自動テスト（`tests/`・HW 非依存の pytest）: `uv run python -m pytest`。dev 依存の追加は `uv add --dev <pkg>`
- ベンチ: `uv run python -m checks.bench_disparity --impl native --source captures --session <name>`（既定の `--source synthetic` は最悪ケース寄りの入力で、予算超の exit 1 になる）
- Rust↔Python 照合: `uv run python -m checks.verify_disparity_rust`（実写の入力が無い環境では `--allow-synthetic-only`）
- Rust（cargo。`rust/disparity/` で実行。`cargo build`/`clippy` は `$env:PYO3_PYTHON` に venv の python.exe を指定するか有効な venv 下で実行する）: `cargo build` ／ `cargo test` ／ `cargo clippy`
- Rust → Python 導入（**`maturin develop` は uv の venv では不可**＝pip 非搭載）: `rust/disparity/` で `maturin build --release --interpreter ..\..\.venv\Scripts\python.exe` → `uv pip install (Get-ChildItem target\wheels\*.whl).FullName`。**`uv sync` を実行すると手動導入した Rust 拡張（`_rt_disparity_native`）は削除される**ため、`uv sync` 後は上記を再実行する
- ※シェルは PowerShell。パス区切りは `\`、ワイルドカード展開は外部コマンドへ自動では効かないため `Get-ChildItem` 等を使う。

## 規約

- `core/` は純関数（config import・ファイル I/O・表示・print 禁止。依存はすべて引数注入）。唯一の例外は `core/params.py`（CSV I/O 可）。
- `core/_disparity_reference.py` は凍結された参照実装で、関数本体は変更しない。
- 全定数・パスは `config.py` に一元管理し、呼び出し側が core へ注入する。
- 無効値は `np.nan`（0.0 埋めは禁止。「視差ちょうど 0＝合焦面上」と「無効」を区別するため）。
- 入力検証は `raise ValueError`（`assert` は `-O` で無効化されるため禁止）。
- 命名: チャンネルは `red`/`blue`（`std`/`tmp` 禁止）、ZNCC 類似度は `sim_neg`/`sim_peak`/`sim_pos`、視差は `disp_map`（深度は `depth_map`）、参照式係数は `pos`/`neg` 接尾辞（前後ぼけの物理ラベルは使わない）。
- `params/`・`params/baseline/` は実行・回帰用の入力であり、文書として扱わない。
