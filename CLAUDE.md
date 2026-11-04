# tilted-optics-color-disparity-depth

アオリ光学系と分割型カラーフィルタ開口を組み合わせた単眼距離推定のリアルタイム実装。入出力・表示は Python（uv）、視差計算は Rust 拡張（PyO3 / maturin）。`core/disparity.py` は `rust/disparity/` の拡張への薄いラッパーである。

## 内部文書（project_knowledge）

要件・設計・理論・進捗・未決事項・実験記録などの内部文書は、このリポジトリには置かない。正本は、リポジトリ直下の `project_knowledge/` に置く別の Git リポジトリ（GitHub の Private リポジトリ `tilted-optics-color-disparity-depth-knowledge`）である。`project_knowledge/` は `.gitignore` 済みで、このリポジトリの Git では追跡せず、submodule にもしない。

- **開始時**: `project_knowledge/INDEX.md` と `project_knowledge/progress/CURRENT.md` を読む → INDEX のルーティングで関連する正本を読む → 実装を確認してから作業に入る。
- **変更時**: 実装・仕様・既知の制約に影響する変更をしたら、同じ作業の中で `project_knowledge/` の該当文書も更新する。
- **終了時**: `project_knowledge/progress/CURRENT.md` の現在地・未決事項・次作業を更新する。
- **新しい内部文書もすべて `project_knowledge/` に作る。** このリポジトリに `docs/` などの内部文書を作らない。内部研究情報（評価値、未決事項、判断の経緯など）をこのリポジトリの文書に重複して書かない。コメントから参照するときは `project_knowledge/<パス>` と書く。
- **文書と実装の食い違い**: 判断の根拠は内部文書を正とし、実際の挙動の一次情報はコードと実行結果である。食い違いを見つけたら確かめて報告する。研究上の手法・評価方法・受入基準・係数は独断で変えない。
- `project_knowledge/` の commit・push は、ユーザーの指示がある場合に限る。秘密情報を書かない。

### `project_knowledge/` が無いとき

- 代わりの文書をこのリポジトリ内に作らない。
- ユーザーに配置を依頼する（例: `git clone https://github.com/motegi485/tilted-optics-color-disparity-depth-knowledge.git project_knowledge`。アクセス権はユーザーが管理する）。
- 配置されるまでは、内部文書に依存しない作業だけを行う。研究判断、校正・評価設定の変更、受入基準に関わる変更は行わない。
- 新しい内部文書が必要になったら、作業用の一時領域（scratchpad など）に下書きしてユーザーに渡し、配置後に `project_knowledge/` へ移す。

## このリポジトリの文書

- 文書として残すのは `README.md`・`CLAUDE.md`・`AGENTS.md` と、README の図 `docs/assets/` だけである。
- `README.md` は利用者向けの入口で、作業の判断材料としては使わない。詳細な研究情報は書かない。掲載図の出典説明は README の「掲載画像について」に置く。
- **`CLAUDE.md` と `AGENTS.md` は同一内容を保つ。** 片方を変えたらもう片方も変える。

## コマンド（Windows 11 / PowerShell）

- Python 実行（uv）: `uv run python <script>.py` ／ パッケージ追加: `uv add <pkg>` ／ 依存同期: `uv sync`
- 校正 → 推論（リポジトリ root を cwd に）: `uv run python -m calibration.calibrate` → `uv run python -m calibration.fit_params` → `uv run python -m checks.evaluate`
- 検証ハーネス（`checks/`）: `uv run python -m checks.evaluate` ／ `uv run python -m checks.verify_pipeline_equivalence` ／ `uv run python -m checks.verify_clip_replay --session <name>`
- 自動テスト（`tests/`・HW 非依存の pytest）: `uv run python -m pytest`。dev 依存の追加は `uv add --dev <pkg>`
- ベンチ: `uv run python -m checks.bench_disparity --impl native`
- Rust↔Python 照合: `uv run python -m checks.verify_disparity_rust`
- Rust（cargo。`rust/disparity/` で実行。`cargo build`/`clippy` は `$env:PYO3_PYTHON` に venv の python.exe を指定するか有効な venv 下で実行する）: `cargo build` ／ `cargo test` ／ `cargo clippy`
- Rust → Python 導入（**`maturin develop` は uv の venv では不可**＝pip 非搭載）: `rust/disparity/` で `maturin build --release` → `uv pip install (Get-ChildItem target\wheels\*.whl).FullName`。**`uv sync` を実行すると手動導入した Rust 拡張（`_rt_disparity_native`）は削除される**ため、`uv sync` 後は上記を再実行する
- `tools/record_repro/` は、`project_knowledge/records/` の実験記録を読み書きする内部の再現・作図スクリプトである。通常の実行と自動テストはこれに依存しない
- ※シェルは PowerShell。パス区切りは `\`、ワイルドカード展開は外部コマンドへ自動では効かないため `Get-ChildItem` 等を使う。

## 規約

- `core/` は純関数（config import・ファイル I/O・表示・print 禁止。依存はすべて引数注入）。唯一の例外は `core/params.py`（CSV I/O 可）。
- `core/_disparity_reference.py` は凍結された参照実装で、変更しない。
- 全定数・パスは `config.py` に一元管理し、呼び出し側が core へ注入する。
- 無効値は `np.nan`（0.0 埋めは禁止。「視差ちょうど 0＝合焦面上」と「無効」を区別するため）。
- 入力検証は `raise ValueError`（`assert` は `-O` で無効化されるため禁止）。
- 命名: チャンネルは `red`/`blue`（`std`/`tmp` 禁止）、ZNCC 類似度は `sim_neg`/`sim_peak`/`sim_pos`、視差は `disp_map`（深度は `depth_map`）、参照式係数は `pos`/`neg` 接尾辞（前後ぼけの物理ラベルは使わない）。
- `params/`・`params/baseline/` は実行・回帰用の入力であり、文書として扱わない。
- 実験は git ブランチで分離し、最小限・非破壊の変更を優先する。
