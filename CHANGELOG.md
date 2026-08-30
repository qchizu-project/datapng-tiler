# 変更履歴

このファイルの書式は [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) に従い、
バージョニングは [Semantic Versioning](https://semver.org/lang/ja/) に従います。

## [Unreleased]

### Changed

- 用語を[データPNG](https://gsj-seamless.jp/labs/datapng/)仕様に合わせ「数値型・パレット型」から
  「数値PNG・パレットPNG」に統一
- README を利用者向けに整理。「開発」「リリース」節と「正確性について」節を削除し、
  開発・貢献手順は CONTRIBUTING.md へ移した

### Fixed

- プレビューの値表示桁数が常に小数3桁固定だった問題を修正。`factor` の量子化ステップに
  応じた桁数（例: `factor=0.01` なら2桁）で表示する

## [0.1.0] - 2026-08-29

初版。[TileJSON DataPNG Extension](https://github.com/qchizu-project/tilejson-datapng-extension)
v0.7.0 に準拠します。

### Added

- `tile`: ラスタから数値型・パレット型のタイル木と TileJSON を生成する
  - 出力形式は WebP（可逆圧縮・既定）と PNG。形式はタイル URL の拡張子が示す
  - `factor` / `offset` による量子化、`specialEncoding`（mapbox / terrarium）互換出力
  - `support`（point / block）に応じた再投影アライメントとオーバービュー方式
  - パレット型は凡例定義（YAML / JSON）から色を引き、PNG ではインデックスカラーで書く
  - `--auto-data-range` で `dataRange` を生成タイルから実測
  - プロセス並列・レジューム・アトミック書き込み
- `convert`: 既存タイル木の再エンコード（Terrain-RGB / Terrarium → 正式エンコード、形式変換）
- `tilejson`: 既存タイル木から TileJSON を生成する（範囲・ズームは走査して実測）
- `validate`: JSON Schema による仕様適合検証と、宣言と実タイルの突合
  （タイル URL の拡張子と中身の食い違いも検出する）
- `inspect`: タイル 1 枚を復号して統計・特定画素の値を表示する
- プレビュー HTML（カーソル位置の値を仕様どおりに復号して表示。背景地図は既定で無し）

[Unreleased]: https://github.com/qchizu-project/datapng-tiler/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/qchizu-project/datapng-tiler/releases/tag/v0.1.0
