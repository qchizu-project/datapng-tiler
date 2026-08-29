# 変更履歴

このファイルの書式は [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) に従い、
バージョニングは [Semantic Versioning](https://semver.org/lang/ja/) に従います。

## [Unreleased]

## [0.1.0] - 2026-08-29

初版。[TileJSON DataPNG Extension](https://github.com/qchizu-project/tilejson-datapng-extension)
v0.7.0 に準拠します。

### Added

- `tile`: ラスタから数値型・パレット型のタイル木と TileJSON を生成する
  - 出力形式は WebP（可逆圧縮・既定）と PNG
  - `factor` / `offset` による量子化、`specialEncoding`（mapbox / terrarium）互換出力
  - `support`（point / block）に応じた再投影アライメントとオーバービュー方式
  - パレット型は凡例定義（YAML / JSON）から色を引き、PNG ではインデックスカラーで書く
  - `--auto-data-range` で `dataRange` を生成タイルから実測
  - プロセス並列・レジューム・アトミック書き込み
- `convert`: 既存タイル木の再エンコード（Terrain-RGB / Terrarium → 正式エンコード、形式変換）
- `tilejson`: 既存タイル木から TileJSON を生成する（範囲・ズームは走査して実測）
- `validate`: JSON Schema による仕様適合検証と、宣言と実タイルの突合
- `inspect`: タイル 1 枚を復号して統計・特定画素の値を表示する
- プレビュー HTML（カーソル位置の値を仕様どおりに復号して表示。背景地図は既定で無し）

[Unreleased]: https://github.com/qchizu-project/datapng-tiler/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/qchizu-project/datapng-tiler/releases/tag/v0.1.0
