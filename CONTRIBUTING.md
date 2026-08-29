# コントリビューションについて

Issue・Pull Request を歓迎します。日本語でも英語でも構いません。

## どこに出すか

このリポジトリは**仕様の実装**です。指摘の内容によって、出し先が変わります。

| 内容 | 出し先 |
|------|--------|
| このツールの不具合・機能追加 | このリポジトリの [Issues](https://github.com/qchizu-project/datapng-tiler/issues) |
| 仕様そのものへの疑問・提案 | [tilejson-datapng-extension](https://github.com/qchizu-project/tilejson-datapng-extension/issues) |

「仕様ではこう読めるのに実装がそうなっていない」という指摘は、どちらでも構いません（こちらで振り分けます）。仕様の解釈が曖昧だった、という結論になることもあります。

## 開発

```sh
uv sync
uv run pytest -v
uv run ruff check . && uv run ruff format .
```

CI は Ubuntu / macOS / Windows × Python 3.12 / 3.13 で回ります。プロセス並列の既定が
Linux は fork、macOS と Windows は spawn と異なるため、**ワーカへ渡す状態はすべて
`initargs` で明示的に渡してください**（グローバル変数は spawn では引き継がれません）。

## テストの方針

- **仕様適合は独立デコーダで確かめます。** `tests/test_codec.py` には仕様書の変換式を
  文字どおり写したデコーダがあります。実装のデコーダで往復させても「実装が自分の仕様に
  従っている」ことしか言えないためです。新しいエンコードを足すときも、まず仕様の式を
  ここへ写してください。
- **タイルの値は解析的に検証します。** 値が座標の 1 次関数になる合成ラスタを使い、
  タイル画素が代表する座標での真値と突き合わせます（`tests/helpers.py`）。
  「絵として正しそう」では幾何の半画素ずれを検出できません。
- **外部データ・ネットワークに依存しません。** 必要なラスタはその場で合成します。
- 決定性（同じ入力から同じバイト列）と、並列・逐次の一致もテストで固定しています。

## 設計上、動かしにくいところ

次の 2 つは、片方だけ変えられない形に閉じてあります。壊さないでください。

- **`support` の宣言と生成方式**（`modes/base.py`）。TileJSON に書く `support` と、
  実際の再投影アライメント・オーバービュー方式は必ず一致します。
- **無効値の表し方**（`modes/numerical.py`）。アルファか `invalidColor` のどちらか一方で、
  併用はできません（仕様 §3.2.2 MUST NOT）。

## コミットメッセージ

[Conventional Commits](https://www.conventionalcommits.org/ja/) の形式（`feat:` / `fix:` /
`docs:` / `refactor:` / `test:` / `chore:`）を使います。本文には**なぜそうしたか**を書いて
ください——何をしたかは diff を見れば分かります。

## ライセンス

貢献いただいたコードは MIT License で配布されます。
