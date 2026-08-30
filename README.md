# datapng-tiler

ラスタデータから [データPNG](https://gsj-seamless.jp/labs/datapng/) タイル（**数値PNG**・**パレットPNG**）と **TileJSON** を生成する CLI / Python ライブラリです。

[TileJSON DataPNG Extension](https://github.com/qchizu-project/tilejson-datapng-extension) v0.7.0 に準拠します。

> **Language**: [English README](./README.en.md)

## これは何か

標高・水深・気温のような**連続値**や、土地利用・浸水深階級のような**区分**を、地図タイルとして配信するときに使います。値を RGB に可逆に埋め込んだタイルと、それを復号するために必要なメタデータ（係数・単位・無効値・凡例など）を記述した TileJSON を、まとめて生成します。

| | 数値PNG（numerical） | パレットPNG（palette） |
|---|---|---|
| 入力 | 連続値のラスタ | クラス値ラスタ、または RGB ラスタ + 凡例定義 |
| 格納 | 値を 24 ビット符号付き整数に量子化して RGB へ | 凡例の色をそのまま |
| 用途 | 標高・水深・気温・濃度 | 土地利用・災害リスク区分 |

タイル画像は **WebP（可逆圧縮）が既定**で、PNG も選べます。どちらも可逆なので値は劣化しません。

## インストール

```sh
uvx datapng-tiler --help          # 実行するだけなら（インストール不要）
pipx install datapng-tiler        # コマンドとして常設する
pip install datapng-tiler         # ライブラリとしても使う
```

Python 3.12 以上が必要です。GDAL は rasterio の wheel に同梱されているので、別途インストールする必要はありません。

## 使い方

### 数値PNGタイル（標高など）

```sh
datapng-tiler tile dem.tif -o tiles/ --factor 0.01 --unit m \
    --description "標高は東京湾平均海面（T.P.）基準。"
```

`tiles/` に以下が生成されます。

```
tiles/
├── {z}/{x}/{y}.webp   タイル
├── tiles.json         TileJSON（datapng 拡張つき）
└── index.html         プレビュー（ブラウザで開くと値を読める）
```

`--factor 0.01` は「0.01 単位で量子化する」という意味です。標高なら 1cm 刻み。値が 24 ビット整数（±8,388,607）に収まらない場合は**エラーで止まります**——黙って折り返した誤った値を出力しないためです。エラーメッセージが適切な `--factor` を提示します。

### パレットPNGタイル（区分など）

凡例定義（YAML または JSON）を用意します。

```yaml
# legend.yaml
title: 洪水浸水想定区域（想定最大規模）浸水深
items:
  - value: 1          # クラス値ラスタを入力にするときの対応値
    r: 245
    g: 245
    b: 50
    title: 0.5m未満
    description: 床下浸水相当。避難行動は徒歩で可能。
  - value: 2
    r: 255
    g: 216
    b: 0
    title: 0.5〜3.0m
```

```sh
datapng-tiler tile flood.tif -o tiles/ --type palette --legend legend.yaml
```

RGB ラスタ（すでに色が塗られたデータ）も入力にできます。その場合 `value` は不要です。**凡例に無い色が見つかるとエラーで止まります**（`--on-unknown-color nodata` で無効値として扱えます）。

### 既存タイルの移行

Mapbox Terrain-RGB や Mapzen/Terrarium で配信されている既存のタイル資産を、正式なデータPNG エンコードへ移せます。タイルはすでに目的の格子に載っているので再投影せず、値を読み替えるだけです。

```sh
datapng-tiler convert ./terrain-rgb/ -o tiles/ --from mapbox --factor 0.01 --unit m
```

逆に、既存のラスタから Terrain-RGB 互換のタイルを作ることもできます。

```sh
datapng-tiler tile dem.tif -o tiles/ --encoding mapbox
```

### 検証

生成物が仕様に適合しているかを確かめます。CI に組み込めます。

```sh
datapng-tiler validate tiles/tiles.json --tiles tiles/
```

2 段階で見ます。

1. `datapng` を仕様の JSON Schema にかける
2. **宣言と実タイルを突き合わせる** — ズーム範囲・形式・無効値の表し方・凡例の色

とくに「アルファチャンネルを持つタイルに `invalidColor` を宣言してはならない」（仕様 §3.2.2 MUST NOT）は TileJSON だけを見ても分からず、実タイルを開いて初めて検出できます。

### 1 枚を確認する

```sh
datapng-tiler inspect tiles/14/14552/6451.webp --tilejson tiles/tiles.json --pixel 100 200
```

## 正確性について

このツールが特に気をつけていることを挙げます。

**無効値の判定に閾値を使いません。** 「nodata に近い値を無効とみなす」実装は、nodata が 0 や正の値であるデータで有効値を消してしまいます。GDAL のマスクを正とし、nodata の宣言が無いソースではワープの被覆マスク（`add_alpha`）でソース範囲外を無効にします。

**24 ビットに収まらない値を黙って通しません。** 既定でエラーにし、`--on-overflow clamp|nodata` で明示的に選べます。

**`support` の宣言と生成方式が必ず一致します。** `--support point`（既定）なら左上法で再投影し、オーバービューも左上の値を運びます。`--support block` なら中心整列で、オーバービューは整数領域での平均になります。TileJSON にはそのとおりの `support` が出ます。片方だけ変えられる API にはしていません。

**再投影は厳密寄りのトランスフォーマで行います。** GDAL 既定の近似トランスフォーマは近似誤差が出力解像度（＝生成ズーム）によって変わるため、同じ地点を狙っても標本位置がサブピクセルでにじみ、「ズームによって値が違う」現象を生みます。

**オーバービューは raw 整数のまま合成します。** 値は raw 整数のアフィン関数なので、整数領域の平均と値領域の平均は一致し、ズームを重ねても量子化誤差が積み上がりません。

**出力は決定的です。** 同じ入力・同じ設定なら、`--jobs` を変えても生成されるタイルはバイト単位で一致します。

## 無効値の表し方

無効値はアルファ 0 か、指定した色のどちらか一方で表します。

| | 無効値 | `invalidColor` |
|---|---|---|
| 既定 | アルファ 0 | 宣言しない |
| `--no-alpha --invalid-color R G B` | 指定した色 | 宣言する |

仕様 §3.2.2 の `invalidColor` は**完全に透明な画素を指せません**（WebP の可逆圧縮が透明画素の RGB を保存しないため）。既定の出力では無効画素が完全に透明になるので、色を宣言しても判定に使われません。CLI は `--invalid-color` の単独指定を拒否します。

なお、無効画素が 1 つも無いタイルはアルファチャンネルを持たない形で書かれます（容量が減ります）。

## 主なオプション

```
--format webp|png            タイル画像形式（既定: webp）
--tile-size N                タイル一辺の画素数（既定: 512）
--support point|block        画素値が代表する領域（既定: point = 左上節点）
--resampling nearest|bilinear|cubic|lanczos    再投影カーネル（既定: bilinear）
--factor F --offset O        v = F × rawValue + O
--encoding mapbox|terrarium  互換エンコードで出力する
--on-overflow error|clamp|nodata               範囲外の値の扱い（既定: error）
--data-range MIN MAX         TileJSON に載せる期待範囲
--auto-data-range            期待範囲を生成タイルから実測する
-z / --min-zoom              ズーム範囲（既定: ソース解像度から自動）
--bounds W S E N             生成範囲（既定: ソース範囲）
-j / --jobs N                並列プロセス数（既定: CPU 数）
--overwrite                  既存タイルも作り直す（入力を更新したときに必要）
--basemap none|gsi|osm       プレビューの背景地図（既定: none）
```

`datapng-tiler <サブコマンド> --help` で全オプションを確認できます。

中断した実行は、同じコマンドをもう一度走らせれば続きから再開します（既存タイルはスキップされます）。**入力データを更新したときは `--overwrite` が必要です**——既定では既存タイルを作り直さないため、1 枚も更新されません。

## プレビューについて

`index.html` はタイル木のルートに置かれ、ブラウザで開くとカーソル位置の値を表示します。画像として並べるだけでなく **TileJSON の宣言どおりに復号して見せる**ので、「絵としては出ているが値が違う」を見つけられます。

- 値の読み取りにはタイルが HTML と同一オリジンにある必要があります（`python -m http.server -d tiles/` などで開いてください）。別オリジンのタイルは表示はできますが値を読めません。
- **背景地図は既定で無しです。** 地理院タイルや OpenStreetMap を既定にすると、このツールを使うすべての人に第三者サービスの利用規約を負わせることになるためです。`--basemap gsi|osm` で明示的に選べます（選ぶと規約の所在を表示します）。

## ライブラリとして使う

```python
from datapng_tiler.codec import NumericalEncoding
from datapng_tiler.engine import tile_raster
from datapng_tiler.modes import NumericalMode
from datapng_tiler.tilejson import from_tree, write_tilejson

mode = NumericalMode(encoding=NumericalEncoding(factor=0.01), unit="m")
result = tile_raster("dem.tif", "tiles/", mode, processes=8)
write_tilejson(from_tree("tiles/", mode, name="標高"), "tiles/tiles.json")
```

`datapng_tiler.codec` は純粋関数だけなので、符号化・復号だけを使うこともできます。

## 開発

```sh
uv sync
uv run pytest -v
uv run ruff check . && uv run ruff format .
```

テストは外部データやネットワークに依存しません。仕様適合の検証には、**仕様書の変換式を文字どおり写した独立デコーダ**を使っています（実装同士の往復では「実装が自分の仕様に従っている」ことしか言えないため）。

性能の実測は [BENCHMARKS.md](./BENCHMARKS.md) を参照してください。

## リリース

1. `src/datapng_tiler/__init__.py` の `__version__` と `CHANGELOG.md` を更新して main にマージ
2. `git tag vX.Y.Z && git push origin vX.Y.Z`

タグの push で GitHub Actions が PyPI（Trusted Publishing）と GitHub Release へ公開します。初回のみ PyPI 側で pending publisher の登録が必要です。

## ライセンス

MIT License（[LICENSE](./LICENSE)）。準拠仕様と依存ライブラリの帰属は [NOTICE](./NOTICE) を参照してください。

準拠する仕様 [TileJSON DataPNG Extension](https://github.com/qchizu-project/tilejson-datapng-extension) は CC0 1.0 で公開されています。
