"""コマンドラインインターフェース。

サブコマンドは 5 つ:

- ``tile``     : ラスタ → タイル木 + TileJSON（+ プレビュー HTML）
- ``convert``  : 既存タイル木の再エンコード（Terrain-RGB → データPNG など）
- ``tilejson`` : 既存タイル木から TileJSON だけを作り直す
- ``validate`` : TileJSON の仕様適合と、実タイルとの突合
- ``inspect``  : タイル 1 枚を復号して統計や特定画素の値を見る

引数の組み立て（`--support` と縮小方式、`--invalid-color` とアルファ）は、宣言と
実タイルが食い違わないよう `modes` 側で束ねてある。CLI はその入口に徹する。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from datapng_tiler import SPEC_VERSION, __version__
from datapng_tiler.codec import (
    DEFAULT_INVALID_COLOR,
    ON_OVERFLOW_CHOICES,
    SPECIAL_ENCODINGS,
    NumericalEncoding,
    ValueRangeError,
)
from datapng_tiler.convert import SourceTiles, convert_tree
from datapng_tiler.engine import scan_tree, tile_raster
from datapng_tiler.fileio import sweep_temp_files
from datapng_tiler.geo import GeometryError
from datapng_tiler.imageio import (
    DEFAULT_PNG_COMPRESS_LEVEL,
    DEFAULT_WEBP_METHOD,
    FORMATS,
    TileFormat,
    load_tile,
)
from datapng_tiler.legend import Legend, LegendError
from datapng_tiler.modes import NumericalMode, PaletteMode
from datapng_tiler.modes.base import SUPPORT_CHOICES
from datapng_tiler.modes.numerical import (
    DEFAULT_RESAMPLING,
    RESAMPLING_CHOICES,
    scan_value_range,
)
from datapng_tiler.tilejson import default_tiles_url, from_tree, read_tilejson, write_tilejson
from datapng_tiler.validate import validate as run_validate
from datapng_tiler.viewer import (
    BASEMAP_CHOICES,
    basemap_notice,
    build_viewer_html,
    write_viewer_html,
)

logger = logging.getLogger("datapng_tiler")

DEFAULT_TILE_SIZE = 512
DEFAULT_TILEJSON_NAME = "tiles.json"
DEFAULT_VIEWER_NAME = "index.html"
DEFAULT_LEGEND_NAME = "legend.json"


# --- 共通の引数 ---------------------------------------------------------------------


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format", choices=FORMATS, default="webp", help="タイル画像形式（既定: webp）"
    )
    parser.add_argument(
        "--webp-method",
        type=int,
        default=DEFAULT_WEBP_METHOD,
        help=f"WebP の圧縮努力 0〜6（既定: {DEFAULT_WEBP_METHOD}）",
    )
    parser.add_argument(
        "--png-compress-level",
        type=int,
        default=DEFAULT_PNG_COMPRESS_LEVEL,
        help=f"PNG の zlib 圧縮レベル 0〜9（既定: {DEFAULT_PNG_COMPRESS_LEVEL}）",
    )
    parser.add_argument(
        "--tile-size", type=int, default=DEFAULT_TILE_SIZE, help="タイル一辺の画素数（既定: 512）"
    )


def _add_metadata_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", help="タイルセット名")
    parser.add_argument(
        "--description",
        help="自由記述。標高の鉛直基準面（測地系）等はここに書く（仕様 §1.1）",
    )
    parser.add_argument("--attribution", help="帰属表示（HTML 可）")
    parser.add_argument("--tileset-version", help="タイルセットのバージョン（semver）")
    parser.add_argument(
        "--tiles-url",
        help="タイル URL テンプレート（既定: ./{z}/{x}/{y}.<拡張子> の相対参照）",
    )


def _add_numerical_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--factor", type=float, default=1.0, help="係数 f（既定: 1）")
    parser.add_argument("--offset", type=float, default=0.0, help="オフセット o（既定: 0）")
    parser.add_argument(
        "--encoding",
        choices=SPECIAL_ENCODINGS,
        help="互換エンコード。指定すると factor / offset は無視される（仕様 §3.2.1）",
    )
    parser.add_argument("--unit", help="変換後の値の単位（例: m）")
    parser.add_argument(
        "--on-overflow",
        choices=ON_OVERFLOW_CHOICES,
        default="error",
        help="24 ビットに収まらない値の扱い（既定: error）",
    )
    parser.add_argument(
        "--no-alpha",
        action="store_true",
        help="アルファを使わず無効色で無効値を表す（--invalid-color と併用）",
    )
    parser.add_argument(
        "--invalid-color",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        help=f"--no-alpha 時の無効色（既定: {' '.join(map(str, DEFAULT_INVALID_COLOR))}）",
    )


def _build_encoding(args: argparse.Namespace) -> NumericalEncoding:
    if args.encoding and (args.factor != 1.0 or args.offset != 0.0):
        logger.warning(
            "--encoding %s では factor / offset は無視されます（仕様 §3.2.1 MUST）", args.encoding
        )
    return NumericalEncoding(
        factor=args.factor,
        offset=args.offset,
        special=args.encoding,
        on_overflow=args.on_overflow,
    )


def _invalid_color(args: argparse.Namespace) -> tuple[int, int, int] | None:
    """無効値の表し方を決める（アルファ or 無効色）。

    本ツールの出力では、無効値はどちらか一方で表す。既定（アルファ）の出力では無効画素が
    完全に透明になり、仕様 §3.2.2 により `invalidColor` の判定対象から外れるため、色を
    指定しても意味を持たない。`--invalid-color` だけを渡された場合は `--no-alpha` を
    忘れているとみなしてエラーにする（黙って無視すると、指定したつもりの出力と実物が
    食い違う）。
    """
    if args.invalid_color and not args.no_alpha:
        raise SystemExit(
            "エラー: --invalid-color は --no-alpha と併せて指定してください"
            "（既定の出力では無効画素が完全に透明になり、invalidColor の判定対象から"
            "外れるため意味を持ちません。仕様 §3.2.2）"
        )
    if not args.no_alpha:
        return None
    if args.invalid_color:
        return tuple(args.invalid_color)  # type: ignore[return-value]
    return DEFAULT_INVALID_COLOR


def _tile_format(args: argparse.Namespace) -> TileFormat:
    return TileFormat(
        name=args.format,
        webp_method=args.webp_method,
        png_compress_level=args.png_compress_level,
    )


def _default_jobs() -> int:
    return os.cpu_count() or 1


# --- tile ---------------------------------------------------------------------------


def _build_mode(args: argparse.Namespace):
    fmt = _tile_format(args)
    if args.type == "palette":
        if args.no_alpha or args.invalid_color:
            # パレットPNGの無効値は透明のみ（仕様 §7: invalidColor は数値PNG専用）。
            # 黙って無視すると、指定したつもりの出力と実物が食い違う。
            raise SystemExit(
                "エラー: --no-alpha / --invalid-color は数値PNG専用です"
                "（パレットPNGの無効値は透明で表します。仕様 §7）"
            )
        legend = Legend.load(args.legend)
        return PaletteMode(
            tile_size=args.tile_size,
            support=args.support,
            fmt=fmt,
            legend=legend,
            band=args.band,
            src_nodata=args.src_nodata,
            on_unknown_color=args.on_unknown_color,
            legend_url=args.legend_url,
        )
    return NumericalMode(
        tile_size=args.tile_size,
        support=args.support,
        fmt=fmt,
        encoding=_build_encoding(args),
        band=args.band,
        src_nodata=args.src_nodata,
        resampling=args.resampling,
        invalid_color=_invalid_color(args),
        unit=args.unit,
        data_range=tuple(args.data_range) if args.data_range else None,
        precision=args.precision,
    )


def cmd_tile(args: argparse.Namespace) -> int:
    mode = _build_mode(args)
    output = Path(args.output)
    if output.exists():
        # 前回の実行が異常終了して残った一時ファイルを掃除してから始める
        sweep_temp_files(output)

    result = tile_raster(
        args.input,
        output,
        mode,
        max_zoom=args.max_zoom,
        min_zoom=args.min_zoom,
        bounds=tuple(args.bounds) if args.bounds else None,
        processes=args.jobs,
        overwrite=args.overwrite,
    )
    print(
        f"タイル: ベース {result.base_tiles} 枚 / オーバービュー {result.overview_tiles} 枚"
        f"（z{result.min_zoom}-{result.max_zoom}）"
    )

    _write_sidecars(args, mode, output)
    return 0


def _write_sidecars(args: argparse.Namespace, mode, output: Path) -> None:
    """TileJSON・凡例・プレビュー HTML を書く。"""
    if args.no_tilejson:
        return

    summary = scan_tree(output, mode.fmt.extension)
    if summary is None:
        logger.warning("タイルが 1 枚も無いため TileJSON は書きません")
        return

    if getattr(args, "auto_data_range", False) and isinstance(mode, NumericalMode):
        measured = scan_value_range(output, mode, summary.max_zoom)
        if measured is None:
            logger.warning("有効画素が無いため dataRange を実測できませんでした")
        else:
            mode = replace(mode, data_range=measured)
            print(f"dataRange 実測: {measured[0]:.6g} 〜 {measured[1]:.6g}")

    if isinstance(mode, PaletteMode) and mode.legend_url:
        legend_path = output / DEFAULT_LEGEND_NAME
        mode.legend.dump(legend_path)
        print(f"凡例: {legend_path}（{mode.legend_url} から参照される想定）")

    doc = from_tree(
        output,
        mode,
        tiles_url=args.tiles_url,
        name=args.name,
        description=args.description,
        attribution=args.attribution,
        version=args.tileset_version,
        summary=summary,
    )
    path = write_tilejson(doc, args.tilejson or output / DEFAULT_TILEJSON_NAME)
    print(f"TileJSON: {path}")

    if not args.no_viewer:
        viewer_doc = dict(doc)
        # プレビューはタイル木のルートに置くので、常に相対参照で引く
        viewer_doc["tiles"] = [default_tiles_url(mode.fmt.extension)]
        html = build_viewer_html(
            viewer_doc,
            basemap=args.basemap,
            basemap_url=args.basemap_url,
            basemap_attribution=args.basemap_attribution,
        )
        viewer_path = write_viewer_html(html, output / DEFAULT_VIEWER_NAME)
        print(f"プレビュー: {viewer_path}")
        notice = basemap_notice(args.basemap)
        if notice:
            print(notice, file=sys.stderr)


# --- convert ------------------------------------------------------------------------


def cmd_convert(args: argparse.Namespace) -> int:
    source = SourceTiles(
        encoding=NumericalEncoding(
            factor=args.from_factor,
            offset=args.from_offset,
            special=None if args.source_encoding == "datapng" else args.source_encoding,
        ),
        invalid_color=tuple(args.from_invalid_color) if args.from_invalid_color else None,
    )
    mode = NumericalMode(
        tile_size=args.tile_size,
        support=args.support,
        fmt=_tile_format(args),
        encoding=_build_encoding(args),
        invalid_color=_invalid_color(args),
        unit=args.unit,
        data_range=tuple(args.data_range) if args.data_range else None,
        precision=args.precision,
    )

    output = Path(args.output)
    result = convert_tree(
        args.input, output, source, mode, processes=args.jobs, overwrite=args.overwrite
    )
    print(f"再エンコード: {result.written} 枚（スキップ {result.skipped}）")

    if args.tilejson_in:
        inherited = read_tilejson(args.tilejson_in)
        args.name = args.name or inherited.get("name")
        args.description = args.description or inherited.get("description")
        args.attribution = args.attribution or inherited.get("attribution")

    _write_sidecars(args, mode, output)
    return 0


# --- tilejson -----------------------------------------------------------------------


def cmd_tilejson(args: argparse.Namespace) -> int:
    mode = _build_mode(args)
    output = Path(args.tiles_dir)
    summary = scan_tree(output, mode.fmt.extension)
    if summary is None:
        print(f"エラー: タイルが 1 枚もありません: {output}", file=sys.stderr)
        return 1

    if isinstance(mode, PaletteMode) and mode.legend_url:
        legend_path = mode.legend.dump(output / DEFAULT_LEGEND_NAME)
        print(f"凡例: {legend_path}（{mode.legend_url} から参照される想定）")

    doc = from_tree(
        output,
        mode,
        tiles_url=args.tiles_url,
        name=args.name,
        description=args.description,
        attribution=args.attribution,
        version=args.tileset_version,
        summary=summary,
    )
    path = write_tilejson(doc, args.output or output / DEFAULT_TILEJSON_NAME)
    print(f"TileJSON: {path}（{summary.tile_count} 枚, z{summary.min_zoom}-{summary.max_zoom}）")
    return 0


# --- validate -----------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    problems = run_validate(args.tilejson, args.tiles, sample=args.sample)
    if not problems:
        target = f"{args.tilejson}" + (f" + {args.tiles}" if args.tiles else "")
        print(f"仕様 {SPEC_VERSION} に適合しています: {target}")
        return 0
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"\n{len(problems)} 件の問題が見つかりました", file=sys.stderr)
    return 1


# --- inspect ------------------------------------------------------------------------


def _checked_pixel(pixel: list[int], shape: tuple[int, ...]) -> tuple[int, int]:
    """`--pixel COL ROW` を検証する。

    そのまま添字にすると、範囲外はトレースバック、負値は反対側の画素の値を黙って
    返してしまう（どちらも「指定した場所の値」ではない）。
    """
    col, row = pixel
    height, width = shape[0], shape[1]
    if not (0 <= col < width and 0 <= row < height):
        raise ValueError(
            f"--pixel {col} {row} はタイルの範囲外です（横 0〜{width - 1}, 縦 0〜{height - 1}）"
        )
    return col, row


def cmd_inspect(args: argparse.Namespace) -> int:
    doc: dict[str, Any] = read_tilejson(args.tilejson) if args.tilejson else {}
    datapng = doc.get("datapng", {})
    rgb, alpha = load_tile(args.tile)

    invalid_color = datapng.get("invalidColor")
    if alpha is not None:
        valid = alpha > 0
    elif invalid_color:
        valid = ~np.all(rgb == np.array(invalid_color, dtype=np.uint8), axis=-1)
    else:
        valid = np.ones(rgb.shape[:2], dtype=bool)

    print(f"タイル: {args.tile}")
    print(f"  サイズ: {rgb.shape[1]}×{rgb.shape[0]}")
    print(f"  アルファチャンネル: {'あり' if alpha is not None else 'なし'}")
    print(f"  有効画素: {int(valid.sum())} / {valid.size}")

    if datapng.get("type") == "palette":
        return _inspect_palette(rgb, valid, datapng, args)

    encoding = NumericalEncoding(
        factor=datapng.get("factor", 1.0),
        offset=datapng.get("offset", 0.0),
        special=datapng.get("specialEncoding") or None,
    )
    values = encoding.decode(rgb)
    if valid.any():
        good = values[valid]
        unit = f" {datapng['unit']}" if datapng.get("unit") else ""
        print(f"  最小: {good.min():.6g}{unit}")
        print(f"  最大: {good.max():.6g}{unit}")
        print(f"  平均: {good.mean():.6g}{unit}")

    if args.pixel:
        col, row = _checked_pixel(args.pixel, rgb.shape)
        if valid[row, col]:
            print(f"  画素 ({col}, {row}): {values[row, col]:.6g}  RGB={tuple(rgb[row, col])}")
        else:
            print(f"  画素 ({col}, {row}): 無効値  RGB={tuple(rgb[row, col])}")
    return 0


def _inspect_palette(
    rgb: np.ndarray, valid: np.ndarray, datapng: dict[str, Any], args: argparse.Namespace
) -> int:
    legend = datapng.get("legend")
    items = legend.get("items", []) if isinstance(legend, dict) else []
    lookup = {(i["r"], i["g"], i["b"]): i["title"] for i in items}

    visible = rgb[valid]
    if visible.size:
        colors, counts = np.unique(visible.reshape(-1, 3), axis=0, return_counts=True)
        print("  色の内訳:")
        for color, count in sorted(zip(colors, counts, strict=True), key=lambda p: -p[1]):
            key = tuple(int(c) for c in color)
            print(f"    {key}: {count} 画素  {lookup.get(key, '（凡例に無い色）')}")

    if args.pixel:
        col, row = _checked_pixel(args.pixel, rgb.shape)
        key = tuple(int(c) for c in rgb[row, col])
        label = "無効値" if not valid[row, col] else lookup.get(key, "（凡例に無い色）")
        print(f"  画素 ({col}, {row}): {label}  RGB={key}")
    return 0


# --- パーサ -------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datapng-tiler",
        description=(
            "データPNG（数値PNG・パレットPNG）タイルと TileJSON を生成する。"
            f" 準拠仕様: TileJSON DataPNG Extension {SPEC_VERSION}"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細ログを出す")
    sub = parser.add_subparsers(dest="command", required=True)

    # tile
    tile = sub.add_parser("tile", help="ラスタからタイルと TileJSON を生成する")
    tile.add_argument("input", help="入力ラスタ（GeoTIFF / VRT など GDAL が読める形式）")
    tile.add_argument("-o", "--output", required=True, help="出力ディレクトリ")
    tile.add_argument(
        "--type", choices=("numerical", "palette"), default="numerical", help="タイル種別"
    )
    tile.add_argument("--band", type=int, default=1, help="対象バンド（1 始まり）")
    tile.add_argument("--src-nodata", type=float, help="ソースの無効値（宣言が無い場合に指定）")
    tile.add_argument("-z", "--max-zoom", type=int, help="最大ズーム（既定: 解像度から自動）")
    tile.add_argument("--min-zoom", type=int, help="最小ズーム（既定: 全体が収まるズーム）")
    tile.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("W", "S", "E", "N"),
        help="生成範囲 WGS84（既定: ソース範囲）",
    )
    tile.add_argument(
        "--support",
        choices=SUPPORT_CHOICES,
        default="point",
        help="画素値が代表する領域（既定: point = 左上節点）。block は中心整列 + 平均縮小",
    )
    tile.add_argument(
        "--resampling",
        choices=tuple(RESAMPLING_CHOICES),
        default=DEFAULT_RESAMPLING,
        help=f"再投影カーネル（数値PNGのみ。既定: {DEFAULT_RESAMPLING}）",
    )
    tile.add_argument("--legend", help="パレットPNGの凡例定義（YAML / JSON）")
    tile.add_argument("--legend-url", help="凡例を外部参照にする URL（仕様 §3.3.2）")
    tile.add_argument(
        "--on-unknown-color",
        choices=("error", "nodata"),
        default="error",
        help="凡例に無い色・クラス値の扱い（既定: error）",
    )
    tile.add_argument(
        "--data-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="デコード後の期待範囲（TileJSON に載せる）",
    )
    tile.add_argument(
        "--auto-data-range",
        action="store_true",
        help="dataRange を生成タイルから実測する（--data-range より優先。数値PNGのみ）",
    )
    tile.add_argument("--precision", type=float, help="元データの有効な最小単位")
    _add_numerical_options(tile)
    _add_output_options(tile)
    _add_metadata_options(tile)
    _add_run_options(tile)
    _add_sidecar_options(tile)
    tile.set_defaults(func=cmd_tile)

    # convert
    convert = sub.add_parser("convert", help="既存タイル木を再エンコードする")
    convert.add_argument("input", help="入力タイル木のルート（{z}/{x}/{y}.* を含む）")
    convert.add_argument("-o", "--output", required=True, help="出力ディレクトリ")
    convert.add_argument(
        "--from",
        dest="source_encoding",
        choices=("datapng", *SPECIAL_ENCODINGS),
        default="mapbox",
        help="入力タイルの符号化（既定: mapbox）",
    )
    convert.add_argument(
        "--from-factor", type=float, default=1.0, help="入力が datapng の場合の係数"
    )
    convert.add_argument(
        "--from-offset", type=float, default=0.0, help="入力が datapng の場合のオフセット"
    )
    convert.add_argument(
        "--from-invalid-color",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        help="アルファを持たない入力タイルでの無効色",
    )
    convert.add_argument("--tilejson-in", help="入力側の TileJSON（名前・帰属表示を引き継ぐ）")
    convert.add_argument(
        "--support", choices=SUPPORT_CHOICES, default="point", help="出力に宣言する support"
    )
    convert.add_argument(
        "--data-range", nargs=2, type=float, metavar=("MIN", "MAX"), help="デコード後の期待範囲"
    )
    convert.add_argument("--precision", type=float, help="元データの有効な最小単位")
    _add_numerical_options(convert)
    _add_output_options(convert)
    _add_metadata_options(convert)
    _add_run_options(convert)
    _add_sidecar_options(convert)
    convert.set_defaults(
        func=cmd_convert, type="numerical", band=1, src_nodata=None, auto_data_range=False
    )

    # tilejson
    tilejson = sub.add_parser("tilejson", help="既存タイル木から TileJSON を作る")
    tilejson.add_argument("tiles_dir", help="タイル木のルート")
    tilejson.add_argument(
        "-o", "--output", help=f"出力先（既定: <tiles_dir>/{DEFAULT_TILEJSON_NAME}）"
    )
    tilejson.add_argument(
        "--type", choices=("numerical", "palette"), default="numerical", help="タイル種別"
    )
    tilejson.add_argument("--legend", help="パレットPNGの凡例定義（YAML / JSON）")
    tilejson.add_argument("--legend-url", help="凡例を外部参照にする URL")
    tilejson.add_argument(
        "--support", choices=SUPPORT_CHOICES, default="point", help="画素値が代表する領域"
    )
    tilejson.add_argument(
        "--data-range", nargs=2, type=float, metavar=("MIN", "MAX"), help="デコード後の期待範囲"
    )
    tilejson.add_argument("--precision", type=float, help="元データの有効な最小単位")
    _add_numerical_options(tilejson)
    _add_output_options(tilejson)
    _add_metadata_options(tilejson)
    tilejson.set_defaults(
        func=cmd_tilejson,
        band=1,
        src_nodata=None,
        resampling=DEFAULT_RESAMPLING,
        on_unknown_color="error",
    )

    # validate
    validate = sub.add_parser("validate", help="TileJSON の仕様適合と実タイルとの突合")
    validate.add_argument("tilejson", help="検証する TileJSON")
    validate.add_argument("--tiles", help="タイル木のルート（指定すると実タイルとも突き合わせる）")
    validate.add_argument(
        "--sample", type=int, default=50, help="中身まで開くタイル数（0 で全件。既定: 50）"
    )
    validate.set_defaults(func=cmd_validate)

    # inspect
    inspect = sub.add_parser("inspect", help="タイル 1 枚を復号して確認する")
    inspect.add_argument("tile", help="タイル画像")
    inspect.add_argument("--tilejson", help="復号に使う TileJSON（省略時は素の 24 ビット整数）")
    inspect.add_argument(
        "--pixel", nargs=2, type=int, metavar=("COL", "ROW"), help="値を表示する画素"
    )
    inspect.set_defaults(func=cmd_inspect)

    return parser


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-j", "--jobs", type=int, default=_default_jobs(), help="並列プロセス数（既定: CPU 数）"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存タイルも作り直す（入力を更新したときに必要）",
    )


def _add_sidecar_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tilejson", help=f"TileJSON の出力先（既定: <出力>/{DEFAULT_TILEJSON_NAME}）"
    )
    parser.add_argument("--no-tilejson", action="store_true", help="TileJSON を書かない")
    parser.add_argument("--no-viewer", action="store_true", help="プレビュー HTML を書かない")
    parser.add_argument(
        "--basemap",
        choices=BASEMAP_CHOICES,
        default="none",
        help="プレビューの背景地図（既定: none）。選ぶと提供元の利用規約に従う必要がある",
    )
    parser.add_argument("--basemap-url", help="独自の背景地図 URL テンプレート")
    parser.add_argument("--basemap-attribution", help="独自の背景地図の帰属表示")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if getattr(args, "type", None) == "palette" and not getattr(args, "legend", None):
        parser.error("--type palette には --legend が必要です（凡例はパレットPNGの必須情報）")

    try:
        return args.func(args)
    except (
        ValueRangeError,
        GeometryError,
        LegendError,
        ValueError,
        FileNotFoundError,
        OSError,
    ) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
