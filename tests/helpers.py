"""テスト共通の定数とヘルパ（フィクスチャではないもの）。

タイル格子に整列した合成ラスタを作り、生成結果を復号するところまでを 1 か所に置く。
「値が座標の 1 次関数」「色がクラス値の対応」という素性のわかった入力を使うことで、
出力を**解析的に予測して**突き合わせられる。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from datapng_tiler.codec import NumericalEncoding
from datapng_tiler.fileio import tile_path
from datapng_tiler.geo import ORIGIN_SHIFT, WEB_MERCATOR, tile_resolution
from datapng_tiler.imageio import TileFormat, load_tile
from datapng_tiler.legend import Legend
from datapng_tiler.modes import NumericalMode, PaletteMode

# 検証に使うタイル（日本付近）。小さめのタイルサイズでテストを軽く保つ。
TILE_SIZE = 64
ZOOM = 8
TX, TY = 227, 100
FACTOR = 0.001
MARGIN = 4

PALETTE_TILE_SIZE = 32

LEGEND = Legend.from_dict(
    {
        "title": "浸水深",
        "items": [
            {"value": 1, "r": 245, "g": 245, "b": 50, "title": "0.5m未満"},
            {"value": 2, "r": 255, "g": 216, "b": 0, "title": "0.5〜3.0m"},
            {"value": 3, "r": 255, "g": 40, "b": 0, "title": "5.0〜10.0m"},
        ],
    }
)


def grid_origin(tile_size: int = TILE_SIZE, margin: int = 0):
    """タイル (ZOOM, TX, TY) の格子に整列した原点と画素サイズ。

    ソース画素の**中心**がタイルの節点に一致するよう半画素ずらす。こうすると左上法の
    再投影が恒等になり、補間の影響を受けずに幾何だけを検証できる。
    """
    res = tile_resolution(ZOOM, tile_size)
    x0 = -ORIGIN_SHIFT + TX * tile_size * res - (margin + 0.5) * res
    y0 = ORIGIN_SHIFT - TY * tile_size * res + (margin + 0.5) * res
    return x0, y0, res


def make_numerical_mode(**kwargs) -> NumericalMode:
    defaults = {
        "tile_size": TILE_SIZE,
        "encoding": NumericalEncoding(factor=FACTOR),
        "fmt": TileFormat("webp"),
        "unit": "m",
    }
    return NumericalMode(**{**defaults, **kwargs})


def make_palette_mode(**kwargs) -> PaletteMode:
    defaults = {"tile_size": PALETTE_TILE_SIZE, "legend": LEGEND, "fmt": TileFormat("webp")}
    return PaletteMode(**{**defaults, **kwargs})


def decode_tile(mode: NumericalMode, root, zoom: int, x: int, y: int):
    """数値型タイルを読んで (値, 有効マスク) にする。"""
    rgb, alpha = load_tile(tile_path(root, zoom, x, y, mode.fmt.extension))
    values = mode.encoding.decode(rgb)
    valid = np.ones(values.shape, dtype=bool) if alpha is None else alpha > 0
    return values, valid


def tree_bytes(root: Path, extension: str) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob(f"*{extension}"))}


def all_tiles(root: Path, extension: str):
    for zoom_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        for x_dir in sorted(zoom_dir.iterdir()):
            for tile in sorted(x_dir.glob(f"*{extension}")):
                yield int(zoom_dir.name), int(x_dir.name), int(tile.stem)


# --- パレット型の合成ラスタ ----------------------------------------------------------


def checker_classes(tile_size: int = PALETTE_TILE_SIZE) -> np.ndarray:
    """1/2/3 のクラス値と nodata(0) を混ぜた市松模様。"""
    classes = np.zeros((tile_size, tile_size), dtype=np.int16)
    classes[0::2, 0::2] = 1
    classes[0::2, 1::2] = 2
    classes[1::2, 0::2] = 3
    return classes


def write_class_raster(path: Path, classes: np.ndarray, nodata: float | None = 0) -> Path:
    x0, y0, res = grid_origin(PALETTE_TILE_SIZE)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=classes.shape[0],
        width=classes.shape[1],
        count=1,
        dtype="int16",
        crs=WEB_MERCATOR,
        transform=from_origin(x0, y0, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(classes.astype(np.int16), 1)
    return path


def write_rgb_raster(path: Path, rgb: np.ndarray) -> Path:
    x0, y0, res = grid_origin(PALETTE_TILE_SIZE)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rgb.shape[0],
        width=rgb.shape[1],
        count=3,
        dtype="uint8",
        crs=WEB_MERCATOR,
        transform=from_origin(x0, y0, res, res),
    ) as dst:
        dst.write(rgb.transpose(2, 0, 1))
    return path
