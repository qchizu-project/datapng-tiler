"""既存タイル木の再エンコードのテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from datapng_tiler.codec import NumericalEncoding
from datapng_tiler.convert import SourceTiles, convert_tree
from datapng_tiler.engine import tile_raster
from datapng_tiler.fileio import tile_path
from datapng_tiler.imageio import TileFormat, load_tile
from tests.helpers import TILE_SIZE, TX, TY, ZOOM, decode_tile, make_numerical_mode


@pytest.fixture
def mapbox_tiles(tmp_path, ramp_raster):
    """Mapbox Terrain-RGB 互換で書いたタイル木（変換元）。"""
    out = tmp_path / "mapbox"
    mode = make_numerical_mode(encoding=NumericalEncoding(special="mapbox"))
    tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM - 1)
    return out, mode


def test_mapbox_タイルを正式エンコードへ移せる(tmp_path, mapbox_tiles):
    src, src_mode = mapbox_tiles
    dst = tmp_path / "datapng"
    target = make_numerical_mode(encoding=NumericalEncoding(factor=0.01), unit="m")

    result = convert_tree(src, dst, SourceTiles(encoding=src_mode.encoding), target)
    assert result.written > 0
    assert result.summary.max_zoom == ZOOM
    assert result.summary.min_zoom == ZOOM - 1

    before, _ = decode_tile(src_mode, src, ZOOM, TX, TY)
    after, _ = decode_tile(target, dst, ZOOM, TX, TY)
    # 再投影しないので、差は両者の量子化分解能の和より小さい
    assert np.abs(after - before).max() <= 0.1 / 2 + 0.01 / 2 + 1e-9


def test_terrarium_タイルも移せる(tmp_path, ramp_raster):
    src = tmp_path / "terrarium"
    src_mode = make_numerical_mode(encoding=NumericalEncoding(special="terrarium"))
    tile_raster(ramp_raster, src, src_mode, max_zoom=ZOOM, min_zoom=ZOOM)

    dst = tmp_path / "datapng"
    target = make_numerical_mode(encoding=NumericalEncoding(factor=0.001))
    convert_tree(src, dst, SourceTiles(encoding=src_mode.encoding), target)

    before, _ = decode_tile(src_mode, src, ZOOM, TX, TY)
    after, _ = decode_tile(target, dst, ZOOM, TX, TY)
    assert np.abs(after - before).max() <= (1 / 256) / 2 + 0.001 / 2 + 1e-9


def test_形式も変換できる(tmp_path, mapbox_tiles):
    src, src_mode = mapbox_tiles
    dst = tmp_path / "png"
    target = make_numerical_mode(encoding=NumericalEncoding(factor=0.01), fmt=TileFormat("png"))
    convert_tree(src, dst, SourceTiles(encoding=src_mode.encoding), target)

    assert tile_path(dst, ZOOM, TX, TY, ".png").exists()
    assert not tile_path(dst, ZOOM, TX, TY, ".webp").exists()


def test_無効値が引き継がれる(tmp_path, holes_raster):
    src = tmp_path / "src"
    src_mode = make_numerical_mode(resampling="nearest")
    tile_raster(holes_raster, src, src_mode, max_zoom=ZOOM, min_zoom=ZOOM)

    dst = tmp_path / "dst"
    target = make_numerical_mode(encoding=NumericalEncoding(factor=0.01))
    convert_tree(src, dst, SourceTiles(encoding=src_mode.encoding), target)

    _, valid = decode_tile(target, dst, ZOOM, TX, TY)
    assert not valid[:, : TILE_SIZE // 2].any()
    assert valid[:, TILE_SIZE // 2 :].all()


def test_アルファ無しの入力は無効色で判定する(tmp_path, holes_raster):
    src = tmp_path / "src"
    src_mode = make_numerical_mode(resampling="nearest", invalid_color=(128, 0, 0))
    tile_raster(holes_raster, src, src_mode, max_zoom=ZOOM, min_zoom=ZOOM)

    dst = tmp_path / "dst"
    target = make_numerical_mode(encoding=NumericalEncoding(factor=0.01))
    convert_tree(
        src,
        dst,
        SourceTiles(encoding=src_mode.encoding, invalid_color=(128, 0, 0)),
        target,
    )

    _, valid = decode_tile(target, dst, ZOOM, TX, TY)
    assert not valid[:, : TILE_SIZE // 2].any()
    assert valid[:, TILE_SIZE // 2 :].all()


def test_既存出力はスキップし_overwrite_で作り直す(tmp_path, mapbox_tiles):
    src, src_mode = mapbox_tiles
    dst = tmp_path / "dst"
    source = SourceTiles(encoding=src_mode.encoding)
    target = make_numerical_mode(encoding=NumericalEncoding(factor=0.01))

    first = convert_tree(src, dst, source, target)
    again = convert_tree(src, dst, source, target)
    assert again.written == 0
    assert again.skipped == first.written

    forced = convert_tree(src, dst, source, target, overwrite=True)
    assert forced.written == first.written


def test_並列と逐次で同じ結果になる(tmp_path, mapbox_tiles):
    src, src_mode = mapbox_tiles
    source = SourceTiles(encoding=src_mode.encoding)
    target = make_numerical_mode(encoding=NumericalEncoding(factor=0.01))

    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    convert_tree(src, serial, source, target, processes=1)
    convert_tree(src, parallel, source, target, processes=4)

    def tree(root):
        return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*.webp"))}

    assert tree(serial) == tree(parallel)


def test_空のタイル木はエラー(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="タイルが 1 枚もありません"):
        convert_tree(tmp_path / "empty", tmp_path / "dst", SourceTiles(), make_numerical_mode())


def test_タイルサイズの食い違いはエラー(tmp_path, mapbox_tiles):
    src, src_mode = mapbox_tiles
    target = make_numerical_mode(tile_size=TILE_SIZE * 2)
    with pytest.raises(ValueError, match="tile-size"):
        convert_tree(src, tmp_path / "dst", SourceTiles(encoding=src_mode.encoding), target)


def test_変換後のタイルはアルファの有無が正しい(tmp_path, mapbox_tiles):
    """全画素有効ならアルファチャンネルを持たない（容量削減 + 宣言の一貫性）。"""
    src, src_mode = mapbox_tiles
    dst = tmp_path / "dst"
    target = make_numerical_mode(encoding=NumericalEncoding(factor=0.01))
    convert_tree(src, dst, SourceTiles(encoding=src_mode.encoding), target)

    _, alpha = load_tile(tile_path(dst, ZOOM, TX, TY, ".webp"))
    assert alpha is None
