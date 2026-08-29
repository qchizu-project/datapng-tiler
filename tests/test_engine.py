"""タイル生成エンジンのテスト（数値型を題材にする）。

「絵として正しそう」では検証にならないので、値が座標の 1 次関数になる合成ラスタを使い、
**タイル画素の値を解析的に予測して突き合わせる**。1 次関数は bilinear 補間で厳密に
再現されるため、リサンプリングによる誤差と幾何のずれを分離できる。
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from datapng_tiler.codec import NumericalEncoding
from datapng_tiler.engine import scan_tree, tile_raster
from datapng_tiler.fileio import tile_path
from datapng_tiler.geo import tile_sample_x3857, tile_sample_y3857
from datapng_tiler.imageio import TileFormat
from tests.helpers import (
    FACTOR,
    TILE_SIZE,
    TX,
    TY,
    ZOOM,
    all_tiles,
    decode_tile,
    make_numerical_mode,
    tree_bytes,
)


def sample_coordinates():
    """タイル (ZOOM, TX, TY) の各画素が代表する EPSG:3857 座標。"""
    xs = np.array([tile_sample_x3857(ZOOM, TILE_SIZE, TX, c, True) for c in range(TILE_SIZE)])
    ys = np.array([tile_sample_y3857(ZOOM, TILE_SIZE, TY, r, True) for r in range(TILE_SIZE)])
    return xs[np.newaxis, :], ys[:, np.newaxis]


# --- 値の正確性 ---------------------------------------------------------------------


def test_タイル画素の値が原典の関数と一致する(tmp_path, ramp_raster, ramp_value):
    """生成したタイルを復号し、その画素が代表する座標での真値と比べる。"""
    out = tmp_path / "tiles"
    mode = make_numerical_mode()
    tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)

    values, valid = decode_tile(mode, out, ZOOM, TX, TY)
    assert valid.all()

    xs, ys = sample_coordinates()
    expected = ramp_value(xs, ys)
    # 許容は量子化の半幅 + float32 で保持したことによる丸め
    assert np.abs(values - expected).max() < FACTOR / 2 + 1e-3


def test_無効値はアルファ0になり全無効タイルは書かれない(tmp_path, holes_raster):
    out = tmp_path / "tiles"
    mode = make_numerical_mode(resampling="nearest")
    tile_raster(holes_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)

    _, valid = decode_tile(mode, out, ZOOM, TX, TY)
    assert not valid[:, : TILE_SIZE // 2].any(), "nodata 領域は無効でなければならない"
    assert valid[:, TILE_SIZE // 2 :].all()

    # 隣（データが無い側）のタイルは 1 枚も書かれない
    assert not tile_path(out, ZOOM, TX + 1, TY, mode.fmt.extension).exists()


def test_無効値の判定に閾値を使わない(tmp_path, write_raster):
    """nodata が 0 のデータで、0 より大きい有効値まで無効化してしまわないこと。

    「nodata + 1.0 以下を無効」といった閾値判定だと、ここで 0.5 が消える。
    """
    data = np.array([[0.0, 0.5, 1.0, 2.0]], dtype=np.float32)
    data = np.repeat(np.repeat(data, 8, axis=0), 8, axis=1)
    src = write_raster(tmp_path / "zero_nodata.tif", data, pixel_size=0.01, nodata=0.0)

    out = tmp_path / "tiles"
    mode = make_numerical_mode(encoding=NumericalEncoding(factor=0.01), resampling="nearest")
    result = tile_raster(src, out, mode, max_zoom=10, min_zoom=10)
    assert result.base_tiles > 0

    seen: list[float] = []
    for zoom, x, y in all_tiles(out, mode.fmt.extension):
        if zoom != 10:
            continue
        values, valid = decode_tile(mode, out, zoom, x, y)
        seen.extend(np.unique(np.round(values[valid], 2)).tolist())
    assert 0.5 in seen, "nodata=0 のとき 0.5 は有効値として残らなければならない"
    assert 0.0 not in seen


def test_ソースが覆っていない領域は無効になる(tmp_path, write_raster):
    """nodata を持たないソースでも、ワープの被覆マスクで範囲外を無効にする。

    これが無いと、ソースの外側が 0 で埋まったまま有効値として符号化される。
    """
    # 45 度回転させた（＝矩形でない）ソースを作り、四隅が覆われないようにする
    import rasterio
    from rasterio.transform import Affine

    from datapng_tiler.geo import WEB_MERCATOR
    from tests.helpers import grid_origin

    x0, y0, res = grid_origin(TILE_SIZE)
    rotated = Affine.translation(x0, y0) * Affine.rotation(20) * Affine.scale(res, -res)
    path = tmp_path / "rotated.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=TILE_SIZE,
        width=TILE_SIZE,
        count=1,
        dtype="float32",
        crs=WEB_MERCATOR,
        transform=rotated,
    ) as dst:
        dst.write(np.full((TILE_SIZE, TILE_SIZE), 5.0, dtype=np.float32), 1)

    out = tmp_path / "tiles"
    mode = make_numerical_mode(encoding=NumericalEncoding(factor=0.01), resampling="nearest")
    tile_raster(path, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)

    values, valid = decode_tile(mode, out, ZOOM, TX, TY)
    assert not valid.all(), "回転したソースの外側は無効でなければならない"
    assert np.allclose(values[valid], 5.0), "有効画素は原典の値のまま"


# --- オーバービュー -----------------------------------------------------------------


def quadrant(parent: np.ndarray) -> np.ndarray:
    """親タイルのうち、子タイル (TX, TY) が占める 1/4 を切り出す。"""
    row0 = (TY % 2) * (TILE_SIZE // 2)
    col0 = (TX % 2) * (TILE_SIZE // 2)
    return parent[row0 : row0 + TILE_SIZE // 2, col0 : col0 + TILE_SIZE // 2]


def test_左上法のオーバービューは子タイルの左上画素を運ぶ(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    mode = make_numerical_mode(support="point")
    tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM - 1)

    parent, _ = decode_tile(mode, out, ZOOM - 1, TX // 2, TY // 2)
    child, _ = decode_tile(mode, out, ZOOM, TX, TY)
    assert np.array_equal(quadrant(parent), child[::2, ::2])


def test_block_supportのオーバービューは整数平均になる(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    mode = make_numerical_mode(support="block")
    tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM - 1)

    parent, _ = decode_tile(mode, out, ZOOM - 1, TX // 2, TY // 2)
    child, _ = decode_tile(mode, out, ZOOM, TX, TY)

    raw = np.rint(child / FACTOR).astype(np.int64)
    blocks = raw.reshape(TILE_SIZE // 2, 2, TILE_SIZE // 2, 2)
    expected = np.rint(blocks.mean(axis=(1, 3))) * FACTOR
    assert np.abs(quadrant(parent) - expected).max() < FACTOR / 2 + 1e-9


def test_supportの宣言と生成方式が連動する():
    assert make_numerical_mode(support="point").overview_method == "topleft"
    assert make_numerical_mode(support="point").datapng()["support"] == {
        "type": "point",
        "anchor": "northwest",
    }
    assert make_numerical_mode(support="block").overview_method == "average"
    assert make_numerical_mode(support="block").datapng()["support"] == {"type": "block"}


def test_point_center_は受け付けない():
    with pytest.raises(ValueError, match="support"):
        make_numerical_mode(support="center")


# --- 無効色（アルファ無し出力） ------------------------------------------------------


def test_アルファ無し出力では無効色が入る(tmp_path, holes_raster):
    out = tmp_path / "tiles"
    mode = make_numerical_mode(resampling="nearest", invalid_color=(128, 0, 0))
    tile_raster(holes_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)

    from datapng_tiler.imageio import load_tile

    rgb, alpha = load_tile(tile_path(out, ZOOM, TX, TY, mode.fmt.extension))
    assert alpha is None, "invalid_color 指定時はアルファチャンネルを持たない"
    assert tuple(rgb[0, 0]) == (128, 0, 0)
    assert mode.datapng()["invalidColor"] == [128, 0, 0]


def test_有効値が無効色と衝突したら弾く(tmp_path, write_raster):
    """(128,0,0) は rawValue = -8388608 でもある。両立させると復号で区別できない。"""
    from datapng_tiler.modes.numerical import InvalidColorCollision

    data = np.full((8, 8), -8388608.0, dtype=np.float32)
    src = write_raster(tmp_path / "collide.tif", data, pixel_size=0.01)
    mode = make_numerical_mode(
        encoding=NumericalEncoding(factor=1.0), invalid_color=(128, 0, 0), resampling="nearest"
    )
    with pytest.raises(InvalidColorCollision, match="128"):
        tile_raster(src, tmp_path / "tiles", mode, max_zoom=10, min_zoom=10)


# --- 再現性・並列・レジューム --------------------------------------------------------


def test_同じ入力から同じタイル木ができる(tmp_path, ramp_raster):
    mode = make_numerical_mode()
    first, second = tmp_path / "a", tmp_path / "b"
    tile_raster(ramp_raster, first, mode, max_zoom=ZOOM, min_zoom=ZOOM - 2)
    tile_raster(ramp_raster, second, mode, max_zoom=ZOOM, min_zoom=ZOOM - 2)
    assert tree_bytes(first, mode.fmt.extension) == tree_bytes(second, mode.fmt.extension)


def test_並列と逐次で同じタイル木ができる(tmp_path, ramp_raster):
    """ワーカへ状態を渡す経路が壊れていると、ここで値やタイル数が食い違う。"""
    mode = make_numerical_mode()
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    tile_raster(ramp_raster, serial, mode, max_zoom=ZOOM, min_zoom=ZOOM - 2, processes=1)
    tile_raster(ramp_raster, parallel, mode, max_zoom=ZOOM, min_zoom=ZOOM - 2, processes=4)
    assert tree_bytes(serial, mode.fmt.extension) == tree_bytes(parallel, mode.fmt.extension)


def test_モードは_pickle_できる():
    """spawn（macOS/Windows の既定）ではワーカへ pickle して渡すため。"""
    mode = make_numerical_mode(support="block", invalid_color=(128, 0, 0))
    assert pickle.loads(pickle.dumps(mode)) == mode


def test_既存タイルはスキップし_overwrite_で作り直す(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    mode = make_numerical_mode()
    first = tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)
    assert first.base_tiles > 0

    again = tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)
    assert again.base_tiles == 0, "既存タイルは作り直さない（レジューム）"

    forced = tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM, overwrite=True)
    assert forced.base_tiles == first.base_tiles


def test_0バイトの残骸は作り直される(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    mode = make_numerical_mode()
    tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)

    victim = tile_path(out, ZOOM, TX, TY, mode.fmt.extension)
    victim.write_bytes(b"")
    result = tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)
    assert result.base_tiles == 1
    assert victim.stat().st_size > 0


# --- 形式をまたいだ一致 --------------------------------------------------------------


def test_PNG_と_WebP_で復号結果が一致する(tmp_path, ramp_raster):
    webp_mode = make_numerical_mode(fmt=TileFormat("webp"))
    png_mode = make_numerical_mode(fmt=TileFormat("png"))
    webp_out, png_out = tmp_path / "webp", tmp_path / "png"
    tile_raster(ramp_raster, webp_out, webp_mode, max_zoom=ZOOM, min_zoom=ZOOM)
    tile_raster(ramp_raster, png_out, png_mode, max_zoom=ZOOM, min_zoom=ZOOM)

    from_webp, _ = decode_tile(webp_mode, webp_out, ZOOM, TX, TY)
    from_png, _ = decode_tile(png_mode, png_out, ZOOM, TX, TY)
    assert np.array_equal(from_webp, from_png)


# --- タイル木の走査 -----------------------------------------------------------------


def test_タイル木から範囲とズームを実測できる(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    mode = make_numerical_mode()
    result = tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM - 2)

    summary = scan_tree(out, mode.fmt.extension)
    assert summary is not None
    assert summary.max_zoom == ZOOM
    assert summary.min_zoom == ZOOM - 2
    assert summary.tile_count == result.total_tiles
    # 実測範囲はソース範囲を含む（タイル境界まで広がる）
    assert summary.bounds[0] <= result.bounds[0]
    assert summary.bounds[2] >= result.bounds[2]


def test_空のタイル木は_None(tmp_path):
    (tmp_path / "empty").mkdir()
    assert scan_tree(tmp_path / "empty") is None


def test_ソース範囲と交差しない指定はエラー(tmp_path, ramp_raster):
    mode = make_numerical_mode()
    with pytest.raises(ValueError, match="交差しません"):
        tile_raster(
            ramp_raster,
            tmp_path / "tiles",
            mode,
            max_zoom=ZOOM,
            min_zoom=ZOOM,
            bounds=(-10.0, -10.0, -9.0, -9.0),
        )


def test_存在しないディレクトリの走査は_None(tmp_path):
    """検証や TileJSON 生成の入口で、パスの打ち間違いが例外ではなく結果になる。"""
    assert scan_tree(tmp_path / "missing") is None
