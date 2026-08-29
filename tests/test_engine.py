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
from datapng_tiler.geo import (
    ORIGIN_SHIFT,
    tile_resolution,
    tile_sample_x3857,
    tile_sample_y3857,
)
from datapng_tiler.imageio import TileFormat, load_tile
from datapng_tiler.modes import NumericalMode

TILE_SIZE = 64
ZOOM = 8
TX, TY = 227, 100
FACTOR = 0.001
MARGIN = 4


@pytest.fixture
def ramp_raster(tmp_path, mercator_ramp):
    """タイル (ZOOM, TX, TY) を余裕をもって覆う 1 次関数ラスタ。

    ソース画素の中心がタイルの節点に一致するよう原点を半画素ずらす。こうすると
    左上法の再投影が恒等になり、補間の影響を受けずに幾何だけを検証できる。
    """
    res = tile_resolution(ZOOM, TILE_SIZE)
    first_node_x = -ORIGIN_SHIFT + TX * TILE_SIZE * res
    first_node_y = ORIGIN_SHIFT - TY * TILE_SIZE * res
    return mercator_ramp(
        tmp_path / "ramp.tif",
        x0=first_node_x - (MARGIN + 0.5) * res,
        y0=first_node_y + (MARGIN + 0.5) * res,
        pixel_size=res,
        width=TILE_SIZE + 2 * MARGIN,
        height=TILE_SIZE + 2 * MARGIN,
    )


def make_mode(**kwargs) -> NumericalMode:
    defaults = {
        "tile_size": TILE_SIZE,
        "encoding": NumericalEncoding(factor=FACTOR),
        "fmt": TileFormat("webp"),
        "unit": "m",
    }
    return NumericalMode(**{**defaults, **kwargs})


def decode_tile(mode: NumericalMode, root, zoom, x, y):
    """タイルを読んで (値, 有効マスク) にする。"""
    rgb, alpha = load_tile(tile_path(root, zoom, x, y, mode.fmt.extension))
    values = mode.encoding.decode(rgb)
    valid = np.ones(values.shape, dtype=bool) if alpha is None else alpha > 0
    return values, valid


# --- 値の正確性 ---------------------------------------------------------------------


def test_タイル画素の値が原典の関数と一致する(tmp_path, ramp_raster, ramp_value):
    """生成したタイルを復号し、その画素が代表する座標での真値と比べる。"""
    out = tmp_path / "tiles"
    mode = make_mode()
    tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)

    values, valid = decode_tile(mode, out, ZOOM, TX, TY)
    assert valid.all()

    rows, cols = np.mgrid[0:TILE_SIZE, 0:TILE_SIZE]
    expected = ramp_value(
        np.array(
            [[tile_sample_x3857(ZOOM, TILE_SIZE, TX, int(c), True) for c in row] for row in cols]
        ),
        np.array(
            [[tile_sample_y3857(ZOOM, TILE_SIZE, TY, int(r), True) for r in row] for row in rows]
        ),
    )
    # 許容は量子化の半幅 + float32 で保持したことによる丸め
    assert np.abs(values - expected).max() < FACTOR / 2 + 1e-3


def test_無効値はアルファ0になり全無効タイルは書かれない(tmp_path, mercator_ramp):
    res = tile_resolution(ZOOM, TILE_SIZE)
    first_node_x = -ORIGIN_SHIFT + TX * TILE_SIZE * res
    first_node_y = ORIGIN_SHIFT - TY * TILE_SIZE * res
    src = mercator_ramp(
        tmp_path / "holes.tif",
        x0=first_node_x - 0.5 * res,
        y0=first_node_y + 0.5 * res,
        pixel_size=res,
        width=TILE_SIZE,
        height=TILE_SIZE,
        nodata=-9999.0,
        nodata_cols=slice(0, TILE_SIZE // 2),
    )
    out = tmp_path / "tiles"
    mode = make_mode(resampling="nearest")
    tile_raster(src, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)

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
    mode = make_mode(encoding=NumericalEncoding(factor=0.01), resampling="nearest")
    result = tile_raster(src, out, mode, max_zoom=10, min_zoom=10)
    assert result.base_tiles > 0

    seen: list[float] = []
    for zoom, x, y in _all_tiles(out, mode.fmt.extension):
        if zoom != 10:
            continue
        values, valid = decode_tile(mode, out, zoom, x, y)
        seen.extend(np.unique(np.round(values[valid], 2)).tolist())
    assert 0.5 in seen, "nodata=0 のとき 0.5 は有効値として残らなければならない"
    assert 0.0 not in seen


def _all_tiles(root, extension):
    for zoom_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        for x_dir in sorted(zoom_dir.iterdir()):
            for tile in sorted(x_dir.glob(f"*{extension}")):
                yield int(zoom_dir.name), int(x_dir.name), int(tile.stem)


# --- オーバービュー -----------------------------------------------------------------


def test_左上法のオーバービューは子タイルの左上画素を運ぶ(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    mode = make_mode(support="point")
    tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM - 1)

    parent_x, parent_y = TX // 2, TY // 2
    parent, _ = decode_tile(mode, out, ZOOM - 1, parent_x, parent_y)
    child, _ = decode_tile(mode, out, ZOOM, TX, TY)

    # 子 (TX, TY) は親の 2×2 のうち (TX%2, TY%2) の位置を占める
    row0 = (TY % 2) * (TILE_SIZE // 2)
    col0 = (TX % 2) * (TILE_SIZE // 2)
    quadrant = parent[row0 : row0 + TILE_SIZE // 2, col0 : col0 + TILE_SIZE // 2]
    assert np.array_equal(quadrant, child[::2, ::2])


def test_block_supportのオーバービューは整数平均になる(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    mode = make_mode(support="block")
    tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM - 1)

    parent, _ = decode_tile(mode, out, ZOOM - 1, TX // 2, TY // 2)
    child, _ = decode_tile(mode, out, ZOOM, TX, TY)

    row0 = (TY % 2) * (TILE_SIZE // 2)
    col0 = (TX % 2) * (TILE_SIZE // 2)
    quadrant = parent[row0 : row0 + TILE_SIZE // 2, col0 : col0 + TILE_SIZE // 2]

    raw = np.rint(child / FACTOR).astype(np.int64)
    blocks = raw.reshape(TILE_SIZE // 2, 2, TILE_SIZE // 2, 2)
    expected = np.rint(blocks.mean(axis=(1, 3))) * FACTOR
    assert np.abs(quadrant - expected).max() < FACTOR / 2 + 1e-9


def test_supportの宣言と生成方式が連動する():
    assert make_mode(support="point").overview_method == "topleft"
    assert make_mode(support="point").datapng()["support"] == {
        "type": "point",
        "anchor": "northwest",
    }
    assert make_mode(support="block").overview_method == "average"
    assert make_mode(support="block").datapng()["support"] == {"type": "block"}


def test_point_center_は受け付けない():
    with pytest.raises(ValueError, match="support"):
        make_mode(support="center")


# --- 再現性・並列・レジューム --------------------------------------------------------


def _tree_bytes(root, extension) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob(f"*{extension}"))}


def test_同じ入力から同じタイル木ができる(tmp_path, ramp_raster):
    mode = make_mode()
    first, second = tmp_path / "a", tmp_path / "b"
    tile_raster(ramp_raster, first, mode, max_zoom=ZOOM, min_zoom=ZOOM - 2)
    tile_raster(ramp_raster, second, mode, max_zoom=ZOOM, min_zoom=ZOOM - 2)
    assert _tree_bytes(first, mode.fmt.extension) == _tree_bytes(second, mode.fmt.extension)


def test_並列と逐次で同じタイル木ができる(tmp_path, ramp_raster):
    """ワーカへ状態を渡す経路が壊れていると、ここで値やタイル数が食い違う。"""
    mode = make_mode()
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    tile_raster(ramp_raster, serial, mode, max_zoom=ZOOM, min_zoom=ZOOM - 2, processes=1)
    tile_raster(ramp_raster, parallel, mode, max_zoom=ZOOM, min_zoom=ZOOM - 2, processes=4)
    assert _tree_bytes(serial, mode.fmt.extension) == _tree_bytes(parallel, mode.fmt.extension)


def test_モードは_pickle_できる():
    """spawn（macOS/Windows の既定）ではワーカへ pickle して渡すため。"""
    mode = make_mode(support="block", invalid_color=(128, 0, 0))
    assert pickle.loads(pickle.dumps(mode)) == mode


def test_既存タイルはスキップし_overwrite_で作り直す(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    mode = make_mode()
    first = tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)
    assert first.base_tiles > 0

    again = tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)
    assert again.base_tiles == 0, "既存タイルは作り直さない（レジューム）"

    forced = tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM, overwrite=True)
    assert forced.base_tiles == first.base_tiles


def test_0バイトの残骸は作り直される(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    mode = make_mode()
    tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)

    victim = tile_path(out, ZOOM, TX, TY, mode.fmt.extension)
    victim.write_bytes(b"")
    result = tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)
    assert result.base_tiles == 1
    assert victim.stat().st_size > 0


# --- 形式をまたいだ一致 --------------------------------------------------------------


def test_PNG_と_WebP_で復号結果が一致する(tmp_path, ramp_raster):
    webp_mode = make_mode(fmt=TileFormat("webp"))
    png_mode = make_mode(fmt=TileFormat("png"))
    webp_out, png_out = tmp_path / "webp", tmp_path / "png"
    tile_raster(ramp_raster, webp_out, webp_mode, max_zoom=ZOOM, min_zoom=ZOOM)
    tile_raster(ramp_raster, png_out, png_mode, max_zoom=ZOOM, min_zoom=ZOOM)

    from_webp, _ = decode_tile(webp_mode, webp_out, ZOOM, TX, TY)
    from_png, _ = decode_tile(png_mode, png_out, ZOOM, TX, TY)
    assert np.array_equal(from_webp, from_png)


# --- タイル木の走査 -----------------------------------------------------------------


def test_タイル木から範囲とズームを実測できる(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    mode = make_mode()
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
