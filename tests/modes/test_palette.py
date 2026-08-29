"""パレット型タイルのテスト。

クライアントは **RGB の完全一致**で凡例を引くので、色が 1 バイトでも変われば意味が
引けなくなる。ここでは「色が変わらないこと」を軸に検証する。
"""

from __future__ import annotations

import numpy as np
import pytest

from datapng_tiler.engine import tile_raster
from datapng_tiler.fileio import tile_path
from datapng_tiler.imageio import TileFormat, load_tile
from datapng_tiler.legend import Legend
from datapng_tiler.modes.palette import TRANSPARENT_INDEX, UnknownColorError, downsample_majority
from tests.helpers import (
    LEGEND,
    TX,
    TY,
    ZOOM,
    checker_classes,
    write_class_raster,
    write_rgb_raster,
)
from tests.helpers import (
    PALETTE_TILE_SIZE as TILE_SIZE,
)
from tests.helpers import (
    make_palette_mode as make_mode,
)

# --- 色の保存 -----------------------------------------------------------------------


@pytest.mark.parametrize("fmt", [TileFormat("webp"), TileFormat("png")], ids=["webp", "png"])
def test_クラス値ラスタの色が完全に保存される(tmp_path, fmt):
    src = write_class_raster(tmp_path / "classes.tif", checker_classes())
    out = tmp_path / f"tiles-{fmt.name}"
    mode = make_mode(fmt=fmt)
    tile_raster(src, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)

    rgb, alpha = load_tile(tile_path(out, ZOOM, TX, TY, fmt.extension))
    assert alpha is not None
    assert tuple(rgb[0, 0]) == (245, 245, 50)
    assert tuple(rgb[0, 1]) == (255, 216, 0)
    assert tuple(rgb[1, 0]) == (255, 40, 0)
    assert alpha[1, 1] == 0, "nodata は透明でなければならない"
    assert alpha[0, 0] == 255


def test_PNG_はインデックスカラーで書かれる(tmp_path):
    from PIL import Image

    src = write_class_raster(tmp_path / "classes.tif", checker_classes())
    out = tmp_path / "tiles"
    mode = make_mode(fmt=TileFormat("png"))
    tile_raster(src, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)

    with Image.open(tile_path(out, ZOOM, TX, TY, ".png")) as image:
        assert image.mode == "P", "パレット型 PNG はインデックスカラーで書く（容量削減）"
        assert image.info.get("transparency") == TRANSPARENT_INDEX


def test_RGB_ラスタの色が完全に保存される(tmp_path):
    rgb = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    rgb[:, :] = (245, 245, 50)
    rgb[:, TILE_SIZE // 2 :] = (255, 40, 0)
    src = write_rgb_raster(tmp_path / "rgb.tif", rgb)

    out = tmp_path / "tiles"
    mode = make_mode()
    tile_raster(src, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)

    loaded, alpha = load_tile(tile_path(out, ZOOM, TX, TY, ".webp"))
    assert alpha is None, "無効画素が無ければアルファチャンネルを持たない"
    assert np.array_equal(loaded, rgb)


# --- 凡例に無い色 -------------------------------------------------------------------


def test_凡例に無い色はエラーになる(tmp_path):
    rgb = np.full((TILE_SIZE, TILE_SIZE, 3), 7, dtype=np.uint8)
    src = write_rgb_raster(tmp_path / "rgb.tif", rgb)

    with pytest.raises(UnknownColorError) as excinfo:
        tile_raster(src, tmp_path / "tiles", make_mode(), max_zoom=ZOOM, min_zoom=ZOOM)
    assert "(7, 7, 7)" in str(excinfo.value)


def test_凡例に無い色を無効値として扱える(tmp_path):
    rgb = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    rgb[:, :] = (245, 245, 50)
    rgb[:, TILE_SIZE // 2 :] = (7, 7, 7)
    src = write_rgb_raster(tmp_path / "rgb.tif", rgb)

    out = tmp_path / "tiles"
    mode = make_mode(on_unknown_color="nodata")
    tile_raster(src, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)

    _, alpha = load_tile(tile_path(out, ZOOM, TX, TY, ".webp"))
    assert alpha is not None
    assert alpha[:, : TILE_SIZE // 2].all()
    assert not alpha[:, TILE_SIZE // 2 :].any()


def test_凡例に無いクラス値はエラーになる(tmp_path):
    classes = np.full((TILE_SIZE, TILE_SIZE), 9, dtype=np.int16)
    src = write_class_raster(tmp_path / "classes.tif", classes)
    with pytest.raises(UnknownColorError, match="9"):
        tile_raster(src, tmp_path / "tiles", make_mode(), max_zoom=ZOOM, min_zoom=ZOOM)


def test_value_が無い凡例ではクラス値ラスタを扱えない(tmp_path):
    legend = Legend.from_dict({"items": [{"r": 1, "g": 2, "b": 3, "title": "あ"}]})
    src = write_class_raster(tmp_path / "classes.tif", checker_classes())
    with pytest.raises(ValueError, match="value"):
        tile_raster(src, tmp_path / "tiles", make_mode(legend=legend), max_zoom=ZOOM, min_zoom=ZOOM)


# --- オーバービュー -----------------------------------------------------------------


def test_オーバービューが凡例に無い色を作らない(tmp_path):
    src = write_class_raster(tmp_path / "classes.tif", checker_classes())
    out = tmp_path / "tiles"
    mode = make_mode(support="block")  # 多数決
    tile_raster(src, out, mode, max_zoom=ZOOM, min_zoom=ZOOM - 2)

    allowed = {(0, 0, 0), *LEGEND.colors()}
    for zoom in (ZOOM - 2, ZOOM - 1, ZOOM):
        for tile in (out / str(zoom)).rglob("*.webp"):
            rgb, alpha = load_tile(tile)
            visible = rgb.reshape(-1, 3) if alpha is None else rgb[alpha > 0]
            colors = {tuple(int(c) for c in row) for row in np.unique(visible, axis=0)}
            assert colors <= allowed, f"{tile}: 凡例に無い色 {colors - allowed}"


def test_多数決は同数なら若い番号を採る():
    """同じ入力から必ず同じタイルが出る（決定的）ことの根拠。"""
    block = np.array([[1, 2], [2, 1]], dtype=np.uint8)
    assert downsample_majority(block).tolist() == [[1]]

    block = np.array([[3, 3], [1, 2]], dtype=np.uint8)
    assert downsample_majority(block).tolist() == [[3]]


def test_多数決は無効値を候補にしない():
    block = np.array([[0, 0], [0, 2]], dtype=np.uint8)
    assert downsample_majority(block).tolist() == [[2]]

    block = np.zeros((2, 2), dtype=np.uint8)
    assert downsample_majority(block).tolist() == [[TRANSPARENT_INDEX]]


def test_左上法のオーバービューは左上の色を運ぶ(tmp_path):
    src = write_class_raster(tmp_path / "classes.tif", checker_classes())
    out = tmp_path / "tiles"
    mode = make_mode(support="point")
    tile_raster(src, out, mode, max_zoom=ZOOM, min_zoom=ZOOM - 1)

    child, _ = load_tile(tile_path(out, ZOOM, TX, TY, ".webp"))
    parent, _ = load_tile(tile_path(out, ZOOM - 1, TX // 2, TY // 2, ".webp"))
    row0 = (TY % 2) * (TILE_SIZE // 2)
    col0 = (TX % 2) * (TILE_SIZE // 2)
    quadrant = parent[row0 : row0 + TILE_SIZE // 2, col0 : col0 + TILE_SIZE // 2]
    assert np.array_equal(quadrant, child[::2, ::2])


# --- TileJSON へ載せるフィールド -----------------------------------------------------


def test_datapng_は凡例を含み数値型のキーを出さない():
    fields = make_mode().datapng()
    assert fields["type"] == "palette"
    assert fields["legend"]["items"][0]["title"] == "0.5m未満"
    # 仕様 §3・§7: パレット型では factor / offset / invalidColor は該当しない
    assert "factor" not in fields
    assert "offset" not in fields
    assert "invalidColor" not in fields


def test_凡例を外部参照にできる():
    fields = make_mode(legend_url="https://example.org/legend.json").datapng()
    assert fields["legend"] == "https://example.org/legend.json"
