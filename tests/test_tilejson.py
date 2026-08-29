"""TileJSON 生成と仕様適合検証のテスト。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from datapng_tiler.codec import NumericalEncoding
from datapng_tiler.engine import tile_raster
from datapng_tiler.imageio import TileFormat
from datapng_tiler.legend import Legend
from datapng_tiler.modes import NumericalMode, PaletteMode
from datapng_tiler.tilejson import build_tilejson, from_tree, write_tilejson
from datapng_tiler.validate import load_schema, validate, validate_document, validate_tiles
from tests.helpers import (
    TILE_SIZE,
    TX,
    TY,
    ZOOM,
)
from tests.helpers import (
    make_numerical_mode as make_mode,
)

BASE_KWARGS = {
    "tiles": ["https://example.org/{z}/{x}/{y}.webp"],
    "bounds": (139.0, 35.0, 140.0, 36.0),
    "minzoom": 0,
    "maxzoom": 14,
    "tile_size": 512,
}


def numerical_datapng(**kwargs):
    return NumericalMode(tile_size=512, encoding=NumericalEncoding(factor=0.01), **kwargs).datapng()


# --- 生成 ---------------------------------------------------------------------------


def test_必須フィールドが揃う():
    doc = build_tilejson(**BASE_KWARGS, datapng=numerical_datapng(unit="m"))
    assert doc["tilejson"] == "3.0.0"
    assert doc["tiles"] == ["https://example.org/{z}/{x}/{y}.webp"]
    assert doc["tileSize"] == 512  # 仕様 §2.1 REQUIRED
    # 形式はタイル URL の拡張子が担うので、専用フィールドは出さない（仕様 §2.2）
    assert "format" not in doc
    assert doc["bounds"] == [139.0, 35.0, 140.0, 36.0]
    assert doc["center"] == [139.5, 35.5, 0]
    assert doc["datapng"]["type"] == "numerical"
    assert doc["datapng"]["factor"] == 0.01
    assert doc["datapng"]["unit"] == "m"


def test_タイルURLが空ならエラー():
    with pytest.raises(ValueError, match="tiles"):
        build_tilejson(**{**BASE_KWARGS, "tiles": []}, datapng=numerical_datapng())


def test_アルファで無効値を表すとき_invalidColor_を出さない():
    """仕様 §3.2.2 MUST NOT。既定（アルファ）では宣言してはならない。"""
    assert "invalidColor" not in numerical_datapng()
    # 明示的にアルファ無し出力を選んだときだけ出す
    assert numerical_datapng(invalid_color=(128, 0, 0))["invalidColor"] == [128, 0, 0]


def test_鉛直基準面は_description_に入れる():
    doc = build_tilejson(
        **BASE_KWARGS,
        datapng=numerical_datapng(unit="m"),
        description="標高は東京湾平均海面（T.P.）基準。",
    )
    assert "T.P." in doc["description"]
    # 仕様 0.6.0 で verticalCrs は廃止され description へ移った
    assert "verticalCrs" not in doc["datapng"]


def test_タイル木から範囲とズームを実測する(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    mode = make_mode()
    result = tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM - 2)

    doc = from_tree(out, mode, name="ramp")
    assert doc["minzoom"] == result.min_zoom
    assert doc["maxzoom"] == result.max_zoom
    assert doc["tileSize"] == TILE_SIZE
    assert doc["tiles"] == ["./{z}/{x}/{y}.webp"]
    assert doc["name"] == "ramp"


def test_タイルが無ければエラー(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="タイルが 1 枚もありません"):
        from_tree(tmp_path / "empty", make_mode())


# --- スキーマ検証 -------------------------------------------------------------------


def test_同梱スキーマが読める():
    schema = load_schema()
    assert schema["title"] == "TileJSON DataPNG Extension"
    assert schema["required"] == ["type"]


def test_生成した_TileJSON_は検証を通る():
    doc = build_tilejson(**BASE_KWARGS, datapng=numerical_datapng(unit="m"))
    assert validate_document(doc) == []


def test_パレット型の_TileJSON_も検証を通る():
    legend = Legend.from_dict({"items": [{"r": 1, "g": 2, "b": 3, "title": "あ"}]})
    doc = build_tilejson(**BASE_KWARGS, datapng=PaletteMode(tile_size=512, legend=legend).datapng())
    assert validate_document(doc) == []


@pytest.mark.parametrize(
    ("mutate", "where"),
    [
        (lambda d: d.update(tilejson="2.2.0"), "tilejson"),
        (lambda d: d.update(tiles=["https://example.org/tile.webp"]), "tiles[0]"),
        (lambda d: d.pop("tileSize"), "tileSize"),
        (lambda d: d.update(minzoom=20), "minzoom"),
        (lambda d: d.update(bounds=[140.0, 35.0, 139.0, 36.0]), "bounds"),
        (lambda d: d.pop("datapng"), "datapng"),
        (lambda d: d["datapng"].pop("type"), "datapng"),
        (lambda d: d["datapng"].update(type="unknown"), "datapng['type']"),
        (lambda d: d["datapng"].update(invalidColor=[1, 2]), "datapng['invalidColor']"),
        (lambda d: d["datapng"].update(precision=0), "datapng['precision']"),
    ],
)
def test_不正な_TileJSON_を検出する(mutate, where):
    doc = build_tilejson(**BASE_KWARGS, datapng=numerical_datapng(unit="m"))
    mutate(doc)
    problems = validate_document(doc)
    assert any(p.where == where for p in problems), problems


def test_パレット型に数値型のキーがあると指摘する():
    legend = Legend.from_dict({"items": [{"r": 1, "g": 2, "b": 3, "title": "あ"}]})
    doc = build_tilejson(**BASE_KWARGS, datapng=PaletteMode(tile_size=512, legend=legend).datapng())
    doc["datapng"]["factor"] = 0.01
    assert any(p.where == "datapng.factor" for p in validate_document(doc))


def test_specialEncoding_と_factor_の併記を指摘する():
    """仕様 §3.2.1: specialEncoding 指定時 factor・offset は無視される（MUST）。"""
    doc = build_tilejson(**BASE_KWARGS, datapng={"type": "numerical", "specialEncoding": "mapbox"})
    doc["datapng"]["factor"] = 0.01
    assert any(p.where == "datapng.factor" for p in validate_document(doc))


def test_block_support_の_anchor_を指摘する():
    doc = build_tilejson(
        **BASE_KWARGS,
        datapng={"type": "numerical", "support": {"type": "block", "anchor": "center"}},
    )
    assert any(p.where == "datapng.support.anchor" for p in validate_document(doc))


# --- 実タイルとの突合 ---------------------------------------------------------------


def test_生成物一式が検証を通る(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    mode = make_mode()
    tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM - 2)
    path = write_tilejson(from_tree(out, mode, name="ramp"), out / "tiles.json")

    assert validate(path, out) == []


def test_ズーム宣言の食い違いを検出する(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    mode = make_mode()
    tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM - 1)
    doc = from_tree(out, mode)
    doc["maxzoom"] = ZOOM + 3

    assert any(p.where == "maxzoom" for p in validate_tiles(doc, out))


def test_タイルURLの拡張子と中身の食い違いを検出する(tmp_path, ramp_raster):
    """.webp というパスで PNG を配っている、という取り違えを見つける。"""
    from PIL import Image

    from datapng_tiler.fileio import tile_path

    out = tmp_path / "tiles"
    mode = make_mode(fmt=TileFormat("webp"))
    tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)
    doc = from_tree(out, mode)
    assert validate_tiles(doc, out) == []

    # 拡張子は .webp のまま、中身だけ PNG に差し替える
    victim = tile_path(out, ZOOM, TX, TY, ".webp")
    with Image.open(victim) as image:
        rgb = image.convert("RGB")
    rgb.save(victim, "PNG")

    problems = validate_tiles(doc, out, sample=0)
    assert any("実体は png" in p.message for p in problems), problems


def test_タイルURLから拡張子を取り出す():
    from datapng_tiler.tilejson import extension_from_tiles_url

    assert extension_from_tiles_url("https://example.org/{z}/{x}/{y}.webp") == ".webp"
    assert extension_from_tiles_url("./{z}/{x}/{y}.png") == ".png"
    assert extension_from_tiles_url("https://example.org/{z}/{x}/{y}.WEBP") == ".webp"
    assert extension_from_tiles_url("https://example.org/{z}/{x}/{y}.png?v=2") == ".png"
    # 拡張子を持たない配信（content negotiation）は None
    assert extension_from_tiles_url("https://example.org/tiles/{z}/{x}/{y}") is None


def test_アルファ付きタイルへの_invalidColor_宣言を検出する(tmp_path, holes_raster):
    """仕様 §3.2.2 MUST NOT。TileJSON 単体では見えず、実タイルと突き合わせて初めて分かる。"""
    out = tmp_path / "tiles"
    mode = make_mode(resampling="nearest")
    tile_raster(holes_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)

    doc = from_tree(out, mode)
    doc["datapng"]["invalidColor"] = [128, 0, 0]
    assert any(p.where == "datapng.invalidColor" for p in validate_tiles(doc, out))


def test_凡例に無い色を検出する(tmp_path):
    from tests.helpers import checker_classes, make_palette_mode, write_class_raster

    src = write_class_raster(tmp_path / "classes.tif", checker_classes())
    out = tmp_path / "tiles"
    mode = make_palette_mode()
    tile_raster(src, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)

    doc = from_tree(out, mode)
    assert validate_tiles(doc, out) == []

    # 凡例から 1 項目落とすと、その色を使っているタイルが指摘される
    doc["datapng"]["legend"]["items"] = doc["datapng"]["legend"]["items"][:1]
    assert any("凡例に無い色" in p.message for p in validate_tiles(doc, out))


def test_タイルが無ければ指摘する(tmp_path):
    (tmp_path / "empty").mkdir()
    doc = build_tilejson(**BASE_KWARGS, datapng=numerical_datapng())
    problems = validate_tiles(doc, tmp_path / "empty")
    assert any("タイルが 1 枚もありません" in p.message for p in problems)


def test_書き出した_TileJSON_は_UTF8_の_JSON(tmp_path):
    doc = build_tilejson(**BASE_KWARGS, datapng=numerical_datapng(unit="m"), name="標高")
    path = write_tilejson(doc, tmp_path / "tiles.json")
    text = path.read_text(encoding="utf-8")
    assert "標高" in text, "日本語をエスケープせずそのまま書く"
    assert json.loads(text) == doc


def test_サンプル数を絞っても決定的(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    mode = make_mode()
    tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM - 2)
    doc = from_tree(out, mode)
    assert validate_tiles(doc, out, sample=3) == validate_tiles(doc, out, sample=3)


def test_復号結果が原典と一致することを確かめられる(tmp_path, ramp_raster, ramp_value):
    """TileJSON の宣言（factor）だけでタイルを復号し、真値に戻せること。"""
    from datapng_tiler.fileio import tile_path
    from datapng_tiler.geo import tile_sample_x3857, tile_sample_y3857
    from datapng_tiler.imageio import load_tile

    out = tmp_path / "tiles"
    mode = make_mode()
    tile_raster(ramp_raster, out, mode, max_zoom=ZOOM, min_zoom=ZOOM)
    doc = from_tree(out, mode)

    # 配信物（TileJSON + タイル）だけを見て復号する
    factor = doc["datapng"]["factor"]
    offset = doc["datapng"].get("offset", 0.0)
    rgb, _ = load_tile(tile_path(out, ZOOM, TX, TY, ".webp"))
    r = rgb[..., 0].astype(np.int64)
    r = np.where(r < 128, r, r - 256)
    raw = r * 65536 + rgb[..., 1].astype(np.int64) * 256 + rgb[..., 2].astype(np.int64)
    values = factor * raw + offset

    expected = ramp_value(
        np.array([[tile_sample_x3857(ZOOM, TILE_SIZE, TX, c, True) for c in range(TILE_SIZE)]]),
        np.array([[tile_sample_y3857(ZOOM, TILE_SIZE, TY, r_, True)] for r_ in range(TILE_SIZE)]),
    )
    assert np.abs(values - expected).max() < factor / 2 + 1e-3
