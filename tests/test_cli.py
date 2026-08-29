"""CLI のスモークテスト（各サブコマンドが一通り通ること）。

細かい正しさは各モジュールのテストで見ているので、ここでは「引数が組み立てられ、
生成物が揃い、宣言と実タイルが食い違わない」ことを確かめる。
"""

from __future__ import annotations

import json

import pytest

from datapng_tiler.cli import main
from datapng_tiler.validate import validate
from tests.helpers import TX, TY, ZOOM, checker_classes, write_class_raster

LEGEND_YAML = """
title: 浸水深
items:
  - value: 1
    r: 245
    g: 245
    b: 50
    title: 0.5m未満
  - value: 2
    r: 255
    g: 216
    b: 0
    title: 0.5〜3.0m
  - value: 3
    r: 255
    g: 40
    b: 0
    title: 5.0〜10.0m
"""


def run(*argv: str) -> int:
    return main(list(argv))


# --- tile ---------------------------------------------------------------------------


def test_数値型の一式が生成され検証を通る(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    assert (
        run(
            "tile",
            str(ramp_raster),
            "-o",
            str(out),
            "--factor",
            "0.001",
            "--unit",
            "m",
            "--tile-size",
            "64",
            "-z",
            str(ZOOM),
            "--min-zoom",
            str(ZOOM - 1),
            "--name",
            "標高テスト",
            "--description",
            "標高は東京湾平均海面（T.P.）基準。",
            "-j",
            "2",
        )
        == 0
    )

    tilejson = out / "tiles.json"
    assert tilejson.exists()
    assert (out / "index.html").exists()
    assert (out / str(ZOOM) / str(TX) / f"{TY}.webp").exists()

    doc = json.loads(tilejson.read_text(encoding="utf-8"))
    assert doc["name"] == "標高テスト"
    assert doc["tileSize"] == 64
    assert doc["format"] == "webp"
    assert doc["datapng"]["unit"] == "m"
    assert "T.P." in doc["description"]

    assert validate(tilejson, out) == []


def test_PNG_出力もできる(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    assert (
        run(
            "tile",
            str(ramp_raster),
            "-o",
            str(out),
            "--format",
            "png",
            "--tile-size",
            "64",
            "-z",
            str(ZOOM),
            "--min-zoom",
            str(ZOOM),
        )
        == 0
    )
    assert (out / str(ZOOM) / str(TX) / f"{TY}.png").exists()
    doc = json.loads((out / "tiles.json").read_text(encoding="utf-8"))
    assert doc["format"] == "png"
    assert doc["tiles"] == ["./{z}/{x}/{y}.png"]


def test_パレット型の一式が生成される(tmp_path):
    legend = tmp_path / "legend.yaml"
    legend.write_text(LEGEND_YAML, encoding="utf-8")
    src = write_class_raster(tmp_path / "classes.tif", checker_classes())

    out = tmp_path / "tiles"
    assert (
        run(
            "tile",
            str(src),
            "-o",
            str(out),
            "--type",
            "palette",
            "--legend",
            str(legend),
            "--tile-size",
            "32",
            "-z",
            str(ZOOM),
            "--min-zoom",
            str(ZOOM - 1),
        )
        == 0
    )

    doc = json.loads((out / "tiles.json").read_text(encoding="utf-8"))
    assert doc["datapng"]["type"] == "palette"
    assert doc["datapng"]["legend"]["title"] == "浸水深"
    assert validate(out / "tiles.json", out) == []


def test_パレット型で凡例を外部参照にすると凡例ファイルも出る(tmp_path):
    legend = tmp_path / "legend.yaml"
    legend.write_text(LEGEND_YAML, encoding="utf-8")
    src = write_class_raster(tmp_path / "classes.tif", checker_classes())

    out = tmp_path / "tiles"
    assert (
        run(
            "tile",
            str(src),
            "-o",
            str(out),
            "--type",
            "palette",
            "--legend",
            str(legend),
            "--legend-url",
            "https://example.org/legend.json",
            "--tile-size",
            "32",
            "-z",
            str(ZOOM),
            "--min-zoom",
            str(ZOOM),
        )
        == 0
    )
    doc = json.loads((out / "tiles.json").read_text(encoding="utf-8"))
    assert doc["datapng"]["legend"] == "https://example.org/legend.json"
    # 参照先の実体も出す（存在しない URL を書かない）
    written = json.loads((out / "legend.json").read_text(encoding="utf-8"))
    assert written["items"][0]["title"] == "0.5m未満"


def test_パレット型に凡例が無ければエラー(tmp_path, ramp_raster):
    with pytest.raises(SystemExit):
        run("tile", str(ramp_raster), "-o", str(tmp_path / "out"), "--type", "palette")


def test_invalid_color_だけの指定はエラー(tmp_path, ramp_raster):
    """仕様 §3.2.2: アルファ付きタイルに invalidColor は指定できない。"""
    with pytest.raises(SystemExit, match="no-alpha"):
        run(
            "tile",
            str(ramp_raster),
            "-o",
            str(tmp_path / "out"),
            "--invalid-color",
            "128",
            "0",
            "0",
        )


def test_no_alpha_でアルファ無しのタイルになる(tmp_path, holes_raster):
    out = tmp_path / "tiles"
    assert (
        run(
            "tile",
            str(holes_raster),
            "-o",
            str(out),
            "--factor",
            "0.001",
            "--tile-size",
            "64",
            "-z",
            str(ZOOM),
            "--min-zoom",
            str(ZOOM),
            "--resampling",
            "nearest",
            "--no-alpha",
        )
        == 0
    )
    doc = json.loads((out / "tiles.json").read_text(encoding="utf-8"))
    assert doc["datapng"]["invalidColor"] == [128, 0, 0]
    assert validate(out / "tiles.json", out) == []


def test_値が範囲を超えるとエラーで止まる(tmp_path, ramp_raster, capsys):
    """黙って折り返した誤値を出力しない。"""
    code = run(
        "tile",
        str(ramp_raster),
        "-o",
        str(tmp_path / "out"),
        "--factor",
        "0.000001",
        "--tile-size",
        "64",
        "-z",
        str(ZOOM),
        "--min-zoom",
        str(ZOOM),
    )
    assert code == 1
    assert "factor" in capsys.readouterr().err


def test_背景地図を選ぶと規約の所在を伝える(tmp_path, ramp_raster, capsys):
    out = tmp_path / "tiles"
    run(
        "tile",
        str(ramp_raster),
        "-o",
        str(out),
        "--tile-size",
        "64",
        "-z",
        str(ZOOM),
        "--min-zoom",
        str(ZOOM),
        "--basemap",
        "osm",
    )
    err = capsys.readouterr().err
    assert "operations.osmfoundation.org" in err
    assert "openstreetmap.org" in (out / "index.html").read_text(encoding="utf-8")


def test_プレビューと_TileJSON_を抑止できる(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    run(
        "tile",
        str(ramp_raster),
        "-o",
        str(out),
        "--tile-size",
        "64",
        "-z",
        str(ZOOM),
        "--min-zoom",
        str(ZOOM),
        "--no-tilejson",
    )
    assert not (out / "tiles.json").exists()
    assert not (out / "index.html").exists()


# --- tilejson / validate / inspect ---------------------------------------------------


def test_tilejson_を作り直せる(tmp_path, ramp_raster):
    out = tmp_path / "tiles"
    run(
        "tile",
        str(ramp_raster),
        "-o",
        str(out),
        "--factor",
        "0.001",
        "--tile-size",
        "64",
        "-z",
        str(ZOOM),
        "--min-zoom",
        str(ZOOM),
        "--no-tilejson",
    )
    target = tmp_path / "custom.json"
    assert (
        run(
            "tilejson",
            str(out),
            "-o",
            str(target),
            "--factor",
            "0.001",
            "--unit",
            "m",
            "--tile-size",
            "64",
            "--tiles-url",
            "https://example.org/{z}/{x}/{y}.webp",
        )
        == 0
    )
    doc = json.loads(target.read_text(encoding="utf-8"))
    assert doc["tiles"] == ["https://example.org/{z}/{x}/{y}.webp"]
    assert doc["datapng"]["factor"] == 0.001


def test_validate_は問題があれば終了コード1(tmp_path, ramp_raster, capsys):
    out = tmp_path / "tiles"
    run(
        "tile",
        str(ramp_raster),
        "-o",
        str(out),
        "--factor",
        "0.001",
        "--tile-size",
        "64",
        "-z",
        str(ZOOM),
        "--min-zoom",
        str(ZOOM),
    )
    assert run("validate", str(out / "tiles.json"), "--tiles", str(out)) == 0

    doc = json.loads((out / "tiles.json").read_text(encoding="utf-8"))
    doc["maxzoom"] = ZOOM + 5
    (out / "tiles.json").write_text(json.dumps(doc), encoding="utf-8")
    assert run("validate", str(out / "tiles.json"), "--tiles", str(out)) == 1
    assert "maxzoom" in capsys.readouterr().err


def test_inspect_で画素の値を確認できる(tmp_path, ramp_raster, capsys):
    out = tmp_path / "tiles"
    run(
        "tile",
        str(ramp_raster),
        "-o",
        str(out),
        "--factor",
        "0.001",
        "--unit",
        "m",
        "--tile-size",
        "64",
        "-z",
        str(ZOOM),
        "--min-zoom",
        str(ZOOM),
    )
    tile = out / str(ZOOM) / str(TX) / f"{TY}.webp"
    assert (
        run("inspect", str(tile), "--tilejson", str(out / "tiles.json"), "--pixel", "0", "0") == 0
    )
    output = capsys.readouterr().out
    assert "有効画素" in output
    assert "画素 (0, 0)" in output


def test_inspect_はパレット型で色の内訳を出す(tmp_path, capsys):
    legend = tmp_path / "legend.yaml"
    legend.write_text(LEGEND_YAML, encoding="utf-8")
    src = write_class_raster(tmp_path / "classes.tif", checker_classes())
    out = tmp_path / "tiles"
    run(
        "tile",
        str(src),
        "-o",
        str(out),
        "--type",
        "palette",
        "--legend",
        str(legend),
        "--tile-size",
        "32",
        "-z",
        str(ZOOM),
        "--min-zoom",
        str(ZOOM),
    )
    tile = out / str(ZOOM) / str(TX) / f"{TY}.webp"
    capsys.readouterr()  # ここまでの出力は捨てる
    assert run("inspect", str(tile), "--tilejson", str(out / "tiles.json")) == 0
    output = capsys.readouterr().out
    assert "色の内訳" in output
    assert "0.5m未満" in output


# --- convert ------------------------------------------------------------------------


def test_convert_で既存タイルを移せる(tmp_path, ramp_raster):
    mapbox = tmp_path / "mapbox"
    run(
        "tile",
        str(ramp_raster),
        "-o",
        str(mapbox),
        "--encoding",
        "mapbox",
        "--tile-size",
        "64",
        "-z",
        str(ZOOM),
        "--min-zoom",
        str(ZOOM),
        "--no-viewer",
    )

    datapng = tmp_path / "datapng"
    assert (
        run(
            "convert",
            str(mapbox),
            "-o",
            str(datapng),
            "--from",
            "mapbox",
            "--factor",
            "0.01",
            "--unit",
            "m",
            "--tile-size",
            "64",
            "--tilejson-in",
            str(mapbox / "tiles.json"),
        )
        == 0
    )
    doc = json.loads((datapng / "tiles.json").read_text(encoding="utf-8"))
    assert doc["datapng"]["factor"] == 0.01
    assert "specialEncoding" not in doc["datapng"]
    assert validate(datapng / "tiles.json", datapng) == []


def test_バージョンを表示できる(capsys):
    with pytest.raises(SystemExit) as excinfo:
        run("--version")
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip()


# --- 引数の取りこぼしを防ぐ ---------------------------------------------------------


def test_inspect_の範囲外の画素指定はエラー(tmp_path, ramp_raster, capsys):
    """添字をそのまま渡すと、範囲外はトレースバック、負値は反対側の値を黙って返す。"""
    out = tmp_path / "tiles"
    run(
        "tile",
        str(ramp_raster),
        "-o",
        str(out),
        "--factor",
        "0.001",
        "--tile-size",
        "64",
        "-z",
        str(ZOOM),
        "--min-zoom",
        str(ZOOM),
        "--no-viewer",
    )
    tile = out / str(ZOOM) / str(TX) / f"{TY}.webp"
    capsys.readouterr()

    assert run("inspect", str(tile), "--pixel", "9999", "0") == 1
    assert "範囲外" in capsys.readouterr().err

    assert run("inspect", str(tile), "--pixel", "-1", "0") == 1
    assert "範囲外" in capsys.readouterr().err


def test_パレット型で数値型専用の無効値オプションを拒む(tmp_path):
    """黙って無視すると、指定したつもりの出力と実物が食い違う。"""
    legend = tmp_path / "legend.yaml"
    legend.write_text(LEGEND_YAML, encoding="utf-8")
    src = write_class_raster(tmp_path / "classes.tif", checker_classes())

    with pytest.raises(SystemExit, match="数値型専用"):
        run(
            "tile",
            str(src),
            "-o",
            str(tmp_path / "out"),
            "--type",
            "palette",
            "--legend",
            str(legend),
            "--no-alpha",
        )
