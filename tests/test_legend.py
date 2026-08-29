"""凡例定義の読み込みと検証のテスト。"""

from __future__ import annotations

import json

import pytest

from datapng_tiler.legend import Legend, LegendError

SAMPLE = """
title: 浸水深
items:
  - value: 1
    r: 245
    g: 245
    b: 50
    title: 0.5m未満
    description: 床下浸水相当。
  - value: 2
    r: 255
    g: 216
    b: 0
    title: 0.5〜3.0m
"""


def test_YAML_から読める(tmp_path):
    path = tmp_path / "legend.yaml"
    path.write_text(SAMPLE, encoding="utf-8")
    legend = Legend.load(path)

    assert legend.title == "浸水深"
    assert len(legend.items) == 2
    assert legend.items[0].rgb == (245, 245, 50)
    assert legend.items[0].description == "床下浸水相当。"
    assert legend.items[1].description is None
    assert legend.has_values


def test_JSON_から読める(tmp_path):
    path = tmp_path / "legend.json"
    path.write_text(
        json.dumps({"items": [{"r": 1, "g": 2, "b": 3, "title": "あ"}]}), encoding="utf-8"
    )
    legend = Legend.load(path)
    assert legend.items[0].title == "あ"
    assert not legend.has_values


def test_TileJSON_へ載せる形に変換できる(tmp_path):
    path = tmp_path / "legend.yaml"
    path.write_text(SAMPLE, encoding="utf-8")
    legend = Legend.load(path)

    data = legend.to_dict()
    assert data["title"] == "浸水深"
    assert data["items"][0] == {
        "r": 245,
        "g": 245,
        "b": 50,
        "title": "0.5m未満",
        "description": "床下浸水相当。",
        "value": 1,
    }


def test_追加メンバーはそのまま通す():
    """仕様 §3.3.1: 凡例項目には任意のメンバーを追加できる。"""
    legend = Legend.from_dict(
        {"items": [{"r": 0, "g": 0, "b": 0, "title": "黒", "symbol": "https://例/a.png"}]}
    )
    assert legend.to_dict()["items"][0]["symbol"] == "https://例/a.png"


def test_外部参照用に書き出せる(tmp_path):
    legend = Legend.from_dict({"items": [{"r": 1, "g": 2, "b": 3, "title": "あ"}]})
    out = legend.dump(tmp_path / "legend.json")
    assert json.loads(out.read_text(encoding="utf-8")) == legend.to_dict()


# --- 検証 ---------------------------------------------------------------------------


def test_items_が無いとエラー():
    with pytest.raises(LegendError, match="items"):
        Legend.from_dict({"title": "だけ"})


def test_空の_items_はエラー():
    with pytest.raises(LegendError, match="items"):
        Legend.from_dict({"items": []})


def test_同じ色が2回現れるとエラー():
    """色から意味を一意に引けなくなる（クライアントは RGB 完全一致で引く）。"""
    with pytest.raises(LegendError, match="同じ色"):
        Legend.from_dict(
            {
                "items": [
                    {"r": 1, "g": 2, "b": 3, "title": "あ"},
                    {"r": 1, "g": 2, "b": 3, "title": "い"},
                ]
            }
        )


def test_同じ_value_が2回現れるとエラー():
    with pytest.raises(LegendError, match="value"):
        Legend.from_dict(
            {
                "items": [
                    {"value": 1, "r": 1, "g": 2, "b": 3, "title": "あ"},
                    {"value": 1, "r": 4, "g": 5, "b": 6, "title": "い"},
                ]
            }
        )


@pytest.mark.parametrize(
    ("item", "message"),
    [
        ({"g": 2, "b": 3, "title": "あ"}, "'r'"),
        ({"r": 300, "g": 2, "b": 3, "title": "あ"}, "0〜255"),
        ({"r": "1", "g": 2, "b": 3, "title": "あ"}, "0〜255"),
        ({"r": 1, "g": 2, "b": 3}, "'title'"),
        ({"r": 1, "g": 2, "b": 3, "title": ""}, "title"),
        ({"r": 1, "g": 2, "b": 3, "title": "あ", "description": 5}, "description"),
    ],
)
def test_項目の検証(item, message):
    with pytest.raises(LegendError, match=message):
        Legend.from_dict({"items": [item]})


def test_項目が多すぎるとエラー():
    """インデックスカラーは 256 色。0 番を無効値に予約するので 255 件まで。"""
    items = [{"r": i % 256, "g": i // 256, "b": 0, "title": f"{i}"} for i in range(256)]
    with pytest.raises(LegendError, match="255"):
        Legend.from_dict({"items": items})


def test_壊れた定義ファイルはパスを添えて報告する(tmp_path):
    path = tmp_path / "legend.yaml"
    path.write_text("items: [1, 2]", encoding="utf-8")
    with pytest.raises(LegendError, match="legend.yaml"):
        Legend.load(path)
