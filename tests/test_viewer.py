"""プレビュー HTML 生成のテスト。"""

from __future__ import annotations

import pytest

from datapng_tiler.legend import Legend
from datapng_tiler.modes import PaletteMode
from datapng_tiler.tilejson import build_tilejson
from datapng_tiler.viewer import BASEMAPS, basemap_notice, build_viewer_html, write_viewer_html
from tests.helpers import make_numerical_mode

TILEJSON = build_tilejson(
    tiles=["./{z}/{x}/{y}.webp"],
    bounds=(139.0, 35.0, 140.0, 36.0),
    minzoom=8,
    maxzoom=12,
    tile_size=512,
    fmt="webp",
    datapng=make_numerical_mode(tile_size=512, unit="m").datapng(),
    name="標高",
)


def test_単一_HTML_が生成される():
    html = build_viewer_html(TILEJSON)
    assert html.startswith("<!DOCTYPE html>")
    assert '<div id="map">' in html
    assert "leaflet.js" in html


def test_背景地図は既定で無し():
    """公開ツールが第三者タイルサービスの規約を全ユーザーに負わせないため。"""
    html = build_viewer_html(TILEJSON)
    for info in BASEMAPS.values():
        assert info["url"] not in html
    assert '"basemap": null' in html.replace(" ", " ")


@pytest.mark.parametrize("name", sorted(BASEMAPS))
def test_背景地図は明示的に選んだときだけ入る(name):
    html = build_viewer_html(TILEJSON, basemap=name)
    assert BASEMAPS[name]["url"] in html
    # 規約の所在を利用者に伝えられる
    assert BASEMAPS[name]["terms"] in (basemap_notice(name) or "")


def test_独自の背景地図を指定できる():
    html = build_viewer_html(
        TILEJSON, basemap_url="https://example.org/{z}/{x}/{y}.png", basemap_attribution="© 例"
    )
    assert "https://example.org/{z}/{x}/{y}.png" in html
    assert "© 例" in html


def test_未知の背景地図はエラー():
    with pytest.raises(ValueError, match="basemap"):
        build_viewer_html(TILEJSON, basemap="google")


def test_bounds_が無ければエラー():
    doc = dict(TILEJSON)
    doc.pop("bounds")
    with pytest.raises(ValueError, match="bounds"):
        build_viewer_html(doc)


def test_設定の埋め込みでスクリプトを閉じさせない():
    """タイルセット名などに `</script>` が入っても HTML を壊さない。"""
    doc = dict(TILEJSON)
    doc["name"] = "</script><script>alert(1)</script>"
    html = build_viewer_html(doc)
    assert "</script><script>alert(1)" not in html
    assert "<\\/script>" in html


def test_外部資源は_SRI_付きで読み込む():
    html = build_viewer_html(TILEJSON)
    assert html.count('integrity="sha384-') == 2
    assert 'crossorigin="anonymous"' in html


def test_パレット型では凡例を描く():
    legend = Legend.from_dict({"title": "区分", "items": [{"r": 1, "g": 2, "b": 3, "title": "あ"}]})
    doc = build_tilejson(
        tiles=["./{z}/{x}/{y}.webp"],
        bounds=(139.0, 35.0, 140.0, 36.0),
        minzoom=8,
        maxzoom=12,
        tile_size=512,
        fmt="webp",
        datapng=PaletteMode(tile_size=512, legend=legend).datapng(),
    )
    html = build_viewer_html(doc)
    assert "renderLegend" in html
    assert "区分" in html


def test_仕様の復号式が埋め込まれている():
    """ビューワーは「絵として出ている」だけでなく値を復号して見せる。"""
    html = build_viewer_html(TILEJSON)
    assert "r < 128 ? r : r - 256" in html
    assert "-10000 + (r * 65536 + g * 256 + b) * 0.1" in html  # mapbox 互換
    assert "(r * 256 + g + b / 256) - 32768" in html  # terrarium 互換


def test_ファイルへ書き出せる(tmp_path):
    path = write_viewer_html(build_viewer_html(TILEJSON), tmp_path / "out" / "index.html")
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_凡例の説明文もエスケープする():
    """legend は外部 URL から取得することがあり、内容は必ずしもページ作者が書いたものではない。"""
    html = build_viewer_html(TILEJSON)
    # title と description の両方が escapeHtml を通っている
    assert "escapeHtml(item.title)" in html
    assert "escapeHtml(item.description)" in html
