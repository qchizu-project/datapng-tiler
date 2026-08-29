"""XYZ タイル幾何と再投影パラメータのテスト。

半画素シフトの向きを間違えると全タイルが半画素ずれるが、絵としては一見正しく見える。
ここで格子の意味（`tile_sample_x3857` / `tile_sample_y3857`）と、実際に作られる
`WarpedVRT` の格子（`compute_warped_vrt_params` + `tile_window`）が一致することを固定する。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import rasterio

from datapng_tiler.geo import (
    MAX_LATITUDE,
    ORIGIN_SHIFT,
    GeometryError,
    auto_max_zoom,
    auto_min_zoom,
    clip_to_mercator,
    compute_warped_vrt_params,
    get_tile_range,
    lng_lat_to_tile,
    source_bounds_wgs84,
    source_resolution_m,
    tile_bounds_lnglat,
    tile_resolution,
    tile_sample_x3857,
    tile_sample_y3857,
    tile_window,
    tiles_in_bounds,
)

# --- タイル座標 ---------------------------------------------------------------------


def test_東京駅のタイル座標():
    """よく知られた値で固定する（z=10 の東京駅は (909, 403)）。"""
    assert lng_lat_to_tile(139.7671, 35.6812, 10) == (909, 403)


def test_ズーム0は常に1タイル():
    assert lng_lat_to_tile(-179.0, 80.0, 0) == (0, 0)
    assert lng_lat_to_tile(179.0, -80.0, 0) == (0, 0)


@pytest.mark.parametrize(("zoom", "x", "y"), [(0, 0, 0), (5, 10, 12), (14, 14552, 6451)])
def test_タイル範囲の中心は元のタイルに戻る(zoom, x, y):
    west, south, east, north = tile_bounds_lnglat(x, y, zoom)
    assert lng_lat_to_tile((west + east) / 2, (south + north) / 2, zoom) == (x, y)


def test_タイル範囲は隣接タイルと隙間なく接する():
    _, _, east, _ = tile_bounds_lnglat(10, 12, 5)
    west_next, _, _, _ = tile_bounds_lnglat(11, 12, 5)
    assert east == pytest.approx(west_next)


def test_範囲を覆うタイルを列挙する():
    bounds = tile_bounds_lnglat(10, 12, 5)
    assert tiles_in_bounds(bounds, 5) == [(5, 10, 12)]

    x_min, y_min, x_max, y_max = get_tile_range((-180.0, -85.0, 180.0, 85.0), 2)
    assert (x_min, y_min, x_max, y_max) == (0, 0, 3, 3)


def test_極域の緯度はメルカトルの限界にクランプされる():
    """クランプしないと log(tan(...)) が発散し、タイル座標が範囲外になる。"""
    assert lng_lat_to_tile(0.0, 90.0, 3) == (4, 0)
    assert lng_lat_to_tile(0.0, -90.0, 3) == (4, 7)


# --- 格子の意味（半画素シフト） ------------------------------------------------------


@pytest.mark.parametrize("topleft", [True, False])
def test_VRT格子とタイル画素の対応が一致する(tmp_path, mercator_ramp, topleft):
    """`tile_window` で切り出した画素の VRT 上の標本位置が `tile_sample_*3857` と一致する。

    タイル化の実装（VRT パラメータ）と、その格子の意味を宣言した関数が食い違うと、
    生成タイルは全体が半画素ずれる。両者を突き合わせて固定する。
    """
    tile_size = 256
    zoom = 8
    res = tile_resolution(zoom, tile_size)
    # z=8 のタイル (227, 100) 付近を覆うソースを置く
    x0 = -ORIGIN_SHIFT + 227 * tile_size * res
    y0 = ORIGIN_SHIFT - 100 * tile_size * res
    path = mercator_ramp(
        tmp_path / "ramp.tif", x0=x0, y0=y0, pixel_size=res, width=tile_size, height=tile_size
    )

    with rasterio.open(path) as src:
        params = compute_warped_vrt_params(src, zoom, tile_size, topleft)
    assert params is not None

    window = tile_window(params, 227, 100, tile_size)
    assert window is not None

    for col, row in [(0, 0), (1, 3), (tile_size - 1, tile_size - 1)]:
        vrt_col = window.read_col + (col - window.dst_col)
        vrt_row = window.read_row + (row - window.dst_row)
        # VRT 画素の中心座標（リサンプリングの標本点）
        sample_x, sample_y = rasterio.transform.xy(
            params.transform, vrt_row, vrt_col, offset="center"
        )
        assert sample_x == pytest.approx(
            tile_sample_x3857(zoom, tile_size, 227, col, topleft), abs=1e-6
        )
        assert sample_y == pytest.approx(
            tile_sample_y3857(zoom, tile_size, 100, row, topleft), abs=1e-6
        )


def test_左上法と中心法は半画素だけずれる():
    zoom, tile_size = 10, 512
    res = tile_resolution(zoom, tile_size)
    topleft_x = tile_sample_x3857(zoom, tile_size, 100, 5, topleft=True)
    center_x = tile_sample_x3857(zoom, tile_size, 100, 5, topleft=False)
    assert center_x - topleft_x == pytest.approx(res / 2)

    topleft_y = tile_sample_y3857(zoom, tile_size, 100, 5, topleft=True)
    center_y = tile_sample_y3857(zoom, tile_size, 100, 5, topleft=False)
    assert topleft_y - center_y == pytest.approx(res / 2)


def test_タイル境界の節点は隣接タイルの先頭画素と一致する():
    """左上法では、あるタイルの右端の 1 つ先の節点が次のタイルの先頭画素になる。"""
    zoom, tile_size = 6, 256
    end = tile_sample_x3857(zoom, tile_size, 20, tile_size, topleft=True)
    next_start = tile_sample_x3857(zoom, tile_size, 21, 0, topleft=True)
    assert end == pytest.approx(next_start)


# --- ウィンドウ ---------------------------------------------------------------------


def test_ソース範囲外のタイルはウィンドウを持たない(tmp_path, mercator_ramp):
    tile_size = 256
    zoom = 8
    res = tile_resolution(zoom, tile_size)
    x0 = -ORIGIN_SHIFT + 227 * tile_size * res
    y0 = ORIGIN_SHIFT - 100 * tile_size * res
    path = mercator_ramp(
        tmp_path / "ramp.tif", x0=x0, y0=y0, pixel_size=res, width=tile_size, height=tile_size
    )
    with rasterio.open(path) as src:
        params = compute_warped_vrt_params(src, zoom, tile_size, topleft=True)

    assert tile_window(params, 227, 100, tile_size) is not None
    assert tile_window(params, 300, 100, tile_size) is None
    assert tile_window(params, 227, 200, tile_size) is None


def test_縁のタイルは部分ウィンドウになる(tmp_path, mercator_ramp):
    """ソースがタイルの一部しか覆わない場合、読み取り幅が縮み貼り付け位置が付く。"""
    tile_size = 256
    zoom = 8
    res = tile_resolution(zoom, tile_size)
    # タイル (227, 100) の右半分だけを覆うソース
    x0 = -ORIGIN_SHIFT + (227 * tile_size + tile_size // 2) * res
    y0 = ORIGIN_SHIFT - 100 * tile_size * res
    path = mercator_ramp(
        tmp_path / "ramp.tif", x0=x0, y0=y0, pixel_size=res, width=tile_size // 2, height=tile_size
    )
    with rasterio.open(path) as src:
        params = compute_warped_vrt_params(src, zoom, tile_size, topleft=True)

    window = tile_window(params, 227, 100, tile_size)
    assert window is not None
    assert window.read_w == tile_size // 2
    assert window.dst_col == tile_size // 2
    assert window.dst_row == 0


# --- ソースの検証 -------------------------------------------------------------------


def test_CRS未定義のソースはエラー(tmp_path, write_raster):
    path = write_raster(tmp_path / "nocrs.tif", np.zeros((4, 4), dtype=np.float32), crs=None)
    with rasterio.open(path) as src:
        with pytest.raises(GeometryError, match="CRS"):
            source_bounds_wgs84(src)


def test_メルカトル限界を超える緯度はクリップされる():
    bounds, clipped = clip_to_mercator((0.0, -89.0, 1.0, 89.0))
    assert clipped is True
    assert bounds[1] == pytest.approx(-MAX_LATITUDE)
    assert bounds[3] == pytest.approx(MAX_LATITUDE)

    bounds, clipped = clip_to_mercator((0.0, -80.0, 1.0, 80.0))
    assert clipped is False


# --- ズームの自動決定 ---------------------------------------------------------------


def test_自動最大ズームはソース解像度で頭打ちになる():
    coarse = auto_max_zoom(100.0, tile_size=256)
    fine = auto_max_zoom(1.0, tile_size=256)
    assert coarse < fine
    # 512px タイルは同じ解像度を 1 段低いズームで達成する
    assert auto_max_zoom(1.0, tile_size=512) == fine - 1


def test_自動最大ズームはタイル解像度と同じ土俵で比べる():
    """ソース解像度もタイル解像度も Web メルカトル上の量。緯度補正を掛けてはいけない。

    片方にだけ cosφ を掛けると、高緯度でソース解像度の半分以下しか使わないタイルになる。
    """
    for zoom in (8, 12, 16):
        for tile_size in (256, 512):
            resolution = tile_resolution(zoom, tile_size)
            # ちょうど z のタイル解像度に等しいソースは、その z で頭打ちになる
            assert auto_max_zoom(resolution, tile_size) == zoom
            # 2 倍未満に粗いだけなら、まだ z が要る（z-1 は 2 倍粗い）
            assert auto_max_zoom(resolution * 1.99, tile_size) == zoom
            # ちょうど 2 倍粗ければ 1 段浅くなる
            assert auto_max_zoom(resolution * 2.0, tile_size) == zoom - 1


def test_高緯度でもズームが浅くならない(tmp_path):
    """緯度 60 度・地上 10m 等方のソース。メルカトル上では約 20m/px なので z12 が正しい。

    cosφ を片側にだけ掛けていた頃はここで z11 を返し、ソース解像度の半分しか使わない
    タイルセットになっていた。
    """
    from rasterio.transform import from_origin

    size = 256
    lat_step = 10.0 / 111320.0
    # 地上で等方にするため、経度方向の刻みは 1/cosφ 倍にする
    lon_step = lat_step / math.cos(math.radians(60.0))
    path = tmp_path / "n60.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(10.0, 60.0, lon_step, lat_step),
    ) as dst:
        dst.write(np.zeros((size, size), dtype=np.float32), 1)

    with rasterio.open(path) as src:
        resolution = source_resolution_m(src)
    assert resolution == pytest.approx(20.0, rel=0.05)
    assert auto_max_zoom(resolution, tile_size=512) == 12
    assert tile_resolution(12, 512) < resolution < tile_resolution(11, 512)


def test_自動最小ズームはデータ全体が収まるズーム():
    # 全球なら z0
    assert auto_min_zoom((-180.0, -85.0, 180.0, 85.0), 10) == 0
    # 1 度四方なら 360/2^z >= 1 を満たす最大の z = 8
    assert auto_min_zoom((139.0, 35.0, 140.0, 36.0), 12) == 8
    # max_zoom を超えない
    assert auto_min_zoom((139.0, 35.0, 139.001, 35.001), 5) == 5


def test_メルカトル限界緯度が定数と一致する():
    assert MAX_LATITUDE == pytest.approx(85.0511287798066)
    assert ORIGIN_SHIFT == pytest.approx(20037508.342789244)
    assert math.isclose(tile_resolution(0, 256), 2 * ORIGIN_SHIFT / 256)
