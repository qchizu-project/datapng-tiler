"""XYZ タイルの幾何と再投影パラメータ。

座標順序は一貫して (lng, lat)。bounds は (west, south, east, north)。

タイル化は「ソースを EPSG:3857 の `WarpedVRT` に載せ、その中からタイル 1 枚ぶんの
ウィンドウを読む」形で行う。VRT の格子は**全球のタイル画素格子に整列**させてあるので、
どのタイルもウィンドウの切り出しだけで取り出せ、タイルごとの再投影は起きない。

**アライメントは 2 通りある**（仕様 §3.4 の support に対応する）:

- ``topleft=True``（point support / anchor=northwest）: 画素値はその画素の**左上の節点**を
  代表する。VRT の原点を半画素だけ外側へずらし、リサンプリングの標本点を節点に一致させる。
- ``topleft=False``（block support / point support with anchor=center）: 画素値は画素の
  **中心**に対応する。GDAL の標準的な画素配置。

半画素をずらす向きを間違えると全タイルが半画素ずれるため、その計算はこのモジュールに
閉じ込め、`tests/test_geo.py` の契約テストで固定する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

WEB_MERCATOR = CRS.from_epsg(3857)
WGS84 = CRS.from_epsg(4326)

# Web メルカトルの原点シフト（地球半周, m）。約 20037508.342789244
ORIGIN_SHIFT = 2 * math.pi * 6378137 / 2.0
# Web メルカトルで表現できる限界緯度。約 85.0511287798066
MAX_LATITUDE = math.degrees(math.atan(math.sinh(math.pi)))

# 赤道上の 256px タイル 1 画素の地上解像度 [m/px]（z=0）。
_BASE_RESOLUTION = 2 * ORIGIN_SHIFT / 256

DEFAULT_TILE_SIZE = 512


class GeometryError(ValueError):
    """タイル化できない入力（CRS 未定義・経度 180 度またぎなど）。"""


# --- タイル座標 ---------------------------------------------------------------------


def _x_fraction(lng: float, n: int) -> float:
    """経度を全球タイル格子上の連続座標（0 〜 n）へ写す。"""
    return (lng + 180.0) / 360.0 * n


def _y_fraction(lat: float, n: int) -> float:
    """緯度を全球タイル格子上の連続座標（0 〜 n）へ写す（北が 0）。"""
    lat = max(-MAX_LATITUDE, min(MAX_LATITUDE, lat))
    lat_rad = math.radians(lat)
    return (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n


def lng_lat_to_tile(lng: float, lat: float, zoom: int) -> tuple[int, int]:
    """経緯度を含む XYZ タイル座標 (x, y) を返す（全球格子の範囲にクランプする）。"""
    n = 2**zoom
    x = math.floor(_x_fraction(lng, n))
    y = math.floor(_y_fraction(lat, n))
    return max(0, min(x, n - 1)), max(0, min(y, n - 1))


def tile_bounds_lnglat(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    """XYZ タイルの経緯度範囲 (west, south, east, north) を返す。"""
    n = 2**zoom
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


def get_tile_range(
    bounds: tuple[float, float, float, float], zoom: int
) -> tuple[int, int, int, int]:
    """経緯度範囲を覆うタイル座標範囲 (x_min, y_min, x_max, y_max) を返す（両端含む）。

    範囲は**半開区間**として扱う（東端・南端の境界ちょうどは次のタイルに含めない）。
    こうしないと、タイル境界にぴったり合う範囲（あるタイルの bounds をそのまま渡した等）で
    東と南に 1 列ぶん余分なタイルが並び、区画を並べたときに重複する。
    """
    west, south, east, north = bounds
    n = 2**zoom
    x_min = max(0, min(math.floor(_x_fraction(west, n)), n - 1))
    y_min = max(0, min(math.floor(_y_fraction(north, n)), n - 1))
    x_max = max(x_min, min(math.ceil(_x_fraction(east, n)) - 1, n - 1))
    y_max = max(y_min, min(math.ceil(_y_fraction(south, n)) - 1, n - 1))
    return x_min, y_min, x_max, y_max


def tiles_in_bounds(
    bounds: tuple[float, float, float, float], zoom: int
) -> list[tuple[int, int, int]]:
    """範囲を覆うタイル座標 (z, x, y) を決定的な順序で列挙する。"""
    x_min, y_min, x_max, y_max = get_tile_range(bounds, zoom)
    return [(zoom, x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)]


def tile_resolution(zoom: int, tile_size: int) -> float:
    """指定ズーム・タイル画素数における EPSG:3857 上の画素間隔 [m/px]。"""
    return 2.0 * ORIGIN_SHIFT / (2**zoom * tile_size)


# --- ソースの範囲・解像度 ------------------------------------------------------------


def source_bounds_wgs84(src: rasterio.DatasetReader) -> tuple[float, float, float, float]:
    """ソースの範囲を WGS84 (west, south, east, north) で返す。

    Raises:
        GeometryError: CRS が未定義、または範囲が経度 180 度をまたぐ場合。
    """
    if src.crs is None:
        raise GeometryError(
            "ソースの CRS が未定義です。--src-crs で明示してください"
            "（誤った CRS を推測すると、タイルが黙って別の場所に置かれます）"
        )
    west, south, east, north = transform_bounds(src.crs, WGS84, *src.bounds)
    if west > east:
        raise GeometryError(
            f"ソースの範囲が経度 180 度をまたいでいます（west={west}, east={east}）。"
            "東西で分割してからタイル化してください"
        )
    return west, south, east, north


def clip_to_mercator(
    bounds: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], bool]:
    """緯度を Web メルカトルの限界（±85.0511…度）へ収める。

    Returns:
        (収めた範囲, クリップしたか)。クリップした場合、呼び出し側は警告すべき
        （黙って切り詰めると「なぜか極域のタイルが無い」という形で後から発覚する）。
    """
    west, south, east, north = bounds
    clipped_south = max(south, -MAX_LATITUDE)
    clipped_north = min(north, MAX_LATITUDE)
    was_clipped = clipped_south != south or clipped_north != north
    return (west, clipped_south, east, clipped_north), was_clipped


def source_resolution_m(src: rasterio.DatasetReader) -> float:
    """ソースの地上解像度 [m/px] を CRS の単位に依存せず推定する。

    ソース範囲を Web メルカトルへ変換して画素数で割る。地理座標系（度）でも
    投影座標系（メートル）でも同じように扱える。
    """
    merc = transform_bounds(src.crs, WEB_MERCATOR, *src.bounds)
    return min((merc[2] - merc[0]) / src.width, (merc[3] - merc[1]) / src.height)


def auto_max_zoom(res_m: float, bounds: tuple[float, float, float, float], tile_size: int) -> int:
    """ソース解像度と同等以上に細かくなる最小のズームを返す。

    「これ以上ズームを上げてもソースに情報が無い」ところで止める。低ズーム側から探索し、
    タイル解像度がソース解像度以下になった最初のズームを採る（高ズーム側から探すと、
    粗いソースでも最大ズームを返してしまう）。
    """
    lat_mid = (bounds[1] + bounds[3]) / 2.0
    scale = math.cos(math.radians(max(-MAX_LATITUDE, min(MAX_LATITUDE, lat_mid))))
    for zoom in range(0, 25):
        if _BASE_RESOLUTION * scale / (2**zoom * tile_size / 256) <= res_m:
            return zoom
    return 24


def auto_min_zoom(bounds: tuple[float, float, float, float], max_zoom: int) -> int:
    """データ全体がおおむね 1 タイルに収まる最小ズームを返す（0 〜 max_zoom）。"""
    span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    if span <= 0:
        return max_zoom
    for zoom in range(max_zoom, -1, -1):
        if 360.0 / (2**zoom) >= span:
            return min(zoom, max_zoom)
    return 0


# --- WarpedVRT パラメータ ------------------------------------------------------------


@dataclass(frozen=True)
class WarpedVrtParams:
    """`WarpedVRT` の作成に必要なパラメータ（ソース範囲とズームだけで決まる）。

    全タイルで 1 つの VRT を共有できる。`vrt_origin_col` / `vrt_origin_row` は VRT の
    原点が全球タイル画素格子の何画素目にあるかで、タイル座標からウィンドウを求めるのに使う。
    """

    transform: object  # rasterio.transform.Affine
    width: int
    height: int
    vrt_origin_col: int
    vrt_origin_row: int


def compute_warped_vrt_params(
    src: rasterio.DatasetReader,
    zoom: int,
    tile_size: int,
    topleft: bool,
) -> WarpedVrtParams | None:
    """ソースとズームから `WarpedVRT` のパラメータを求める。

    ソース範囲を全球タイル画素格子に整列させる（gdalwarp の targetAlignedPixels 相当）。
    `topleft=True` では原点を半画素だけ外側へずらし、標本点をタイル画素の左上節点に置く。

    Returns:
        パラメータ。ソース範囲が退化していて 1 画素も作れない場合は ``None``。
    """
    res = tile_resolution(zoom, tile_size)
    merc = transform_bounds(src.crs, WEB_MERCATOR, *src.bounds)

    left = math.floor(merc[0] / res) * res
    right = math.ceil(merc[2] / res) * res
    bottom = math.floor(merc[1] / res) * res
    top = math.ceil(merc[3] / res) * res

    width = int(round((right - left) / res))
    height = int(round((top - bottom) / res))
    if width <= 0 or height <= 0:
        return None

    if topleft:
        # 画素中心を節点に一致させるため、原点（画素の外角）を半画素だけ外へ出す。
        transform = from_origin(left - 0.5 * res, top + 0.5 * res, res, res)
    else:
        transform = from_origin(left, top, res, res)

    half_globe_px = 2**zoom * tile_size // 2
    return WarpedVrtParams(
        transform=transform,
        width=width,
        height=height,
        vrt_origin_col=int(round(left / res)) + half_globe_px,
        vrt_origin_row=half_globe_px - int(round(top / res)),
    )


@dataclass(frozen=True)
class TileWindow:
    """タイル 1 枚ぶんの VRT 読み取りウィンドウと、タイル内での貼り付け位置。

    ソースの縁にかかるタイルでは `read_w` / `read_h` がタイルサイズに満たない。その場合は
    (`dst_col`, `dst_row`) の位置に読み取り結果を置き、残りを呼び出し側が無効値で埋める。
    """

    read_col: int
    read_row: int
    read_w: int
    read_h: int
    dst_col: int
    dst_row: int


def tile_window(params: WarpedVrtParams, x: int, y: int, tile_size: int) -> TileWindow | None:
    """タイル座標から VRT 内の読み取りウィンドウを求める（VRT 範囲外なら ``None``）。"""
    col_off = x * tile_size - params.vrt_origin_col
    row_off = y * tile_size - params.vrt_origin_row

    read_col = max(col_off, 0)
    read_row = max(row_off, 0)
    read_col_end = min(col_off + tile_size, params.width)
    read_row_end = min(row_off + tile_size, params.height)
    read_w = read_col_end - read_col
    read_h = read_row_end - read_row
    if read_w <= 0 or read_h <= 0:
        return None

    return TileWindow(
        read_col=read_col,
        read_row=read_row,
        read_w=read_w,
        read_h=read_h,
        dst_col=read_col - col_off,
        dst_row=read_row - row_off,
    )


def tile_sample_x3857(zoom: int, tile_size: int, tile_x: int, col: int, topleft: bool) -> float:
    """タイル (zoom, tile_x) の col 列目の画素が代表する EPSG:3857 の x 座標。

    `compute_warped_vrt_params` が作る格子の**意味**をここに書き下す。タイル化の結果を
    解析的に検証するために使い、テストが両者の一致を固定する（片方だけ直すと壊れる）。
    """
    res = tile_resolution(zoom, tile_size)
    node = -ORIGIN_SHIFT + (tile_x * tile_size + col) * res
    return node if topleft else node + 0.5 * res


def tile_sample_y3857(zoom: int, tile_size: int, tile_y: int, row: int, topleft: bool) -> float:
    """タイル (zoom, tile_y) の row 行目の画素が代表する EPSG:3857 の y 座標。"""
    res = tile_resolution(zoom, tile_size)
    node = ORIGIN_SHIFT - (tile_y * tile_size + row) * res
    return node if topleft else node - 0.5 * res
