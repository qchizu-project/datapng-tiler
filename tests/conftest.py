"""テスト共通のフィクスチャ。

外部データやネットワークには依存しない。必要なラスタはその場で合成する。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from datapng_tiler.geo import ORIGIN_SHIFT, WEB_MERCATOR, WGS84


@pytest.fixture
def write_raster() -> Callable[..., Path]:
    """任意の配列を GeoTIFF として書き出すヘルパを返す。"""

    def _write(
        path: Path,
        data: np.ndarray,
        *,
        west: float = 139.0,
        north: float = 36.0,
        pixel_size: float = 0.001,
        crs=WGS84,
        nodata: float | None = None,
        dtype: str | None = None,
    ) -> Path:
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        count, height, width = data.shape
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=dtype or data.dtype.name,
            crs=crs,
            transform=from_origin(west, north, pixel_size, pixel_size),
            nodata=nodata,
        ) as dst:
            dst.write(data)
        return path

    return _write


# `mercator_ramp` が表す 1 次関数 v = A*x + B*y + C の係数。
# EPSG:3857 の座標は 1e7 のオーダーなので、量子化しても 24 ビットに収まる大きさに抑える。
RAMP_A = 1e-4
RAMP_B = 5e-5
RAMP_C = 0.0


def ramp(x, y):
    """`mercator_ramp` が表す関数（配列も可）。"""
    return RAMP_A * x + RAMP_B * y + RAMP_C


@pytest.fixture
def mercator_ramp(write_raster) -> Callable[..., Path]:
    """EPSG:3857 上で「値 = a·x + b·y」となるラスタを書き出すヘルパを返す。

    値が座標の 1 次関数なので、どんな補間カーネルでも標本点の値を解析的に予測できる
    （線形補間は 1 次関数を厳密に再現する）。タイル画素と地理座標の対応を検証するのに使う。
    """

    def _write(
        path: Path,
        *,
        x0: float,
        y0: float,
        pixel_size: float,
        width: int,
        height: int,
        a: float = RAMP_A,
        b: float = RAMP_B,
        c: float = RAMP_C,
        nodata: float | None = None,
        nodata_cols: slice | None = None,
    ) -> Path:
        # 画素中心の座標で値を作る
        cols = x0 + (np.arange(width) + 0.5) * pixel_size
        rows = y0 - (np.arange(height) + 0.5) * pixel_size
        data = (a * cols[np.newaxis, :] + b * rows[:, np.newaxis] + c).astype(np.float32)
        if nodata_cols is not None:
            if nodata is None:
                raise ValueError("nodata_cols を使うには nodata の指定が必要です")
            data[:, nodata_cols] = nodata
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=WEB_MERCATOR,
            transform=from_origin(x0, y0, pixel_size, pixel_size),
            nodata=nodata,
        ) as dst:
            dst.write(data, 1)
        return path

    return _write


@pytest.fixture
def ramp_value() -> Callable[..., Any]:
    """`mercator_ramp` が表す関数（既定の係数）。"""
    return ramp


@pytest.fixture
def origin_shift() -> float:
    return ORIGIN_SHIFT


@pytest.fixture
def ramp_raster(tmp_path, mercator_ramp) -> Path:
    """タイル (ZOOM, TX, TY) を余裕をもって覆う 1 次関数ラスタ。"""
    from tests.helpers import MARGIN, TILE_SIZE, grid_origin

    x0, y0, res = grid_origin(TILE_SIZE, margin=MARGIN)
    return mercator_ramp(
        tmp_path / "ramp.tif",
        x0=x0,
        y0=y0,
        pixel_size=res,
        width=TILE_SIZE + 2 * MARGIN,
        height=TILE_SIZE + 2 * MARGIN,
    )


@pytest.fixture
def holes_raster(tmp_path, mercator_ramp) -> Path:
    """タイル (ZOOM, TX, TY) ちょうどを覆い、左半分が nodata のラスタ。"""
    from tests.helpers import TILE_SIZE, grid_origin

    x0, y0, res = grid_origin(TILE_SIZE)
    return mercator_ramp(
        tmp_path / "holes.tif",
        x0=x0,
        y0=y0,
        pixel_size=res,
        width=TILE_SIZE,
        height=TILE_SIZE,
        nodata=-9999.0,
        nodata_cols=slice(0, TILE_SIZE // 2),
    )
