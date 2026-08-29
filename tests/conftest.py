"""テスト共通のフィクスチャ。

外部データやネットワークには依存しない。必要なラスタはその場で合成する。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

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
        a: float = 1e-3,
        b: float = 5e-4,
        c: float = 100.0,
        nodata: float | None = None,
    ) -> Path:
        # 画素中心の座標で値を作る
        cols = x0 + (np.arange(width) + 0.5) * pixel_size
        rows = y0 - (np.arange(height) + 0.5) * pixel_size
        data = (a * cols[np.newaxis, :] + b * rows[:, np.newaxis] + c).astype(np.float32)
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
def ramp_value() -> Callable[[float, float], float]:
    """`mercator_ramp` が表す関数（既定の係数）。"""

    def _value(x: float, y: float, a: float = 1e-3, b: float = 5e-4, c: float = 100.0) -> float:
        return a * x + b * y + c

    return _value


@pytest.fixture
def origin_shift() -> float:
    return ORIGIN_SHIFT
