#!/usr/bin/env python3
"""圧縮設定と並列数の実測（BENCHMARKS.md の元データ）。

外部データに依存せず誰でも再現できるよう、地形らしい空間相関を持つ合成 DEM を使う。
実データではないので**絶対値には意味が無い**——設定どうしの相対比較のために使う。

    uv run python scripts/benchmark.py --size 4096 --max-zoom 14
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from datapng_tiler.codec import NumericalEncoding
from datapng_tiler.engine import tile_raster
from datapng_tiler.imageio import TileFormat
from datapng_tiler.modes import NumericalMode


def synthetic_dem(path: Path, size: int, seed: int = 0) -> Path:
    """地形らしい空間相関を持つ合成 DEM を書く（0〜2000m 程度）。

    白色雑音では圧縮率が現実離れするので、粗い格子から段階的に補間して
    低周波が支配的な（＝実際の地形に近い）分布にする。
    """
    rng = np.random.default_rng(seed)
    field = np.zeros((size, size), dtype=np.float64)
    amplitude = 1.0
    grid = 4
    while grid <= size:
        coarse = rng.random((grid, grid))
        # 最近傍で拡大してから移動平均で滑らかにする（依存を増やさない簡易補間）
        expanded = np.kron(coarse, np.ones((size // grid, size // grid)))
        field += amplitude * expanded[:size, :size]
        amplitude *= 0.5
        grid *= 2
    field -= field.min()
    field *= 2000.0 / field.max()

    # 縁を無効値にして、アルファ付きタイルも混ざるようにする
    data = field.astype(np.float32)
    data[: size // 32, :] = -9999.0
    data[:, : size // 32] = -9999.0

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(139.0, 36.0, 1.0 / size, 1.0 / size),
        nodata=-9999.0,
        compress="deflate",
    ) as dst:
        dst.write(data, 1)
    return path


def tree_size(root: Path) -> tuple[int, int]:
    """(タイル枚数, 合計バイト数)。"""
    files = [p for p in root.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def run_case(
    src: Path, out: Path, mode: NumericalMode, *, max_zoom: int, min_zoom: int, jobs: int
) -> tuple[float, int, int]:
    if out.exists():
        shutil.rmtree(out)
    start = time.perf_counter()
    tile_raster(src, out, mode, max_zoom=max_zoom, min_zoom=min_zoom, processes=jobs)
    elapsed = time.perf_counter() - start
    count, size = tree_size(out)
    return elapsed, count, size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4096, help="合成 DEM の一辺（画素）")
    parser.add_argument("--max-zoom", type=int, default=13)
    parser.add_argument("--min-zoom", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/datapng-benchmark"))
    args = parser.parse_args()

    src = synthetic_dem(args.workdir / "dem.tif", args.size)
    base = {
        "tile_size": 512,
        "encoding": NumericalEncoding(factor=0.01),
        "unit": "m",
    }
    common = {"max_zoom": args.max_zoom, "min_zoom": args.min_zoom, "jobs": args.jobs}

    print(f"合成 DEM {args.size}×{args.size}, z{args.min_zoom}-{args.max_zoom}, -j{args.jobs}\n")

    print("## 形式と圧縮設定\n")
    print("| 設定 | 時間 [s] | タイル数 | 合計 [MB] |")
    print("|------|---------:|--------:|----------:|")
    cases: list[tuple[str, NumericalMode]] = []
    for method in (0, 1, 4, 6):
        cases.append(
            (
                f"WebP lossless method={method}",
                NumericalMode(**base, fmt=TileFormat("webp", webp_method=method)),
            )
        )
    for level in (1, 6, 9):
        cases.append(
            (
                f"PNG compress_level={level}",
                NumericalMode(**base, fmt=TileFormat("png", png_compress_level=level)),
            )
        )
    for label, mode in cases:
        elapsed, count, size = run_case(src, args.workdir / "out", mode, **common)
        print(f"| {label} | {elapsed:.1f} | {count} | {size / 1e6:.1f} |")

    print("\n## 並列数\n")
    print("| -j | 時間 [s] |")
    print("|---:|---------:|")
    for jobs in (1, 2, 4, 8, 16):
        mode = NumericalMode(**base, fmt=TileFormat("webp"))
        elapsed, _, _ = run_case(
            src,
            args.workdir / "out",
            mode,
            max_zoom=args.max_zoom,
            min_zoom=args.min_zoom,
            jobs=jobs,
        )
        print(f"| {jobs} | {elapsed:.1f} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
