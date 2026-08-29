"""タイル生成エンジン（種別に依存しないスキャフォールド）。

ベースタイル（最大ズーム）→ オーバービュータイル（低ズーム）の流れ、プロセス並列、
レジューム、進捗のログを担う。符号化と入出力は `modes` に委ねる。

**プロセス並列は spawn でも動くように書く。** fork（Linux の既定）ならグローバル変数が
子へ引き継がれるが、macOS/Windows の既定である spawn では引き継がれない。ワーカが必要と
する状態は `initargs` で明示的に渡し、`TileMode` は pickle 可能な不変オブジェクトにしてある。

**VRT はバッチごとに開いて閉じる。** GDAL は VRT のソースを一度読むと、データセットが
生きている限りその状態を保持する。ソース数の多い VRT を開きっぱなしにすると、ワーカの
常駐メモリが触れたソース数に比例して積み上がり、並列数を上げたときにメモリを使い切る。
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import rasterio

from datapng_tiler.fileio import is_complete, tile_path
from datapng_tiler.geo import (
    auto_max_zoom,
    auto_min_zoom,
    clip_to_mercator,
    compute_warped_vrt_params,
    source_bounds_wgs84,
    source_resolution_m,
    tile_bounds_lnglat,
    tiles_in_bounds,
)
from datapng_tiler.modes.base import TileMode

logger = logging.getLogger(__name__)

# 1 つの future に載せるタイル数の上限。大きすぎると進捗が粗くなり、小さすぎると
# プロセス間通信の回数が増える。
_MAX_BATCH = 256
# ワーカ 1 個あたりの目標バッチ数。バッチ数がプロセス数を十分に上回らないと、
# 早く終わったワーカが遊んで並列数どおりの速度が出ない。
_BATCHES_PER_WORKER = 8

# ワーカ 1 個あたりの GDAL ブロックキャッシュ（MB）。GDAL の既定は「物理メモリの 5%」で、
# 並列数を上げるとその倍数だけ確保されてメモリを使い切る。タイル 1 枚の読み出しに大きな
# キャッシュは要らないので小さく固定する（既に設定済みなら尊重する）。
_WORKER_GDAL_CACHEMAX_MB = "256"


def batch_size(count: int, processes: int) -> int:
    """タイル群をワーカへ均等に配るバッチサイズを決める。"""
    if processes <= 1 or count <= 0:
        return _MAX_BATCH
    return max(1, min(_MAX_BATCH, math.ceil(count / (processes * _BATCHES_PER_WORKER))))


def init_worker() -> None:
    """ワーカプロセスの共通初期化（GDAL のキャッシュ上限とログ抑制）。

    GDAL は最初にデータセットを開いたときに `GDAL_CACHEMAX` を読むため、
    データセットを開く前に設定する必要がある。
    """
    os.environ.setdefault("GDAL_CACHEMAX", _WORKER_GDAL_CACHEMAX_MB)
    os.environ.setdefault("PROJ_LOG_LEVEL", "0")
    logging.getLogger("rasterio").setLevel(logging.ERROR)


# --- ベースタイル -------------------------------------------------------------------

_base_mode: TileMode | None = None
_base_src: str | None = None
_base_zoom: int | None = None
_base_overwrite: bool = False


def _init_base_worker(src_path: str, zoom: int, mode: TileMode, overwrite: bool) -> None:
    """ベースタイル用ワーカの初期化。状態はすべて引数で受け取る（spawn 対応）。"""
    global _base_mode, _base_src, _base_zoom, _base_overwrite
    init_worker()
    _base_mode, _base_src, _base_zoom, _base_overwrite = mode, src_path, zoom, overwrite


def _render_base_batch(output_dir: str, tiles: list[tuple[int, int, int]]) -> int:
    """ベースタイルをまとめて生成する（ワーカ側）。"""
    assert _base_mode is not None and _base_src is not None and _base_zoom is not None
    mode = _base_mode
    written = 0
    with rasterio.open(_base_src) as src:
        params = compute_warped_vrt_params(src, _base_zoom, mode.tile_size, mode.topleft)
        if params is None:
            return 0
        with mode.make_warped_vrt(src, params) as vrt:
            for zoom, x, y in tiles:
                written += mode.render_base_tile(
                    vrt, params, Path(output_dir), zoom, x, y, overwrite=_base_overwrite
                )
    return written


def generate_base_tiles(
    src_path: Path | str,
    output_dir: Path | str,
    mode: TileMode,
    *,
    zoom: int,
    bounds: tuple[float, float, float, float],
    processes: int = 1,
    overwrite: bool = False,
) -> int:
    """指定ズームのベースタイルを生成し、書き出した枚数を返す。"""
    output_dir = Path(output_dir)
    tiles = tiles_in_bounds(bounds, zoom)
    logger.info("ベースタイル z%d: %d 枚", zoom, len(tiles))

    if processes <= 1:
        _init_base_worker(str(src_path), zoom, mode, overwrite)
        return _render_base_batch(str(output_dir), tiles)

    size = batch_size(len(tiles), processes)
    batches = [tiles[i : i + size] for i in range(0, len(tiles), size)]
    written = 0
    with ProcessPoolExecutor(
        max_workers=processes,
        initializer=_init_base_worker,
        initargs=(str(src_path), zoom, mode, overwrite),
    ) as executor:
        futures = [executor.submit(_render_base_batch, str(output_dir), b) for b in batches]
        for future in as_completed(futures):
            written += future.result()
    return written


# --- オーバービュータイル ------------------------------------------------------------


def _render_overview_batch(
    mode: TileMode, output_dir: str, zoom: int, tiles: list[tuple[int, int]], overwrite: bool
) -> int:
    """オーバービュータイルをまとめて生成する（ワーカ側）。"""
    written = 0
    for x, y in tiles:
        written += mode.render_overview_tile(Path(output_dir), zoom, x, y, overwrite=overwrite)
    return written


def _existing_coords(zoom_dir: Path, extension: str) -> set[tuple[int, int]]:
    """あるズームのディレクトリに実在するタイル座標を集める。"""
    coords: set[tuple[int, int]] = set()
    if not zoom_dir.is_dir():
        return coords
    for x_dir in zoom_dir.iterdir():
        if not x_dir.is_dir() or not x_dir.name.isdigit():
            continue
        x = int(x_dir.name)
        for tile in x_dir.iterdir():
            if tile.suffix == extension and tile.stem.isdigit():
                coords.add((x, int(tile.stem)))
    return coords


def generate_overview_tiles(
    output_dir: Path | str,
    mode: TileMode,
    *,
    max_zoom: int,
    min_zoom: int,
    processes: int = 1,
    overwrite: bool = False,
) -> int:
    """ベースタイルから低ズームのタイルを積み上げ、書き出した枚数を返す。

    ズーム間には依存があるので逐次、同じズームの中は並列に処理する。
    """
    output_dir = Path(output_dir)
    written = 0
    zooms = range(max_zoom - 1, min_zoom - 1, -1)

    if processes <= 1:
        init_worker()
        for zoom in zooms:
            parents = _parents_for(output_dir, zoom, mode)
            if not parents:
                continue
            logger.info("オーバービュー z%d: %d 枚", zoom, len(parents))
            written += _render_overview_batch(mode, str(output_dir), zoom, parents, overwrite)
        return written

    # プールはズームをまたいで 1 つだけ作り、起動コストを払い直さない。
    with ProcessPoolExecutor(max_workers=processes, initializer=init_worker) as executor:
        for zoom in zooms:
            parents = _parents_for(output_dir, zoom, mode)
            if not parents:
                continue
            logger.info("オーバービュー z%d: %d 枚", zoom, len(parents))
            size = batch_size(len(parents), processes)
            batches = [parents[i : i + size] for i in range(0, len(parents), size)]
            futures = [
                executor.submit(
                    _render_overview_batch, mode, str(output_dir), zoom, batch, overwrite
                )
                for batch in batches
            ]
            for future in as_completed(futures):
                written += future.result()
    return written


def _parents_for(output_dir: Path, zoom: int, mode: TileMode) -> list[tuple[int, int]]:
    """1 段下のズームに実在するタイルから、作るべき親タイル座標を決定的な順序で返す。"""
    children = _existing_coords(output_dir / str(zoom + 1), mode.fmt.extension)
    return sorted({(x // 2, y // 2) for x, y in children})


# --- まとめて実行 -------------------------------------------------------------------


@dataclass(frozen=True)
class TilingResult:
    """タイル化の結果（TileJSON の生成に必要な情報を含む）。"""

    bounds: tuple[float, float, float, float]
    min_zoom: int
    max_zoom: int
    base_tiles: int
    overview_tiles: int

    @property
    def total_tiles(self) -> int:
        return self.base_tiles + self.overview_tiles


def tile_raster(
    src_path: Path | str,
    output_dir: Path | str,
    mode: TileMode,
    *,
    max_zoom: int | None = None,
    min_zoom: int | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    processes: int = 1,
    overwrite: bool = False,
) -> TilingResult:
    """ラスタを XYZ タイル木へ変換する（ベース → オーバービュー）。

    Args:
        max_zoom: 最大ズーム。``None`` ならソース解像度から決める
        min_zoom: 最小ズーム。``None`` ならデータ全体が 1 タイルに収まるズーム
        bounds: 生成範囲 WGS84。``None`` ならソース範囲
        processes: 並列プロセス数
        overwrite: 既存タイルも作り直す（入力を更新したときに必要）
    """
    src_path = Path(src_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        src_bounds = source_bounds_wgs84(src)
        resolution = source_resolution_m(src)

    target = src_bounds if bounds is None else _intersect(bounds, src_bounds)
    if target is None:
        raise ValueError(
            f"指定範囲 {bounds} がソース範囲 {src_bounds} と交差しません"
            "（生成するタイルがありません）"
        )
    target, clipped = clip_to_mercator(target)
    if clipped:
        logger.warning(
            "Web メルカトルの限界緯度（±85.0511 度）を超える範囲を切り詰めました: %s", target
        )

    z_max = max_zoom if max_zoom is not None else auto_max_zoom(resolution, target, mode.tile_size)
    z_min = min_zoom if min_zoom is not None else auto_min_zoom(target, z_max)
    if z_min > z_max:
        raise ValueError(f"min_zoom({z_min}) が max_zoom({z_max}) を超えています")

    logger.info(
        "タイル化 [%s]: %s → %s (z%d-%d)", mode.key, src_path.name, output_dir, z_min, z_max
    )

    base = generate_base_tiles(
        src_path,
        output_dir,
        mode,
        zoom=z_max,
        bounds=target,
        processes=processes,
        overwrite=overwrite,
    )
    logger.info("ベースタイル: %d 枚", base)

    overview = generate_overview_tiles(
        output_dir,
        mode,
        max_zoom=z_max,
        min_zoom=z_min,
        processes=processes,
        overwrite=overwrite,
    )
    logger.info("オーバービュータイル: %d 枚 / 合計 %d 枚", overview, base + overview)

    return TilingResult(
        bounds=target,
        min_zoom=z_min,
        max_zoom=z_max,
        base_tiles=base,
        overview_tiles=overview,
    )


def _intersect(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float] | None:
    west, south = max(a[0], b[0]), max(a[1], b[1])
    east, north = min(a[2], b[2]), min(a[3], b[3])
    if west >= east or south >= north:
        return None
    return west, south, east, north


# --- タイル木の走査 -----------------------------------------------------------------


@dataclass(frozen=True)
class TreeSummary:
    """既存タイル木から読み取った実測値。"""

    bounds: tuple[float, float, float, float]
    min_zoom: int
    max_zoom: int
    tile_count: int
    extension: str


def scan_tree(root: Path | str, extension: str | None = None) -> TreeSummary | None:
    """タイル木を走査し、範囲・ズーム範囲・枚数を実測する（タイルが無ければ ``None``）。

    TileJSON の `bounds` / `minzoom` / `maxzoom` は、引数の申告ではなく**実際に置かれた
    タイル**から作る。両者が食い違うと、クライアントが空のタイルを取りに行ったり、
    データがあるのに表示されなかったりする。
    """
    root = Path(root)
    zooms: dict[int, set[tuple[int, int]]] = {}
    for zoom_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        zoom = int(zoom_dir.name)
        found = _existing_coords(zoom_dir, extension) if extension else _any_coords(zoom_dir)
        if found:
            zooms[zoom] = found
    if not zooms:
        return None

    max_zoom = max(zooms)
    min_zoom = min(zooms)
    # 範囲は最大ズームのタイルから求める（低ズームのタイル 1 枚は広い範囲を覆うため、
    # 全ズームの和を取ると最も粗いタイルに引きずられて実際より広い範囲になる）。
    west = south = float("inf")
    east = north = float("-inf")
    for x, y in zooms[max_zoom]:
        w, s, e, n = tile_bounds_lnglat(x, y, max_zoom)
        west, south, east, north = min(west, w), min(south, s), max(east, e), max(north, n)

    if extension is None:
        extension = _detect_extension(root, max_zoom, zooms[max_zoom])

    return TreeSummary(
        bounds=(west, south, east, north),
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        tile_count=sum(len(v) for v in zooms.values()),
        extension=extension,
    )


def _any_coords(zoom_dir: Path) -> set[tuple[int, int]]:
    """拡張子を問わずタイル座標を集める（既存タイル木の取り込み用）。"""
    coords: set[tuple[int, int]] = set()
    for x_dir in zoom_dir.iterdir():
        if not x_dir.is_dir() or not x_dir.name.isdigit():
            continue
        x = int(x_dir.name)
        for tile in x_dir.iterdir():
            if tile.is_file() and tile.stem.isdigit() and tile.suffix != ".tmp":
                coords.add((x, int(tile.stem)))
    return coords


def _detect_extension(root: Path, zoom: int, coords: set[tuple[int, int]]) -> str:
    for x, y in sorted(coords):
        x_dir = root / str(zoom) / str(x)
        for tile in sorted(x_dir.glob(f"{y}.*")):
            if is_complete(tile):
                return tile.suffix
    return ""


def tile_exists(root: Path | str, zoom: int, x: int, y: int, extension: str) -> bool:
    return is_complete(tile_path(root, zoom, x, y, extension))
