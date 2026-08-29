"""既存タイル木の再エンコード。

Mapbox Terrain-RGB や Mapzen/Terrarium で配信されている既存のタイル資産を、正式な
データPNG エンコードへ移し替える（逆向き・形式変換も同じ経路）。タイルは**すでに
目的の格子に載っている**ので、再投影は行わず 1 枚ずつ値を読み替えるだけ。したがって
リサンプリングによる値の劣化は起きない（量子化の分解能が変わるぶんだけが差になる）。

無効値の扱いは仕様 §3.2.2 に従う。入力タイルがアルファチャンネルを持てばアルファ 0 が
無効、持たなければ `invalid_color` に一致する色が無効。出力側の表し方は `NumericalMode`
の設定（アルファ or 無効色）に従う。
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from datapng_tiler.codec import NumericalEncoding
from datapng_tiler.engine import TreeSummary, batch_size, init_worker, scan_tree
from datapng_tiler.fileio import is_complete, tile_path
from datapng_tiler.imageio import load_tile
from datapng_tiler.modes.numerical import NumericalMode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceTiles:
    """入力タイル木の読み方。

    Args:
        encoding: 入力タイルの符号化方式（`specialEncoding="mapbox"` など）
        invalid_color: アルファを持たない入力タイルでの無効色。``None`` なら、
            アルファを持たないタイルの画素はすべて有効とみなす
    """

    encoding: NumericalEncoding = field(default_factory=NumericalEncoding)
    invalid_color: tuple[int, int, int] | None = None

    def read(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """タイルを読み、(値 float64, 有効マスク) を返す。"""
        rgb, alpha = load_tile(path)
        values = self.encoding.decode(rgb)
        if alpha is not None:
            valid = alpha > 0
        elif self.invalid_color is not None:
            valid = ~np.all(rgb == np.array(self.invalid_color, dtype=np.uint8), axis=-1)
        else:
            valid = np.ones(values.shape, dtype=bool)
        return values, valid


@dataclass(frozen=True)
class ConvertResult:
    """再エンコードの結果。"""

    summary: TreeSummary
    written: int
    skipped: int


# --- ワーカ -------------------------------------------------------------------------

_source: SourceTiles | None = None
_mode: NumericalMode | None = None
_overwrite: bool = False


def _init_convert_worker(source: SourceTiles, mode: NumericalMode, overwrite: bool) -> None:
    global _source, _mode, _overwrite
    init_worker()
    _source, _mode, _overwrite = source, mode, overwrite


def _convert_batch(
    src_root: str, dst_root: str, extension: str, tiles: list[tuple[int, int, int]]
) -> tuple[int, int]:
    assert _source is not None and _mode is not None
    written = skipped = 0
    for zoom, x, y in tiles:
        out_path = tile_path(dst_root, zoom, x, y, _mode.fmt.extension)
        if not _overwrite and is_complete(out_path):
            skipped += 1
            continue

        values, valid = _source.read(tile_path(src_root, zoom, x, y, extension))
        raw, valid = _mode.encoding.encode_raw(values, valid=valid)
        if not valid.any():
            skipped += 1
            continue
        _mode.save_image(_mode.compose(raw, valid), out_path)
        written += 1
    return written, skipped


# --- 公開 API -----------------------------------------------------------------------


def convert_tree(
    src_root: Path | str,
    dst_root: Path | str,
    source: SourceTiles,
    mode: NumericalMode,
    *,
    processes: int = 1,
    overwrite: bool = False,
) -> ConvertResult:
    """タイル木を丸ごと再エンコードする。

    Raises:
        ValueError: 入力にタイルが 1 枚も無い、またはタイルサイズが宣言と食い違う場合。
    """
    src_root, dst_root = Path(src_root), Path(dst_root)
    summary = scan_tree(src_root)
    if summary is None:
        raise ValueError(f"入力にタイルが 1 枚もありません: {src_root}")

    tiles = _list_tiles(src_root, summary.extension)
    first_rgb, _ = load_tile(tile_path(src_root, *tiles[0], summary.extension))
    if first_rgb.shape[0] != mode.tile_size or first_rgb.shape[1] != mode.tile_size:
        raise ValueError(
            f"入力タイルは {first_rgb.shape[1]}×{first_rgb.shape[0]} ですが、"
            f"--tile-size は {mode.tile_size} です。実体に合わせてください"
        )

    dst_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "再エンコード: %s → %s（%d 枚, z%d-%d）",
        src_root,
        dst_root,
        len(tiles),
        summary.min_zoom,
        summary.max_zoom,
    )

    if processes <= 1:
        _init_convert_worker(source, mode, overwrite)
        written, skipped = _convert_batch(str(src_root), str(dst_root), summary.extension, tiles)
    else:
        size = batch_size(len(tiles), processes)
        batches = [tiles[i : i + size] for i in range(0, len(tiles), size)]
        written = skipped = 0
        with ProcessPoolExecutor(
            max_workers=processes,
            initializer=_init_convert_worker,
            initargs=(source, mode, overwrite),
        ) as executor:
            futures = [
                executor.submit(
                    _convert_batch, str(src_root), str(dst_root), summary.extension, batch
                )
                for batch in batches
            ]
            for future in as_completed(futures):
                w, s = future.result()
                written += w
                skipped += s

    logger.info("再エンコード完了: %d 枚（スキップ %d）", written, skipped)
    return ConvertResult(summary=summary, written=written, skipped=skipped)


def _list_tiles(root: Path, extension: str) -> list[tuple[int, int, int]]:
    """タイル木の (z, x, y) を決定的な順序で列挙する。"""
    tiles: list[tuple[int, int, int]] = []
    for zoom_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        zoom = int(zoom_dir.name)
        for x_dir in sorted(p for p in zoom_dir.iterdir() if p.is_dir() and p.name.isdigit()):
            x = int(x_dir.name)
            for tile in sorted(x_dir.glob(f"*{extension}")):
                if tile.stem.isdigit() and is_complete(tile):
                    tiles.append((zoom, x, int(tile.stem)))
    return sorted(tiles)
