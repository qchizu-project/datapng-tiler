"""TileJSON 3.0.0 + DataPNG 拡張の生成。

`bounds` / `minzoom` / `maxzoom` は、引数の申告ではなく**実際に置かれたタイル**から作る
（`from_tree`）。両者が食い違うと、クライアントが存在しないタイルを取りに行ったり、
データがあるのに範囲外として表示されなかったりする。

ルートに載せる拡張フィールドは `tileSize`（仕様 §2.1）だけ。タイル画像の形式は
**タイル URL の拡張子**が担うので、専用フィールドは出さない（仕様 §2.2）。
`datapng` の中身は種別（`TileMode.datapng()`）が組み立てる。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datapng_tiler.engine import TreeSummary, scan_tree
from datapng_tiler.modes.base import TileMode

TILEJSON_VERSION = "3.0.0"

DEFAULT_TILES_URL_TEMPLATE = "./{z}/{x}/{y}"


def default_tiles_url(extension: str) -> str:
    """相対パスのタイル URL テンプレート（タイル木のルートに置いた TileJSON 用）。"""
    return f"{DEFAULT_TILES_URL_TEMPLATE}{extension}"


def build_tilejson(
    *,
    tiles: list[str],
    bounds: tuple[float, float, float, float],
    minzoom: int,
    maxzoom: int,
    tile_size: int,
    datapng: dict[str, Any] | None = None,
    name: str | None = None,
    description: str | None = None,
    attribution: str | None = None,
    version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """TileJSON 3.0.0 の辞書を組み立てる。

    Args:
        tiles: タイル URL テンプレート（例: ``["https://…/{z}/{x}/{y}.webp"]``）
        bounds: (west, south, east, north) WGS84 十進度
        tile_size: タイル一辺の画素数（仕様 §2.1・REQUIRED）
        datapng: `datapng` 拡張オブジェクト
        description: 自由記述。鉛直基準面（測地系）等はここに書く（仕様 §1.1）
        extra: 追加で merge するキー
    """
    if not tiles:
        raise ValueError("tiles は 1 つ以上必要です（TileJSON 3.0.0 の必須フィールド）")

    west, south, east, north = bounds
    doc: dict[str, Any] = {"tilejson": TILEJSON_VERSION, "tiles": list(tiles)}
    if name:
        doc["name"] = name
    if description:
        doc["description"] = description
    if version:
        doc["version"] = version
    if attribution:
        doc["attribution"] = attribution
    doc["bounds"] = [round(v, 7) for v in (west, south, east, north)]
    doc["center"] = [
        round((west + east) / 2, 7),
        round((south + north) / 2, 7),
        minzoom,
    ]
    doc["minzoom"] = minzoom
    doc["maxzoom"] = maxzoom
    doc["tileSize"] = tile_size
    if datapng is not None:
        doc["datapng"] = datapng
    if extra:
        doc.update(extra)
    return doc


def from_tree(
    root: Path | str,
    mode: TileMode,
    *,
    tiles_url: str | None = None,
    name: str | None = None,
    description: str | None = None,
    attribution: str | None = None,
    version: str | None = None,
    extra: dict[str, Any] | None = None,
    summary: TreeSummary | None = None,
) -> dict[str, Any]:
    """既存のタイル木を走査して TileJSON を組み立てる。

    Raises:
        ValueError: タイルが 1 枚も無い場合（範囲もズーム範囲も決められない）。
    """
    if summary is None:
        summary = scan_tree(root, mode.fmt.extension)
    if summary is None:
        raise ValueError(f"タイルが 1 枚もありません: {root}")

    return build_tilejson(
        tiles=[tiles_url or default_tiles_url(mode.fmt.extension)],
        bounds=summary.bounds,
        minzoom=summary.min_zoom,
        maxzoom=summary.max_zoom,
        tile_size=mode.tile_size,
        datapng=mode.datapng(),
        name=name,
        description=description,
        attribution=attribution,
        version=version,
        extra=extra,
    )


def write_tilejson(doc: dict[str, Any], path: Path | str) -> Path:
    """TileJSON を UTF-8 の JSON として書き出す。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def read_tilejson(path: Path | str) -> dict[str, Any]:
    """TileJSON を読む。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extension_from_tiles_url(url: str) -> str | None:
    """タイル URL テンプレートから拡張子（``".webp"`` 等）を取り出す。

    仕様 §2.2 のとおり、タイル画像の形式は URL の拡張子が担う。`{y}.webp` のように
    プレースホルダの直後に付くので、最後の `}` 以降を見る。拡張子が無い URL
    （content negotiation で配信する等）では ``None`` を返す。
    """
    tail = url.rsplit("}", 1)[-1] if "}" in url else url.rsplit("/", 1)[-1]
    tail = tail.split("?", 1)[0].split("#", 1)[0]
    if "." not in tail:
        return None
    extension = "." + tail.rsplit(".", 1)[-1]
    return extension.lower() if len(extension) > 1 else None
