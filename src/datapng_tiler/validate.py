"""TileJSON の仕様適合検証と、実タイルとの突合。

2 段階で見る:

1. **スキーマ検証** — `datapng` を仕様が配布する JSON Schema にかける
2. **実体との突合** — 宣言（ズーム範囲・形式・無効値の表し方・凡例の色）が、実際に
   置かれたタイルと合っているかを確かめる

スキーマだけ通っても、宣言と実タイルが食い違っていればクライアントは正しく復号できない。
とくに仕様 §3.2.2 の「アルファチャンネルを持つタイルに `invalidColor` を指定しては
ならない（MUST NOT）」は、TileJSON 単体では検出できない。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from datapng_tiler import SPEC_VERSION
from datapng_tiler.engine import scan_tree
from datapng_tiler.imageio import FORMATS, detect_format, load_tile

DEFAULT_SAMPLE_SIZE = 50


@dataclass(frozen=True)
class Problem:
    """検証で見つかった不整合 1 件。"""

    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.where}: {self.message}"


def load_schema() -> dict[str, Any]:
    """同梱している `datapng` の JSON Schema を読む。"""
    path = resources.files("datapng_tiler.schema") / f"datapng-{SPEC_VERSION}.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def validate_document(doc: dict[str, Any]) -> list[Problem]:
    """TileJSON 文書そのものを検証する（タイルは見ない）。"""
    problems: list[Problem] = []

    if doc.get("tilejson") != "3.0.0":
        problems.append(Problem("tilejson", f"3.0.0 であるべきです: {doc.get('tilejson')!r}"))

    tiles = doc.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        problems.append(Problem("tiles", "URL テンプレートの配列（1 つ以上）が必要です"))
    else:
        for index, url in enumerate(tiles):
            if not isinstance(url, str) or not all(t in url for t in ("{z}", "{x}", "{y}")):
                problems.append(
                    Problem(f"tiles[{index}]", f"{{z}}/{{x}}/{{y}} を含む必要があります: {url!r}")
                )

    tile_size = doc.get("tileSize")
    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size <= 0:
        problems.append(
            Problem("tileSize", f"正の整数が必要です（仕様 §2.1 REQUIRED）: {tile_size!r}")
        )

    fmt = doc.get("format")
    if fmt is not None and fmt not in FORMATS:
        problems.append(Problem("format", f"{' / '.join(FORMATS)} のいずれか: {fmt!r}"))

    minzoom, maxzoom = doc.get("minzoom"), doc.get("maxzoom")
    if isinstance(minzoom, int) and isinstance(maxzoom, int) and minzoom > maxzoom:
        problems.append(Problem("minzoom", f"maxzoom({maxzoom}) を超えています: {minzoom}"))

    bounds = doc.get("bounds")
    if bounds is not None:
        if not isinstance(bounds, list) or len(bounds) != 4:
            problems.append(Problem("bounds", "[west, south, east, north] が必要です"))
        else:
            west, south, east, north = bounds
            if west >= east or south >= north:
                problems.append(Problem("bounds", f"west<east, south<north が必要です: {bounds}"))

    datapng = doc.get("datapng")
    if datapng is None:
        problems.append(Problem("datapng", "拡張キーがありません（データPNG タイルではない？）"))
        return problems

    validator = Draft202012Validator(load_schema())
    for error in sorted(validator.iter_errors(datapng), key=lambda e: list(e.path)):
        location = "datapng" + "".join(f"[{p!r}]" for p in error.path)
        problems.append(Problem(location, error.message))

    problems.extend(_check_datapng_semantics(datapng))
    return problems


def _check_datapng_semantics(datapng: dict[str, Any]) -> list[Problem]:
    """スキーマでは表せない意味的な制約を見る。"""
    problems: list[Problem] = []
    kind = datapng.get("type")

    if kind == "palette":
        # 仕様 §3・§7: パレット型に数値型のフィールドは該当しない
        for key in ("factor", "offset", "invalidColor", "specialEncoding", "unit", "dataRange"):
            if key in datapng:
                problems.append(
                    Problem(f"datapng.{key}", "パレット型には該当しないフィールドです（§7）")
                )
        legend = datapng.get("legend")
        if isinstance(legend, dict):
            colors = [(i.get("r"), i.get("g"), i.get("b")) for i in legend.get("items", [])]
            if len(colors) != len(set(colors)):
                problems.append(
                    Problem(
                        "datapng.legend.items",
                        "同じ色の項目が複数あります（意味を一意に引けません）",
                    )
                )

    if kind == "numerical" and datapng.get("specialEncoding") not in (None, False):
        # 仕様 §3.2.1: specialEncoding 指定時、factor・offset は無視される（MUST）
        for key in ("factor", "offset"):
            if key in datapng:
                problems.append(
                    Problem(
                        f"datapng.{key}",
                        f"specialEncoding={datapng['specialEncoding']!r} では無視されます"
                        "（宣言すると読み手を誤らせます）",
                    )
                )

    support = datapng.get("support")
    if isinstance(support, dict) and support.get("type") == "block" and "anchor" in support:
        problems.append(Problem("datapng.support.anchor", "type=block では無視されます（§3.4）"))

    return problems


def validate_tiles(
    doc: dict[str, Any],
    root: Path | str,
    *,
    sample: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 0,
) -> list[Problem]:
    """宣言と実タイルの突合。

    Args:
        sample: 中身まで開くタイル数。0 以下で全件。決定的に選ぶため乱数種を固定する
    """
    root = Path(root)
    problems: list[Problem] = []

    declared_format = doc.get("format", "webp")
    extension = f".{declared_format}" if declared_format in FORMATS else None
    summary = scan_tree(root, extension)
    if summary is None:
        problems.append(Problem(str(root), "タイルが 1 枚もありません"))
        return problems

    if doc.get("minzoom") != summary.min_zoom:
        problems.append(
            Problem("minzoom", f"宣言 {doc.get('minzoom')} に対し実際は {summary.min_zoom}")
        )
    if doc.get("maxzoom") != summary.max_zoom:
        problems.append(
            Problem("maxzoom", f"宣言 {doc.get('maxzoom')} に対し実際は {summary.max_zoom}")
        )

    bounds = doc.get("bounds")
    if isinstance(bounds, list) and len(bounds) == 4:
        west, south, east, north = bounds
        aw, as_, ae, an = summary.bounds
        # 実際のタイルが宣言範囲からはみ出していたら、クライアントがそこを描かない
        if aw < west - 1e-6 or as_ < south - 1e-6 or ae > east + 1e-6 or an > north + 1e-6:
            problems.append(
                Problem("bounds", f"宣言 {bounds} の外にタイルがあります（実際 {summary.bounds}）")
            )

    tiles = _list_tiles(root, summary.extension)
    if sample > 0 and len(tiles) > sample:
        tiles = random.Random(seed).sample(tiles, sample)
        tiles.sort()

    datapng = doc.get("datapng") or {}
    problems.extend(_check_tile_contents(tiles, datapng, declared_format, doc.get("tileSize")))
    return problems


def _list_tiles(root: Path, extension: str) -> list[Path]:
    return sorted(
        p
        for p in root.rglob(f"*{extension}")
        if p.is_file() and p.stem.isdigit() and p.parent.name.isdigit()
    )


def _check_tile_contents(
    tiles: list[Path],
    datapng: dict[str, Any],
    declared_format: str,
    tile_size: Any,
) -> list[Problem]:
    problems: list[Problem] = []
    kind = datapng.get("type")
    invalid_color = datapng.get("invalidColor")
    legend = datapng.get("legend")
    legend_colors: set[tuple[int, int, int]] | None = None
    if kind == "palette" and isinstance(legend, dict):
        legend_colors = {
            (item.get("r"), item.get("g"), item.get("b")) for item in legend.get("items", [])
        }

    alpha_tiles_seen = False
    for path in tiles:
        where = str(path)
        actual_format = detect_format(path)
        if actual_format is not None and actual_format != declared_format:
            problems.append(
                Problem(where, f"format 宣言 {declared_format!r} に対し実体は {actual_format!r}")
            )

        rgb, alpha = load_tile(path)
        if isinstance(tile_size, int) and rgb.shape[:2] != (tile_size, tile_size):
            problems.append(
                Problem(
                    where,
                    f"tileSize 宣言 {tile_size} に対し実体は {rgb.shape[1]}×{rgb.shape[0]}",
                )
            )

        if alpha is not None:
            alpha_tiles_seen = True

        if legend_colors is not None:
            visible = rgb.reshape(-1, 3) if alpha is None else rgb[alpha > 0]
            if visible.size:
                found = {tuple(int(c) for c in row) for row in np.unique(visible, axis=0)}
                unknown = found - legend_colors
                if unknown:
                    listed = ", ".join(str(c) for c in sorted(unknown)[:5])
                    problems.append(Problem(where, f"凡例に無い色があります: {listed}"))

    if invalid_color is not None and alpha_tiles_seen:
        # 仕様 §3.2.2: アルファチャンネルを持つタイルと invalidColor は併用できない
        problems.append(
            Problem(
                "datapng.invalidColor",
                "アルファチャンネルを持つタイルと併用できません（仕様 §3.2.2 MUST NOT）。"
                " WebP の可逆圧縮は透明画素の RGB を保存しないため、宣言した色が実タイルに"
                "入っている保証がありません",
            )
        )
    return problems


def validate(
    tilejson_path: Path | str,
    tiles_root: Path | str | None = None,
    *,
    sample: int = DEFAULT_SAMPLE_SIZE,
) -> list[Problem]:
    """TileJSON を検証し、`tiles_root` があれば実タイルとも突き合わせる。"""
    doc = json.loads(Path(tilejson_path).read_text(encoding="utf-8"))
    problems = validate_document(doc)
    if tiles_root is not None:
        problems.extend(validate_tiles(doc, tiles_root, sample=sample))
    return problems
