"""パレット型タイルの凡例定義（仕様 §3.3）。

凡例は「色 ↔ 意味」の対応表であり、パレット型タイルでは**必須**（REQUIRED）。
本ツールでは、タイルを作るための入力（どのクラス値をどの色にするか）と、TileJSON へ
載せるメタデータの両方を、同じ 1 つの定義ファイルから作る。片方だけ手で書くと、
凡例に無い色がタイルに現れる／タイルに無い色が凡例に載る、という食い違いが起きる。

定義ファイル（YAML または JSON）:

```yaml
title: 洪水浸水想定区域（想定最大規模）浸水深
items:
  - value: 1          # クラス値ラスタを入力にするときの対応値（RGB ラスタでは省略）
    r: 245
    g: 245
    b: 50
    title: 0.5m未満
    description: 床下浸水相当。
```

仕様は凡例項目への任意メンバー追加を許す（クライアントは処理できないメンバーを無視する
MUST）。`value` もそのひとつとして TileJSON にそのまま載せる——タイルを作り直すときに
定義ファイルが手元に無くても、配信物から対応が読み取れる。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# インデックスカラー PNG のパレットは 256 色。0 番を無効値（透明）に予約する。
MAX_ITEMS = 255

# 凡例項目の予約キー（これ以外は追加メンバーとしてそのまま通す）
_ITEM_KEYS = ("r", "g", "b", "title", "description")


class LegendError(ValueError):
    """凡例定義が不正。"""


@dataclass(frozen=True)
class LegendItem:
    """凡例項目 1 つ（仕様 §3.3.1）。"""

    r: int
    g: int
    b: int
    title: str
    description: str | None = None
    extra: tuple[tuple[str, Any], ...] = ()

    @property
    def rgb(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)

    @property
    def value(self) -> Any:
        """クラス値ラスタでの対応値（追加メンバー `value`）。無ければ ``None``。"""
        return dict(self.extra).get("value")

    def to_dict(self) -> dict[str, Any]:
        item: dict[str, Any] = {"r": self.r, "g": self.g, "b": self.b, "title": self.title}
        if self.description is not None:
            item["description"] = self.description
        item.update(dict(self.extra))
        return item

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, where: str) -> LegendItem:
        for key in ("r", "g", "b", "title"):
            if key not in data:
                raise LegendError(f"{where}: 必須キー {key!r} がありません")
        channels = []
        for key in ("r", "g", "b"):
            value = data[key]
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255:
                raise LegendError(f"{where}: {key} は 0〜255 の整数であるべきです: {value!r}")
            channels.append(value)
        title = data["title"]
        if not isinstance(title, str) or not title:
            raise LegendError(f"{where}: title は空でない文字列であるべきです: {title!r}")
        description = data.get("description")
        if description is not None and not isinstance(description, str):
            raise LegendError(f"{where}: description は文字列であるべきです: {description!r}")
        extra = tuple((k, v) for k, v in data.items() if k not in _ITEM_KEYS)
        return cls(*channels, title=title, description=description, extra=extra)


@dataclass(frozen=True)
class Legend:
    """凡例（仕様 §3.3.1 の凡例オブジェクト）。"""

    items: tuple[LegendItem, ...]
    title: str | None = None
    _color_index: dict[tuple[int, int, int], int] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.items:
            raise LegendError("凡例に items が 1 つもありません")
        if len(self.items) > MAX_ITEMS:
            raise LegendError(
                f"凡例項目は {MAX_ITEMS} 件までです（インデックスカラーの 256 色から"
                f"無効値の 1 色を除いた数）: {len(self.items)} 件"
            )
        colors: dict[tuple[int, int, int], int] = {}
        for index, item in enumerate(self.items):
            if item.rgb in colors:
                raise LegendError(
                    f"凡例に同じ色が 2 回現れます: {item.rgb}"
                    f"（{colors[item.rgb]} 番目と {index} 番目）。"
                    "色から意味を一意に引けなくなります"
                )
            colors[item.rgb] = index
        values = [item.value for item in self.items if item.value is not None]
        if len(values) != len(set(values)):
            raise LegendError("凡例に同じ value が複数あります（クラス値から色を一意に引けません）")
        object.__setattr__(self, "_color_index", colors)

    # --- 引き当て ---------------------------------------------------------------

    def index_of_color(self, rgb: tuple[int, int, int]) -> int | None:
        """色に対応する凡例項目の番号（0 始まり）。無ければ ``None``。"""
        return self._color_index.get(rgb)

    def value_to_index(self) -> dict[Any, int]:
        """クラス値 → 凡例項目番号。`value` を持つ項目だけを含む。"""
        return {
            item.value: index for index, item in enumerate(self.items) if item.value is not None
        }

    def colors(self) -> list[tuple[int, int, int]]:
        return [item.rgb for item in self.items]

    @property
    def has_values(self) -> bool:
        """全項目が `value` を持つ（クラス値ラスタを入力にできる）か。"""
        return all(item.value is not None for item in self.items)

    # --- 入出力 -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        legend: dict[str, Any] = {}
        if self.title:
            legend["title"] = self.title
        legend["items"] = [item.to_dict() for item in self.items]
        return legend

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Legend:
        if not isinstance(data, dict):
            raise LegendError(f"凡例はオブジェクトであるべきです: {type(data).__name__}")
        raw_items = data.get("items")
        if not isinstance(raw_items, list):
            raise LegendError("凡例に items（配列）がありません")
        items = tuple(
            LegendItem.from_dict(item, where=f"items[{index}]")
            if isinstance(item, dict)
            else _raise_item_type(index, item)
            for index, item in enumerate(raw_items)
        )
        title = data.get("title")
        if title is not None and not isinstance(title, str):
            raise LegendError(f"title は文字列であるべきです: {title!r}")
        return cls(items=items, title=title)

    @classmethod
    def load(cls, path: Path | str) -> Legend:
        """YAML または JSON の凡例定義を読む（拡張子ではなく YAML パーサで両方扱う）。"""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise LegendError(f"凡例定義を解釈できません: {path}（{exc}）") from exc
        try:
            return cls.from_dict(data)
        except LegendError as exc:
            raise LegendError(f"{path}: {exc}") from None

    def dump(self, path: Path | str) -> Path:
        """凡例オブジェクトを JSON で書き出す（`legend` の外部参照用）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return path


def _raise_item_type(index: int, item: Any) -> LegendItem:
    raise LegendError(f"items[{index}] はオブジェクトであるべきです: {type(item).__name__}")
