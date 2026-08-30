"""パレットPNGタイル（仕様 §3.3）。

区分（土地利用・浸水深の階級など）を色で表すタイル。クライアントは**RGB の完全一致**で
凡例を引くので、色は 1 バイトも変えてはならない。したがって:

- 再投影は常に nearest（補間すると凡例に無い中間色ができる）
- オーバービューも色を作らない方式のみ（左上法 or 多数決）
- 出力は可逆圧縮のみ（`imageio` が WebP lossless / PNG を保証する）

入力は 2 通り:

- **クラス値ラスタ**（1 バンドの整数）: 凡例の `value` で色に対応づける
- **RGB ラスタ**（3 バンド）: 色をそのまま使う。凡例に無い色が現れたら報告する

内部ではどちらも「凡例項目の番号」の配列に正規化して扱う。PNG ではその番号を
インデックスカラーとしてそのまま書けるため、容量が大きく下がる。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from datapng_tiler.geo import WEB_MERCATOR, TileWindow, WarpedVrtParams
from datapng_tiler.imageio import indexed_to_image, load_tile, rgb_to_image, save_indexed
from datapng_tiler.legend import Legend
from datapng_tiler.modes.base import ChildSlot, TileMode

# インデックス 0 は無効値（透明）に予約する。凡例項目は 1 番から並べる。
TRANSPARENT_INDEX = 0

# 凡例に無い色を見つけたときに報告する上限（1 タイルあたり）。全部集めても読めないので、
# 「どんな色が」を知るのに十分な数だけ持つ。
_UNKNOWN_COLOR_SAMPLES = 8


class UnknownColorError(ValueError):
    """凡例に無い色が入力に現れた。"""


@dataclass(frozen=True)
class PaletteTile:
    """タイル 1 枚ぶんの凡例インデックス（0 = 無効）。"""

    indices: np.ndarray  # (h, w) uint8


def _unknown_color_message(colors: list[tuple[int, int, int]], count: int) -> str:
    listed = ", ".join(str(c) for c in colors)
    return (
        f"凡例に無い色が {count} 画素ありました（例: {listed}）。"
        " 凡例定義に項目を足すか、--on-unknown-color nodata で無効値として扱ってください"
    )


def downsample_majority(indices: np.ndarray) -> np.ndarray:
    """2×2 ブロックの多数決（block support）。

    同数のときは**若い凡例番号**を採る。順序が決まっているので、同じ入力からは必ず
    同じタイルが出る（決定的）。無効（0）は候補に入れず、4 画素すべて無効のときだけ
    無効を返す。
    """
    h, w = indices.shape
    blocks = indices.reshape(h // 2, 2, w // 2, 2).transpose(0, 2, 1, 3)
    flat = blocks.reshape(h // 2, w // 2, 4)

    # 各凡例番号の出現数を数え、(出現数が最大, 番号が最小) を選ぶ。
    # 256 通りしかないので、番号ごとの一致数を積み上げるのが単純かつ速い。
    best = np.zeros(flat.shape[:2], dtype=np.uint8)
    best_count = np.zeros(flat.shape[:2], dtype=np.int8)
    present = np.unique(flat)
    for index in present:
        if index == TRANSPARENT_INDEX:
            continue
        count = (flat == index).sum(axis=2).astype(np.int8)
        take = count > best_count
        best = np.where(take, np.uint8(index), best)
        best_count = np.where(take, count, best_count)
    return best


@dataclass(frozen=True, kw_only=True)
class PaletteMode(TileMode):
    """パレットPNGタイルの生成。

    Args:
        legend: 凡例（色と意味の対応。TileJSON にも同じものを載せる）
        band: クラス値ラスタのときの対象バンド（RGB ラスタでは無視）
        src_nodata: ソースの無効値。``None`` ならソース自身の宣言を使う
        on_unknown_color: 凡例に無い色/値の扱い。``error``（既定）/ ``nodata``
        legend_url: 凡例を外部参照にする場合の URL（仕様 §3.3.2）
    """

    legend: Legend
    band: int = 1
    src_nodata: float | None = None
    on_unknown_color: str = "error"
    legend_url: str | None = None

    key = "palette"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.on_unknown_color not in ("error", "nodata"):
            raise ValueError(
                f"未知の on_unknown_color: {self.on_unknown_color!r}（error / nodata）"
            )
        if self.band < 1:
            raise ValueError(f"band は 1 以上であるべきです: {self.band!r}")

    @property
    def overview_method(self) -> str:
        return "topleft" if self.support == "point" else "majority"

    @property
    def palette(self) -> list[tuple[int, int, int]]:
        """インデックスカラー用のパレット（0 番は無効値の予約席）。"""
        return [(0, 0, 0), *self.legend.colors()]

    # --- 読み取り -------------------------------------------------------------------

    def make_warped_vrt(self, src: rasterio.DatasetReader, params: WarpedVrtParams) -> WarpedVRT:
        nodata = self.src_nodata if self.src_nodata is not None else src.nodata
        return WarpedVRT(
            src,
            crs=WEB_MERCATOR,
            # 補間は凡例に無い中間色を作るため、常に nearest。
            resampling=Resampling.nearest,
            src_nodata=nodata,
            nodata=nodata,
            # nodata が無いと「ソースが覆っていない場所」も 0 で埋まり、凡例に無い色
            # (0,0,0) として現れる。add_alpha でワープの被覆マスクを足して有効判定に使う。
            add_alpha=nodata is None,
            transform=params.transform,
            width=params.width,
            height=params.height,
        )

    def read_tile(self, vrt: WarpedVRT, window: TileWindow) -> PaletteTile:
        rio_window = rasterio.windows.Window(
            window.read_col, window.read_row, window.read_w, window.read_h
        )
        if self._source_band_count(vrt) >= 3:
            indices, valid = self._read_rgb(vrt, rio_window)
        else:
            indices, valid = self._read_classes(vrt, rio_window)

        indices = np.where(valid, indices, TRANSPARENT_INDEX).astype(np.uint8)

        if window.read_w != self.tile_size or window.read_h != self.tile_size:
            full = np.full((self.tile_size, self.tile_size), TRANSPARENT_INDEX, dtype=np.uint8)
            r0, r1 = window.dst_row, window.dst_row + window.read_h
            c0, c1 = window.dst_col, window.dst_col + window.read_w
            full[r0:r1, c0:c1] = indices
            indices = full

        return PaletteTile(indices=indices)

    @staticmethod
    def _source_band_count(vrt: WarpedVRT) -> int:
        """ソース由来のバンド数（`add_alpha` で足した被覆マスクを除く）。"""
        return vrt.count - 1 if vrt.nodata is None else vrt.count

    def _read_rgb(self, vrt: WarpedVRT, window) -> tuple[np.ndarray, np.ndarray]:
        """RGB ラスタを凡例インデックスへ写す。"""
        bands = vrt.read(indexes=[1, 2, 3], window=window, masked=True)
        rgb = np.ma.getdata(bands).astype(np.uint8).transpose(1, 2, 0)
        valid = ~np.ma.getmaskarray(bands).any(axis=0)
        if self._source_band_count(vrt) >= 4:
            # ソース自身の 4 バンド目（アルファ）も有効判定に使う
            valid = valid & (vrt.read(indexes=4, window=window) > 0)
        if vrt.nodata is None:
            # add_alpha で足した被覆マスク（最終バンド）
            valid = valid & (vrt.read(indexes=vrt.count, window=window) > 0)

        indices = np.zeros(rgb.shape[:2], dtype=np.uint8)
        matched = np.zeros(rgb.shape[:2], dtype=bool)
        for position, color in enumerate(self.legend.colors(), start=1):
            hit = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
            indices = np.where(hit, np.uint8(position), indices)
            matched |= hit

        unknown = valid & ~matched
        if unknown.any():
            valid = self._handle_unknown(rgb[unknown], int(unknown.sum()), valid, unknown)
        return indices, valid

    def _read_classes(self, vrt: WarpedVRT, window) -> tuple[np.ndarray, np.ndarray]:
        """クラス値ラスタを凡例インデックスへ写す。"""
        if not self.legend.has_values:
            raise ValueError(
                "1 バンドのラスタを入力にするには、凡例の全項目に value（クラス値）が必要です"
            )
        data = vrt.read(indexes=self.band, window=window, masked=True)
        values = np.ma.getdata(data)
        valid = ~np.ma.getmaskarray(data)

        indices = np.zeros(values.shape, dtype=np.uint8)
        matched = np.zeros(values.shape, dtype=bool)
        for position, item in enumerate(self.legend.items, start=1):
            hit = values == item.value
            indices = np.where(hit, np.uint8(position), indices)
            matched |= hit

        unknown = valid & ~matched
        if unknown.any():
            unique_values = np.unique(values[unknown])[:_UNKNOWN_COLOR_SAMPLES]
            samples = [(int(v), int(v), int(v)) for v in unique_values]
            valid = self._handle_unknown(
                None, int(unknown.sum()), valid, unknown, value_samples=samples
            )
        return indices, valid

    def _handle_unknown(
        self,
        rgb_samples: np.ndarray | None,
        count: int,
        valid: np.ndarray,
        unknown: np.ndarray,
        *,
        value_samples: list[tuple[int, int, int]] | None = None,
    ) -> np.ndarray:
        if self.on_unknown_color == "nodata":
            return valid & ~unknown
        if value_samples is not None:
            listed = ", ".join(str(v[0]) for v in value_samples)
            raise UnknownColorError(
                f"凡例に無いクラス値が {count} 画素ありました（例: {listed}）。"
                " 凡例定義に項目を足すか、--on-unknown-color nodata を指定してください"
            )
        colors = [tuple(int(c) for c in row) for row in np.asarray(rgb_samples)]
        common = [color for color, _ in Counter(colors).most_common(_UNKNOWN_COLOR_SAMPLES)]
        raise UnknownColorError(_unknown_color_message(common, count))

    # --- 符号化 ---------------------------------------------------------------------

    def build_image(self, data: PaletteTile) -> Image.Image | None:
        return self._compose(data.indices)

    def _compose(self, indices: np.ndarray) -> Image.Image | None:
        has_invalid = bool((indices == TRANSPARENT_INDEX).any())
        if not (indices != TRANSPARENT_INDEX).any():
            return None
        if self.fmt.supports_palette:
            return indexed_to_image(
                indices, self.palette, TRANSPARENT_INDEX if has_invalid else None
            )
        # WebP はインデックスカラーを持たないので RGB(A) へ展開する
        lut = np.array(self.palette, dtype=np.uint8)
        rgb = lut[indices]
        if not has_invalid:
            return rgb_to_image(rgb)
        alpha = np.where(indices == TRANSPARENT_INDEX, 0, 255).astype(np.uint8)
        return rgb_to_image(rgb, alpha)

    def save_image(self, image: Image.Image, path: Path) -> None:
        if image.mode == "P":
            save_indexed(image, path, self.fmt, image.info.get("transparency"))
        else:
            self.fmt.save(image, path)

    # --- オーバービュー -------------------------------------------------------------

    def _load_child(self, path: Path) -> np.ndarray:
        """子タイルを凡例インデックスの配列として読む。"""
        rgb, alpha = load_tile(path)
        indices = np.zeros(rgb.shape[:2], dtype=np.uint8)
        for position, color in enumerate(self.legend.colors(), start=1):
            indices = np.where(
                np.all(rgb == np.array(color, dtype=np.uint8), axis=-1),
                np.uint8(position),
                indices,
            )
        if alpha is not None:
            indices = np.where(alpha > 0, indices, np.uint8(TRANSPARENT_INDEX))
        return indices

    def combine_children(self, slots: list[ChildSlot]) -> Image.Image | None:
        ts = self.tile_size
        canvas = np.full((ts * 2, ts * 2), TRANSPARENT_INDEX, dtype=np.uint8)
        for slot in slots:
            child = self._load_child(slot.path)
            canvas[slot.row : slot.row + ts, slot.col : slot.col + ts] = child[:ts, :ts]

        if self.overview_method == "topleft":
            indices = canvas[::2, ::2]
        else:
            indices = downsample_majority(canvas)
        return self._compose(indices)

    # --- TileJSON -------------------------------------------------------------------

    def datapng(self) -> dict[str, Any]:
        # 仕様 §3・§7: パレットPNGでは factor / offset / invalidColor を出さない。
        # 無効値は透明のみで表す。
        return {
            "type": "palette",
            "legend": self.legend_url if self.legend_url else self.legend.to_dict(),
            "support": self.support_field(),
        }
