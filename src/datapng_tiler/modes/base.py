"""タイル種別の抽象基底。

共通スキャフォールド（`engine` + `geo` + 本基底）が、再投影・XYZ 幾何・プロセス並列・
レジューム・base→overview のループ・ウィンドウ読み取り・子タイル列挙を担う。
種別が差し替えるのは次の 4 点だけ:

- `make_warped_vrt`  : バンド・リサンプリング・無効値の設定
- `read_tile`        : ウィンドウを読んでタイル 1 枚ぶんの配列にする
- `build_image`      : 配列を保存可能な画像にする（全画素無効なら ``None``）
- `combine_children` : 既存の子タイル 4 枚を 1 枚のオーバービュータイルに束ねる

`support`（仕様 §3.4）の宣言と実際の生成方式はここで束ねる。片方だけ変えられる API に
すると、TileJSON の宣言と実タイルが食い違ったまま配信されうる。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.windows
from PIL import Image
from rasterio.vrt import WarpedVRT

from datapng_tiler.fileio import is_complete, remove_if_exists, tile_path
from datapng_tiler.geo import TileWindow, WarpedVrtParams, tile_window
from datapng_tiler.imageio import TileFormat

# 対応する support（仕様 §3.4）。
# - point/northwest: 画素値は画素の左上節点を代表する（左上法）
# - block:           画素値は画素の範囲全体の代表値（中心整列 + 平均縮小）
#
# point/center は**対応しない**。中心整列では、親タイルの画素中心に一致する子タイルの
# 画素が存在せず（1/4 画素ずれる）、オーバービューを「点の値」として作れないため。
# 中心の値が欲しい場合は block（平均）を使う——それが実際に得られる量である。
SUPPORT_CHOICES = ("point", "block")


@dataclass(frozen=True)
class ChildSlot:
    """オーバービュータイルへ貼り付ける子タイル 1 枚の位置と既存パス。"""

    row: int  # 2 倍キャンバス上の行オフセット
    col: int  # 2 倍キャンバス上の列オフセット
    path: Path


@dataclass(frozen=True, kw_only=True)
class TileMode:
    """タイル種別の基底（不変・pickle 可能）。

    ワーカプロセスへは、この値オブジェクトを丸ごと渡す。macOS/Windows の既定は
    spawn で、fork のようにグローバル変数が引き継がれないため、必要な状態は
    すべてここに入れて明示的に渡す必要がある。
    """

    tile_size: int = 512
    support: str = "point"
    fmt: TileFormat = TileFormat()

    key = "base"

    def __post_init__(self) -> None:
        if self.support not in SUPPORT_CHOICES:
            raise ValueError(
                f"未知の support: {self.support!r}（利用可能: {', '.join(SUPPORT_CHOICES)}）"
                "。point/center は対応していません（親画素の中心に一致する子画素が"
                "存在しないため、オーバービューを点の値として作れません）"
            )
        if self.tile_size <= 0 or self.tile_size % 2 != 0:
            raise ValueError(f"tile_size は正の偶数であるべきです: {self.tile_size!r}")

    @property
    def topleft(self) -> bool:
        """左上法（画素値が左上節点を代表する）か。`support` から一意に決まる。"""
        return self.support == "point"

    # --- 種別ごとの差し替え点 -------------------------------------------------------

    def make_warped_vrt(self, src: rasterio.DatasetReader, params: WarpedVrtParams) -> WarpedVRT:
        raise NotImplementedError

    def read_tile(self, vrt: WarpedVRT, window: TileWindow) -> Any:
        raise NotImplementedError

    def build_image(self, data: Any) -> Image.Image | None:
        """タイル配列を保存用の画像にする。全画素が無効なら ``None``（書き出さない）。"""
        raise NotImplementedError

    def save_image(self, image: Image.Image, path: Path) -> None:
        """既定は形式そのままの保存。インデックスカラーを使う種別が上書きする。"""
        self.fmt.save(image, path)

    def combine_children(self, slots: list[ChildSlot]) -> Image.Image | None:
        raise NotImplementedError

    def datapng(self) -> dict[str, Any]:
        """TileJSON の `datapng` オブジェクトを組み立てる。"""
        raise NotImplementedError

    def support_field(self) -> dict[str, str]:
        """`datapng.support`（仕様 §3.4）。生成方式と必ず一致する。"""
        if self.support == "point":
            return {"type": "point", "anchor": "northwest"}
        return {"type": "block"}

    # --- 共通スキャフォールド -------------------------------------------------------

    def read_windowed(
        self,
        vrt: WarpedVRT,
        window: TileWindow,
        indexes: int | list[int],
        fill_value: Any,
        dtype: Any,
    ) -> np.ndarray:
        """VRT からウィンドウを読み、縁を `fill_value` で埋めてタイルサイズに揃える。"""
        rio_window = rasterio.windows.Window(
            window.read_col, window.read_row, window.read_w, window.read_h
        )
        data = vrt.read(indexes=indexes, window=rio_window)
        if window.read_w == self.tile_size and window.read_h == self.tile_size:
            return data

        r0, r1 = window.dst_row, window.dst_row + window.read_h
        c0, c1 = window.dst_col, window.dst_col + window.read_w
        if isinstance(indexes, int):
            out = np.full((self.tile_size, self.tile_size), fill_value, dtype=dtype)
            out[r0:r1, c0:c1] = data
        else:
            out = np.full((len(indexes), self.tile_size, self.tile_size), fill_value, dtype=dtype)
            out[:, r0:r1, c0:c1] = data
        return out

    def render_base_tile(
        self,
        vrt: WarpedVRT,
        params: WarpedVrtParams,
        output_dir: Path,
        zoom: int,
        x: int,
        y: int,
        *,
        overwrite: bool = False,
    ) -> bool:
        """ベースタイル 1 枚を生成する。書き出したら ``True``。

        既存タイルはスキップする（レジューム）。`overwrite=True` は入力を更新したときに
        使う——既存を必ず飛ばす作りだと、元データを更新しても 1 枚も更新されない。
        """
        out_path = tile_path(output_dir, zoom, x, y, self.fmt.extension)
        if not overwrite and is_complete(out_path):
            return False

        window = tile_window(params, x, y, self.tile_size)
        if window is None:
            return False

        image = self.build_image(self.read_tile(vrt, window))
        if image is None:
            return False

        self.save_image(image, out_path)
        return True

    def render_overview_tile(
        self, output_dir: Path, zoom: int, x: int, y: int, *, overwrite: bool = False
    ) -> bool:
        """子タイル 4 枚から オーバービュータイル 1 枚を生成する。書き出したら ``True``。"""
        out_path = tile_path(output_dir, zoom, x, y, self.fmt.extension)
        if not overwrite and is_complete(out_path):
            return False

        # 0 バイトの壊れた子タイルは「無い」扱いにする（読み込みで落ちるのを防ぐ）
        slots = [s for s in self.child_slots(output_dir, zoom, x, y) if is_complete(s.path)]
        if not slots:
            return False

        image = self.combine_children(slots)
        if image is None:
            return False

        self.save_image(image, out_path)
        return True

    def prune_tile(self, output_dir: Path, zoom: int, x: int, y: int) -> bool:
        """指定タイルの既存ファイルを削除する（消えたら ``True``）。"""
        return remove_if_exists(tile_path(output_dir, zoom, x, y, self.fmt.extension))

    def child_slots(self, output_dir: Path, zoom: int, x: int, y: int) -> Iterator[ChildSlot]:
        """オーバービュータイル (zoom, x, y) の子タイル 4 枚の貼り付け位置とパスを返す。"""
        ts = self.tile_size
        for index, (cx, cy) in enumerate(child_coords(x, y)):
            yield ChildSlot(
                row=(index // 2) * ts,
                col=(index % 2) * ts,
                path=tile_path(output_dir, zoom + 1, cx, cy, self.fmt.extension),
            )


def child_coords(x: int, y: int) -> list[tuple[int, int]]:
    """オーバービュータイル (x, y) の子タイル 4 枚の座標（左上→右上→左下→右下）。"""
    return [(2 * x, 2 * y), (2 * x + 1, 2 * y), (2 * x, 2 * y + 1), (2 * x + 1, 2 * y + 1)]
