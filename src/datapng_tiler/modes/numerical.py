"""数値型タイル（仕様 §3.2）。

任意の連続値ラスタ（標高・水深・気温・濃度など分野を問わない）を、`factor` 刻みの整数へ
量子化して RGB に格納する。無効値は既定でアルファ 0 で表す（仕様 §3.2.2）。

**無効値の表し方は 2 択で、混ぜない**:

- `invalid_color=None`（既定）: アルファ 0 で表す。無効画素があるタイルは RGBA、
  1 画素も無い場合は RGB（アルファチャンネルを持たない）で書く。アルファを持たない
  タイルに無効画素は無いので、`invalidColor` を宣言しなくても判定は一意に定まる。
- `invalid_color=(r, g, b)`: アルファを使わず、無効画素をその色で塗る。TileJSON には
  `invalidColor` を出す。有効値がその色に符号化されると区別できなくなるため、衝突を検査する。

**オーバービューは raw 整数のまま合成する**（`factor` に依存しない）。値は raw のアフィン
関数なので、整数領域の平均と値領域の平均は一致し、量子化誤差を積み重ねずに済む。

**再投影は厳密寄りのトランスフォーマで行う**（`DEFAULT_TOLERANCE`）。GDAL 既定の近似
トランスフォーマは、近似誤差が出力解像度（＝生成ズーム）によって変わるため、同じ地点を
狙っても標本位置がサブピクセルでにじみ、急斜面で「ズームによって値が違う」現象を生む。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from datapng_tiler.codec import NumericalEncoding
from datapng_tiler.geo import WEB_MERCATOR, TileWindow, WarpedVrtParams
from datapng_tiler.imageio import load_tile, rgb_to_image
from datapng_tiler.modes.base import ChildSlot, TileMode

# 再投影トランスフォーマの誤差許容（単位: ソース画素）。GDAL 既定の 0.125 は近似多項式を
# 出力格子上でフィットするため、近似誤差が生成ズームによって変わる。数値タイルは可逆性が
# 要なので、この偽誤差が量子化の半幅より十分小さくなるところまで絞る。0 は GDAL が
# 受け付けないため微小な正値を使う。
DEFAULT_TOLERANCE = 1e-4

# 選べる再投影カーネル。数値データにとって意味のあるものだけを載せる
# （cubic_spline は補間ではなく平滑化フィルタなので入れない）。
RESAMPLING_CHOICES: dict[str, Resampling] = {
    "nearest": Resampling.nearest,  # 補間しない（原典値をそのまま運ぶ。最大半画素の位置ずれ）
    "bilinear": Resampling.bilinear,  # 既定。偽値を作らないが高周波成分を位相依存で減衰
    "cubic": Resampling.cubic,  # 減衰は減るが、崖でオーバーシュート（偽値）が出る
    "lanczos": Resampling.lanczos,  # 減衰は最小。オーバーシュートは最大
}
DEFAULT_RESAMPLING = "bilinear"


class InvalidColorCollision(ValueError):
    """有効値が無効色と同じ RGB に符号化され、区別できなくなる。"""


@dataclass(frozen=True)
class NumericalTile:
    """タイル 1 枚ぶんの値と有効マスク。"""

    values: np.ndarray  # (h, w) float64
    valid: np.ndarray  # (h, w) bool


def downsample_topleft(raw: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2×2 ブロックの左上画素を選ぶ（左上法）。

    左上法では親タイル画素 (i, j) の節点が子タイル画素 (2i, 2j) の節点とちょうど一致する。
    したがって「左上を取る」ことは間引きではなく**同じ点の値をそのまま運ぶ**操作であり、
    補間で生まれる中間値（量子化値として意味を持たない）を作らない。
    """
    return raw[::2, ::2], valid[::2, ::2]


def downsample_average(raw: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """無効画素を除いた 2×2 の整数平均（block support）。

    raw 整数のまま平均し、半数は偶数側へ丸める（`np.rint`）。カスケードしても平均が
    片側へ偏らない。有効画素が 1 つも無いブロックだけが無効になる。
    """
    h, w = raw.shape
    blocks = raw.reshape(h // 2, 2, w // 2, 2).astype(np.int64)
    mask = valid.reshape(h // 2, 2, w // 2, 2)

    count = mask.sum(axis=(1, 3))
    total = np.where(mask, blocks, 0).sum(axis=(1, 3))

    out_valid = count > 0
    out = np.zeros(count.shape, dtype=np.int64)
    np.rint(
        np.divide(total, count, out=np.zeros(count.shape, dtype=np.float64), where=out_valid),
        out=out,
        where=out_valid,
        casting="unsafe",
    )
    return out.astype(np.int32), out_valid


@dataclass(frozen=True, kw_only=True)
class NumericalMode(TileMode):
    """数値型タイルの生成。

    Args:
        encoding: 符号化方式（`factor` / `offset` / `specialEncoding`）
        band: 対象バンド（1 始まり）
        src_nodata: ソースの無効値。``None`` ならソース自身の宣言を使う
        resampling: 再投影カーネル名（`RESAMPLING_CHOICES`）
        tolerance: 再投影トランスフォーマの誤差許容（`DEFAULT_TOLERANCE` 参照）
        invalid_color: 無効画素を塗る色。``None`` ならアルファ 0 で表す
        unit / data_range / precision: TileJSON に載せる補助情報
    """

    encoding: NumericalEncoding = field(default_factory=NumericalEncoding)
    band: int = 1
    src_nodata: float | None = None
    resampling: str = DEFAULT_RESAMPLING
    tolerance: float = DEFAULT_TOLERANCE
    invalid_color: tuple[int, int, int] | None = None
    unit: str | None = None
    data_range: tuple[float, float] | None = None
    precision: float | None = None

    key = "numerical"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.resampling not in RESAMPLING_CHOICES:
            raise ValueError(
                f"未知の resampling: {self.resampling!r}"
                f"（利用可能: {', '.join(RESAMPLING_CHOICES)}）"
            )
        if self.band < 1:
            raise ValueError(f"band は 1 以上であるべきです: {self.band!r}")
        if self.invalid_color is not None:
            if len(self.invalid_color) != 3 or not all(
                isinstance(v, int) and 0 <= v <= 255 for v in self.invalid_color
            ):
                raise ValueError(f"invalid_color は 0〜255 の整数 3 つ: {self.invalid_color!r}")

    @property
    def overview_method(self) -> str:
        """オーバービューの縮小方式。`support` から一意に決まる。"""
        return "topleft" if self.support == "point" else "average"

    # --- 読み取り -------------------------------------------------------------------

    def make_warped_vrt(self, src: rasterio.DatasetReader, params: WarpedVrtParams) -> WarpedVRT:
        nodata = self.src_nodata if self.src_nodata is not None else src.nodata
        return WarpedVRT(
            src,
            crs=WEB_MERCATOR,
            resampling=RESAMPLING_CHOICES[self.resampling],
            src_nodata=nodata,
            nodata=nodata,
            # nodata が無いと「ソースが覆っていない場所」も 0 で埋まって有効値に見える。
            # add_alpha でワープの被覆マスクを 1 バンド増やし、それを有効判定に使う。
            add_alpha=nodata is None,
            transform=params.transform,
            width=params.width,
            height=params.height,
            tolerance=self.tolerance,
        )

    def read_tile(self, vrt: WarpedVRT, window: TileWindow) -> NumericalTile:
        """ウィンドウを読み、値と有効マスクを返す。

        有効判定は **GDAL のマスク**に従う。「nodata に近い値を無効とみなす」ような
        閾値判定はしない——nodata が 0 や正の値であるデータでは、閾値判定が有効値を
        無効化してしまう。nodata が無いソースでは `add_alpha` の被覆マスクを使う。
        """
        rio_window = rasterio.windows.Window(
            window.read_col, window.read_row, window.read_w, window.read_h
        )
        data = vrt.read(indexes=self.band, window=rio_window, masked=True)
        values = np.ma.getdata(data).astype(np.float64)
        valid = ~np.ma.getmaskarray(data)

        if vrt.nodata is None:
            # nodata が無い場合、masked=True は何もマスクしない。`make_warped_vrt` が
            # add_alpha で足した被覆マスク（最終バンド）を有効判定に使う。
            coverage = vrt.read(indexes=vrt.count, window=rio_window)
            valid = valid & (coverage > 0)

        valid = valid & ~np.isnan(values)

        if window.read_w != self.tile_size or window.read_h != self.tile_size:
            full_values = np.zeros((self.tile_size, self.tile_size), dtype=np.float64)
            full_valid = np.zeros((self.tile_size, self.tile_size), dtype=bool)
            r0, r1 = window.dst_row, window.dst_row + window.read_h
            c0, c1 = window.dst_col, window.dst_col + window.read_w
            full_values[r0:r1, c0:c1] = values
            full_valid[r0:r1, c0:c1] = valid
            values, valid = full_values, full_valid

        return NumericalTile(values=values, valid=valid)

    # --- 符号化 ---------------------------------------------------------------------

    def build_image(self, data: NumericalTile) -> Image.Image | None:
        if not data.valid.any():
            return None
        rgb, valid = self.encoding.encode(data.values, valid=data.valid)
        if not valid.any():
            return None
        return self._compose(rgb, valid)

    def _compose(self, rgb: np.ndarray, valid: np.ndarray) -> Image.Image:
        """RGB と有効マスクから、無効値の表し方に応じた画像を作る。"""
        if self.invalid_color is None:
            if valid.all():
                # 無効画素が 1 つも無いので、アルファチャンネルを持たない RGB で書く。
                # 読み手はアルファが無い＝無効画素が無いと解釈でき、容量も減る。
                return rgb_to_image(rgb)
            alpha = np.where(valid, 255, 0).astype(np.uint8)
            # 透明画素の RGB は仕様上意味を持たない（WebP は保存しない）ので 0 に潰す
            rgb = np.where(valid[..., np.newaxis], rgb, 0).astype(np.uint8)
            return rgb_to_image(rgb, alpha)

        color = np.array(self.invalid_color, dtype=np.uint8)
        collision = valid & np.all(rgb == color, axis=-1)
        if collision.any():
            raise InvalidColorCollision(
                f"有効値が無効色 {tuple(int(v) for v in color)} と同じ RGB に符号化されました"
                f"（{int(collision.sum())} 画素）。この色は無効値と区別できません。"
                " --invalid-color で別の色を指定するか、アルファで無効値を表してください"
            )
        rgb = np.where(valid[..., np.newaxis], rgb, color).astype(np.uint8)
        return rgb_to_image(rgb)

    # --- オーバービュー -------------------------------------------------------------

    def _load_child(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """子タイルを raw 整数と有効マスクとして読む。"""
        rgb, alpha = load_tile(path)
        raw = self.encoding.rgb_to_raw(rgb).astype(np.int32)
        if alpha is not None:
            valid = alpha > 0
        elif self.invalid_color is not None:
            valid = ~np.all(rgb == np.array(self.invalid_color, dtype=np.uint8), axis=-1)
        else:
            # アルファを持たないタイルに無効画素は無い（`_compose` の不変条件）
            valid = np.ones(raw.shape, dtype=bool)
        return raw, valid

    def combine_children(self, slots: list[ChildSlot]) -> Image.Image | None:
        ts = self.tile_size
        canvas = np.zeros((ts * 2, ts * 2), dtype=np.int32)
        canvas_valid = np.zeros((ts * 2, ts * 2), dtype=bool)
        for slot in slots:
            raw, valid = self._load_child(slot.path)
            canvas[slot.row : slot.row + ts, slot.col : slot.col + ts] = raw[:ts, :ts]
            canvas_valid[slot.row : slot.row + ts, slot.col : slot.col + ts] = valid[:ts, :ts]

        if self.overview_method == "topleft":
            raw, valid = downsample_topleft(canvas, canvas_valid)
        else:
            raw, valid = downsample_average(canvas, canvas_valid)

        if not valid.any():
            return None
        return self._compose(self.encoding.raw_to_rgb(raw), valid)

    # --- TileJSON -------------------------------------------------------------------

    def datapng(self) -> dict[str, Any]:
        fields: dict[str, Any] = {"type": "numerical"}
        fields.update(self.encoding.datapng_fields())
        if self.unit:
            fields["unit"] = self.unit
        # invalidColor はアルファを持たないタイル専用（仕様 §3.2.2）。アルファで無効値を
        # 表す既定の出力では、宣言してはならない（MUST NOT）。
        if self.invalid_color is not None:
            fields["invalidColor"] = list(self.invalid_color)
        if self.data_range is not None:
            fields["dataRange"] = {"min": self.data_range[0], "max": self.data_range[1]}
        if self.precision is not None:
            fields["precision"] = self.precision
        fields["support"] = self.support_field()
        return fields
