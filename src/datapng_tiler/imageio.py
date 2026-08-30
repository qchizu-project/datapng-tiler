"""タイル画像の読み書き（WebP / PNG）。

仕様 §2.2 の標準形式は **WebP（可逆圧縮）**で、PNG も許容される。ここは形式の差を
1 か所に閉じ込め、上位（mode / engine）が形式を意識しないようにする。

**無効値とアルファの関係（仕様 §3.2.2）**: アルファチャンネルを持つタイルの無効値は
アルファ 0 だけで表す。WebP の可逆圧縮は完全に透明な画素の RGB を保存しない
（`exact` を有効にしない限りエンコーダが書き換える）ため、透明画素の RGB に意味を
持たせてはならない。逆にアルファを持たない出力では、指定した無効色が**バイト単位で
保存される**必要がある（WebP・PNG とも可逆なので保存される）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from datapng_tiler.fileio import atomic_write

FORMATS = ("webp", "png")

# WebP の圧縮努力（0=最速/やや大, 6=最遅/最小）。可逆圧縮では method を上げても容量が
# ほとんど変わらない一方でエンコードが目に見えて遅くなるため、既定を小さく取る
# （実測は BENCHMARKS.md）。
DEFAULT_WEBP_METHOD = 1
# PNG の zlib 圧縮レベル（0〜9）。Pillow の既定と同じ 6。
DEFAULT_PNG_COMPRESS_LEVEL = 6


class ImageIOError(RuntimeError):
    """タイル画像の読み書きに失敗した。"""


@dataclass(frozen=True)
class TileFormat:
    """タイル画像の出力形式と圧縮設定（プロセス間で渡すため不変・pickle 可能）。"""

    name: str = "webp"
    webp_method: int = DEFAULT_WEBP_METHOD
    png_compress_level: int = DEFAULT_PNG_COMPRESS_LEVEL

    def __post_init__(self) -> None:
        if self.name not in FORMATS:
            raise ValueError(f"未知の形式: {self.name!r}（利用可能: {', '.join(FORMATS)}）")
        if not 0 <= self.webp_method <= 6:
            raise ValueError(f"webp_method は 0〜6: {self.webp_method!r}")
        if not 0 <= self.png_compress_level <= 9:
            raise ValueError(f"png_compress_level は 0〜9: {self.png_compress_level!r}")

    @property
    def extension(self) -> str:
        return f".{self.name}"

    @property
    def supports_palette(self) -> bool:
        """インデックスカラー（パレット）で保存できるか。

        PNG のみ。WebP にインデックスカラーモードは無い（可逆圧縮の内部で
        パレット化されるので、RGBA のまま書いても容量は大きくは増えない）。
        """
        return self.name == "png"

    def save(self, image: Image.Image, path: Path) -> None:
        """PIL 画像をタイルとして書き出す（アトミック）。"""
        with atomic_write(path) as tmp:
            if self.name == "webp":
                # lossless=True でも、既定ではアルファ 0 の画素の RGB は保存されない。
                # 無効値はアルファだけで表すため（仕様 §3.2.2）これで問題ない。
                image.save(tmp, "WEBP", lossless=True, method=self.webp_method)
            else:
                image.save(tmp, "PNG", optimize=False, compress_level=self.png_compress_level)


def rgb_to_image(rgb: np.ndarray, alpha: np.ndarray | None = None) -> Image.Image:
    """RGB 配列（+ 任意のアルファ）から PIL 画像を作る。

    Args:
        rgb: `(h, w, 3)` uint8
        alpha: `(h, w)` uint8。``None`` ならアルファチャンネルを持たない RGB 画像にする
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb は (h, w, 3) の uint8 配列であるべきです: {rgb.shape}/{rgb.dtype}")
    if alpha is None:
        return Image.fromarray(np.ascontiguousarray(rgb), "RGB")
    rgba = np.empty(rgb.shape[:2] + (4,), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, "RGBA")


def rgba_to_image(rgba: np.ndarray) -> Image.Image:
    """RGBA 配列 `(h, w, 4)` uint8 から PIL 画像を作る（並べ替え済みの配列を受け取る）。"""
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"rgba は (h, w, 4) の uint8 配列であるべきです: {rgba.shape}")
    return Image.fromarray(np.ascontiguousarray(rgba), "RGBA")


def indexed_to_image(
    indices: np.ndarray, palette: list[tuple[int, int, int]], transparent_index: int | None
) -> Image.Image:
    """インデックス配列とパレットから、インデックスカラーの PIL 画像を作る（PNG 用）。

    ブラウザで復号した結果はパレットを展開した RGB と一致するため、パレットPNGタイルを
    インデックスカラー PNG で配信しても仕様上の色一致判定は変わらない（容量だけが減る）。
    """
    if indices.dtype != np.uint8:
        raise ValueError(f"indices は uint8 であるべきです: {indices.dtype}")
    if len(palette) > 256:
        raise ValueError(f"パレットは 256 色までです: {len(palette)}")
    image = Image.fromarray(np.ascontiguousarray(indices), "P")
    flat: list[int] = []
    for r, g, b in palette:
        flat.extend((r, g, b))
    flat.extend([0] * (768 - len(flat)))
    image.putpalette(flat)
    if transparent_index is not None:
        image.info["transparency"] = transparent_index
    return image


def save_indexed(
    image: Image.Image, path: Path, fmt: TileFormat, transparent_index: int | None
) -> None:
    """インデックスカラー画像を書き出す（PNG は P のまま、WebP は RGBA へ展開）。"""
    if fmt.supports_palette:
        with atomic_write(path) as tmp:
            kwargs = {} if transparent_index is None else {"transparency": transparent_index}
            image.save(tmp, "PNG", optimize=False, compress_level=fmt.png_compress_level, **kwargs)
        return
    fmt.save(image.convert("RGBA" if transparent_index is not None else "RGB"), path)


def load_tile(path: Path | str) -> tuple[np.ndarray, np.ndarray | None]:
    """タイル画像を読み、(RGB `(h, w, 3)` uint8, アルファ `(h, w)` uint8 または ``None``) を返す。

    アルファが ``None`` なのは「そのタイルがアルファチャンネルを持たない」ことを意味する。
    仕様 §3.2.2 の無効値判定はこの区別に依存するので、呼び出し側へそのまま渡す。
    """
    path = Path(path)
    try:
        with Image.open(path) as image:
            mode = image.mode
            has_alpha = mode in ("RGBA", "LA") or (mode == "P" and "transparency" in image.info)
            converted = image.convert("RGBA" if has_alpha else "RGB")
            array = np.asarray(converted, dtype=np.uint8)
    except OSError as exc:
        raise ImageIOError(f"タイル画像を読めませんでした: {path}（{exc}）") from exc

    if has_alpha:
        return np.ascontiguousarray(array[..., :3]), np.ascontiguousarray(array[..., 3])
    return np.ascontiguousarray(array), None


def detect_format(path: Path | str) -> str | None:
    """ファイルの中身から形式名（"webp" / "png"）を判定する（判定できなければ ``None``）。

    宣言された `format` ではなく実バイト列で判定する（仕様 §2.2 SHOULD）。
    """
    try:
        with Image.open(path) as image:
            name = (image.format or "").lower()
    except OSError:
        return None
    return name if name in FORMATS else None
