"""数値の符号化・復号（仕様 §3.2 / §3.2.1）。

純粋関数と 1 つの値オブジェクトだけを置く。ラスタ入出力・タイル幾何・画像形式には
依存しないので、ここだけを読めば「値がどうバイトになるか」が分かる。

正式なデータPNGエンコードは、R・G・B を上位バイトから並べた 24 ビット整数を
2 の補数で符号付き解釈する:

    r' = (r < 128) ? r : r - 256
    rawValue = r' × 65536 + g × 256 + b
    v = factor × rawValue + offset

`specialEncoding`（mapbox / terrarium）は固定の復号式を持ち、仕様上 `factor`・`offset`
は無視される（MUST）。本モジュールは**符号化側でも同様に無視する**——片方だけ効かせると、
TileJSON の宣言と実タイルが食い違う。

すべての符号化は `v` が raw 値の**アフィン関数**になっている。オーバービュー合成が
raw 整数のまま平均を取れる（＝量子化誤差を積まない）のはこの性質による。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# 正式なデータPNGエンコードで rawValue が取りうる範囲（24 ビット符号付き整数の全域）。
RAW_MIN = -(1 << 23)  # -8,388,608
RAW_MAX = (1 << 23) - 1  # 8,388,607
# 互換エンコード（mapbox / terrarium）は 24 ビット**符号なし**整数を使う。
URAW_MIN = 0
URAW_MAX = (1 << 24) - 1  # 16,777,215

# 無効値を表す色（アルファを持たない出力で使う既定値）。データPNG仕様の慣行に合わせる。
# 注意: この色は rawValue = -8,388,608 でもあるため、有効値がここに落ちると区別できない
# （検出は `modes/numerical.py` の責務。ここは色の定義だけを持つ）。
DEFAULT_INVALID_COLOR = (128, 0, 0)

SPECIAL_ENCODINGS = ("mapbox", "terrarium")
ON_OVERFLOW_CHOICES = ("error", "clamp", "nodata")

# suggest_factor が候補にする分解能の指数（10**e）。
_FACTOR_EXPONENTS = range(-6, 13)


class ValueRangeError(ValueError):
    """符号化できない値（24 ビット整数の表現範囲外）が含まれていた。"""


def suggest_factor(vmin: float, vmax: float) -> float:
    """値域 [vmin, vmax] が 24 ビット符号付き整数に収まる最小の 10 のべき乗を返す。

    「分解能はできるだけ細かく、ただし溢れない」桁を選ぶ。溢れたときのエラーメッセージで
    利用者に提示するために使う（自動的に factor を差し替えることはしない——分解能は
    データの意味に関わる決定であり、ツールが黙って変えてよいものではない）。
    """
    limit = max(abs(vmin), abs(vmax))
    if limit == 0.0 or not math.isfinite(limit):
        return 10.0 ** _FACTOR_EXPONENTS[0]
    for exponent in _FACTOR_EXPONENTS:
        factor = 10.0**exponent
        if limit / factor <= RAW_MAX:
            return factor
    return 10.0 ** _FACTOR_EXPONENTS[-1]


def _quantize(
    scaled: np.ndarray,
    valid: np.ndarray,
    lo: int,
    hi: int,
    on_overflow: str,
    *,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """スケール済みの実数を整数 raw 値へ丸め、表現範囲外を `on_overflow` に従って処理する。

    丸めは `np.rint`（最近接・半数は偶数側）。系統的な偏りを持たないため、オーバービューで
    カスケード平均しても平均値がずれない。

    Returns:
        (raw 値 int64, 更新後の有効マスク)
    """
    # np.rint(..., where=) はマスク付き演算のぶん遅く、無効画素の値は後で 0 に潰すので
    # 全画素まとめて丸める（無効画素の scaled は呼び出し側が有限値に均してある）。
    raw = np.rint(scaled).astype(np.int64)

    below = raw < lo
    np.logical_or(below, raw > hi, out=below)
    out_of_range = np.logical_and(below, valid, out=below)
    if out_of_range.any():
        if on_overflow == "error":
            bad = values[out_of_range]
            vmin, vmax = float(np.min(bad)), float(np.max(bad))
            raise ValueRangeError(
                f"{int(out_of_range.sum())} 個の値が符号化できる範囲を超えました"
                f"（raw 値の範囲 [{lo}, {hi}] に対し、値域 [{vmin}, {vmax}]）。"
                f" factor を {suggest_factor(vmin, vmax)} 以上にするか、"
                f" --on-overflow clamp / nodata を指定してください。"
            )
        if on_overflow == "clamp":
            np.clip(raw, lo, hi, out=raw)
        else:  # "nodata"
            valid = valid & ~out_of_range

    raw[~valid] = 0
    return raw, valid


@dataclass(frozen=True)
class NumericalEncoding:
    """数値型タイルの符号化方式（仕様 §3.2 / §3.2.1）。

    プロセス間で受け渡すため、pickle 可能な不変の値オブジェクトにしてある。

    Args:
        factor: 係数 *f*（`v = f × rawValue + offset`）。`special` 指定時は無視される
        offset: オフセット *o*。`special` 指定時は無視される
        special: `specialEncoding` の値。``None`` が正式なデータPNGエンコード
        on_overflow: 表現範囲を超えた値の扱い。``error``（既定）/ ``clamp`` / ``nodata``
    """

    factor: float = 1.0
    offset: float = 0.0
    special: str | None = None
    on_overflow: str = "error"

    # 表現できる raw 値の範囲（__post_init__ で埋める）
    raw_min: int = field(init=False, default=RAW_MIN)
    raw_max: int = field(init=False, default=RAW_MAX)

    def __post_init__(self) -> None:
        if self.special is not None and self.special not in SPECIAL_ENCODINGS:
            raise ValueError(
                f"未知の specialEncoding: {self.special!r}"
                f"（利用可能: {', '.join(SPECIAL_ENCODINGS)}）"
            )
        if self.on_overflow not in ON_OVERFLOW_CHOICES:
            raise ValueError(
                f"未知の on_overflow: {self.on_overflow!r}"
                f"（利用可能: {', '.join(ON_OVERFLOW_CHOICES)}）"
            )
        if self.special is None:
            if not (self.factor > 0) or not math.isfinite(self.factor):
                raise ValueError(f"factor は正の有限値でなければなりません: {self.factor!r}")
            if not math.isfinite(self.offset):
                raise ValueError(f"offset は有限値でなければなりません: {self.offset!r}")
        else:
            object.__setattr__(self, "raw_min", URAW_MIN)
            object.__setattr__(self, "raw_max", URAW_MAX)

    # --- 値 ↔ raw ------------------------------------------------------------------

    def _to_scaled(self, values: np.ndarray) -> np.ndarray:
        """値を raw 値のスケール（丸める前の実数）へ変換する。

        **必ず float64 で計算する。** NumPy は Python の float を「弱い」型として扱うため、
        float32 の配列に素直に演算子を使うと float32 のまま計算される。factor が小さい
        （= raw 値が大きい）ときに float32 の仮数では丸めが 1 整数ぶんずれることがあり、
        入力の dtype によって生成タイルが変わってしまう。`dtype=np.float64` を明示し、
        以降は同じ配列上で計算して余分なコピーを作らない。
        """
        values = np.asarray(values)
        if self.special == "mapbox":
            scaled = np.add(values, 10000.0, dtype=np.float64)
            return np.divide(scaled, 0.1, out=scaled)
        if self.special == "terrarium":
            scaled = np.add(values, 32768.0, dtype=np.float64)
            return np.multiply(scaled, 256.0, out=scaled)
        scaled = np.subtract(values, self.offset, dtype=np.float64)
        return np.divide(scaled, self.factor, out=scaled)

    def raw_to_values(self, raw: np.ndarray) -> np.ndarray:
        """raw 値を実数値へ復号する（仕様の各復号式）。"""
        raw = raw.astype(np.float64)
        if self.special == "mapbox":
            return -10000.0 + raw * 0.1
        if self.special == "terrarium":
            return raw / 256.0 - 32768.0
        return self.factor * raw + self.offset

    # --- raw ↔ RGB -----------------------------------------------------------------

    @staticmethod
    def _raw_to_rgb(raw: np.ndarray) -> np.ndarray:
        """raw 値（符号付き / 符号なしのどちらでも）を RGB バイトへ分解する。

        下位 24 ビットを取り出すため、符号付きの負値も 2 の補数表現のまま正しく並ぶ。

        ビッグエンディアンの uint32 として並べ替え、バイト列として見る。こうすると
        シフトとマスクを 3 回ずつ繰り返さずに済み、タイル数が多いときの差が大きい
        （バイト順は `>u4` が保証するので、実行環境のエンディアンには依存しない）。
        """
        packed = (np.asarray(raw) & 0xFFFFFF).astype(">u4")
        # バイトは [0, R, G, B] の順に並ぶ。先頭の 0 を落とす
        return np.ascontiguousarray(packed.view(np.uint8).reshape(raw.shape + (4,))[..., 1:])

    @staticmethod
    def raw_to_rgba(raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """raw 値と有効マスクから RGBA バイトを作る（無効画素は完全透明・RGB は 0）。

        `_raw_to_rgb` と同じバイトビューの手口で、アルファまで 1 回で並べる。
        無効画素の RGB を 0 に潰しているのは、透明画素の RGB が仕様上意味を持たない
        （WebP の可逆圧縮でも保存されない）ため。
        """
        packed = ((np.asarray(raw) & 0xFFFFFF) << 8).astype(">u4")
        packed[~valid] = 0
        packed |= np.where(valid, np.uint32(0xFF), np.uint32(0))
        # バイトは [R, G, B, A] の順に並ぶ
        return np.ascontiguousarray(packed.view(np.uint8).reshape(raw.shape + (4,)))

    def rgb_to_raw(self, rgb: np.ndarray) -> np.ndarray:
        """RGB バイトを raw 値へ戻す（正式エンコードは符号付き、互換は符号なし）。"""
        r = rgb[..., 0].astype(np.int32)
        g = rgb[..., 1].astype(np.int32)
        b = rgb[..., 2].astype(np.int32)
        if self.special is None:
            # 仕様 §3.2: r >= 128 は符号ビットが立っている（2 の補数）
            r = np.where(r < 128, r, r - 256)
            return r * 65536 + g * 256 + b
        return r * 65536 + g * 256 + b

    def raw_to_rgb(self, raw: np.ndarray) -> np.ndarray:
        """raw 値を RGB バイトへ分解する（`rgb_to_raw` の逆）。"""
        return self._raw_to_rgb(raw)

    # --- 公開 API ------------------------------------------------------------------

    def encode_raw(
        self, values: np.ndarray, valid: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """値の配列を raw 整数へ量子化する（バイトへ並べる前段）。

        RGB と RGBA のどちらに並べるかは、無効値の表し方を知っている呼び出し側が決める。
        ここで RGB を作ってから塗り直すと、配列を何度も走査することになる。

        Returns:
            (raw 値 int64, 有効マスク)。無効画素の raw は 0。``on_overflow="nodata"`` では
            有効マスクが更新されるため、**戻り値のマスクを使うこと**。
        """
        values = np.asarray(values)
        if valid is None:
            valid = np.ones(values.shape, dtype=bool)
        else:
            valid = np.asarray(valid, dtype=bool)
        if np.issubdtype(values.dtype, np.floating):
            valid = valid & ~np.isnan(values)

        # 無効画素は 0 に均してから変換する。NaN を丸めると未定義の整数になるうえ、
        # 範囲外の値が紛れていても無効画素は検査対象から外したいため。
        # 変換式は Python の float と混ざるので、float32 入力でも float64 に上がる。
        scaled = self._to_scaled(np.where(valid, values, 0))
        return _quantize(scaled, valid, self.raw_min, self.raw_max, self.on_overflow, values=values)

    def encode(
        self, values: np.ndarray, valid: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """値の配列を RGB バイトへ符号化する。

        Args:
            values: 値の配列（任意形状）。NaN は無効値として扱う
            valid: 有効画素マスク。``None`` なら NaN 以外をすべて有効とみなす

        Returns:
            (RGB 配列 `values.shape + (3,)` uint8, 有効マスク)。無効画素の RGB は 0 で埋める
            （呼び出し側がアルファ 0 または無効色を載せる）。``on_overflow="nodata"`` では
            有効マスクが更新されるため、**戻り値のマスクを使うこと**。
        """
        raw, valid = self.encode_raw(values, valid)
        return self._raw_to_rgb(raw), valid

    def decode(self, rgb: np.ndarray) -> np.ndarray:
        """RGB バイトを値へ復号する（無効画素の判定は呼び出し側の責務）。"""
        return self.raw_to_values(self.rgb_to_raw(rgb))

    def value_range(self) -> tuple[float, float]:
        """この符号化で表現できる値域（両端含む）。"""
        lo = float(self.raw_to_values(np.array(self.raw_min))[()])
        hi = float(self.raw_to_values(np.array(self.raw_max))[()])
        return (lo, hi) if lo <= hi else (hi, lo)

    def datapng_fields(self) -> dict[str, object]:
        """TileJSON の `datapng` に載せるキーを返す（仕様の既定値と同じものは出さない）。"""
        if self.special is not None:
            return {"specialEncoding": self.special}
        fields: dict[str, object] = {}
        if self.factor != 1.0:
            fields["factor"] = self.factor
        if self.offset != 0.0:
            fields["offset"] = self.offset
        return fields
