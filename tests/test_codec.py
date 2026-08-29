"""数値エンコードの符号化・復号のテスト。

**要点**: 実装のデコーダで往復させても「実装が自分の仕様に従っている」ことしか言えない。
そこでこのファイルには、仕様書
（tilejson-datapng-extension §3.2 / §3.2.1）の式を**文字どおり写した独立デコーダ**を置き、
それで実装の出力を検証する。ベクトル化も最適化もしない（式との対応を目で追えることが優先）。
"""

import math

import numpy as np
import pytest

from datapng_tiler.codec import (
    RAW_MAX,
    RAW_MIN,
    NumericalEncoding,
    ValueRangeError,
    suggest_factor,
)

# --- 仕様書から書き写した独立デコーダ（実装を参照しない） ---------------------------


def spec_decode_datapng(rgb: np.ndarray, factor: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """仕様 §3.2「正式なデータPNGエンコード」の復号式をそのまま写したもの。

    r' = (r < 128) ? r : r - 256
    rawValue = r' × 65536 + g × 256 + b
    v = factor × rawValue + offset
    """
    h, w = rgb.shape[:2]
    out = np.empty((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            r, g, b = int(rgb[i, j, 0]), int(rgb[i, j, 1]), int(rgb[i, j, 2])
            r_prime = r if r < 128 else r - 256
            raw_value = r_prime * 65536 + g * 256 + b
            out[i, j] = factor * raw_value + offset
    return out


def spec_decode_mapbox(rgb: np.ndarray) -> np.ndarray:
    """仕様 §3.2.1「Mapbox Terrain-RGB 互換」: v = -10000 + (r*65536 + g*256 + b) * 0.1"""
    h, w = rgb.shape[:2]
    out = np.empty((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            r, g, b = int(rgb[i, j, 0]), int(rgb[i, j, 1]), int(rgb[i, j, 2])
            out[i, j] = -10000 + (r * 65536 + g * 256 + b) * 0.1
    return out


def spec_decode_terrarium(rgb: np.ndarray) -> np.ndarray:
    """仕様 §3.2.1「Mapzen/Terrarium 互換」: v = (r*256 + g + b/256) - 32768"""
    h, w = rgb.shape[:2]
    out = np.empty((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            r, g, b = int(rgb[i, j, 0]), int(rgb[i, j, 1]), int(rgb[i, j, 2])
            out[i, j] = (r * 256 + g + b / 256) - 32768
    return out


# --- 正式なデータPNGエンコード ------------------------------------------------------


def test_エンコード結果が仕様の復号式と一致する():
    enc = NumericalEncoding(factor=0.01, offset=0.0)
    values = np.array([[0.0, 1.23, -45.67], [8000.0, -500.0, 0.01]], dtype=np.float64)

    rgb, _ = enc.encode(values)
    assert rgb.dtype == np.uint8
    assert rgb.shape == (2, 3, 3)

    got = spec_decode_datapng(rgb, factor=0.01, offset=0.0)
    # 量子化されるので、半幅（factor/2）以内で一致すればよい
    assert np.all(np.abs(got - values) <= 0.01 / 2 + 1e-9)


def test_offset_が符号化と復号の両方に効く():
    enc = NumericalEncoding(factor=0.5, offset=1000.0)
    values = np.array([[1000.0, 1000.5, 999.5, -7000.0]], dtype=np.float64)

    rgb, _ = enc.encode(values)
    got = spec_decode_datapng(rgb, factor=0.5, offset=1000.0)
    assert np.allclose(got, values)

    # 実装のデコーダも同じ値を返す
    assert np.allclose(enc.decode(rgb), values)


def test_offset_を無視すると別の値になる():
    """offset が本当に効いていることの裏取り（qchizu-tools は offset を符号化に反映しない）。"""
    enc = NumericalEncoding(factor=1.0, offset=1000.0)
    rgb, _ = enc.encode(np.array([[1000.0]]))
    # offset を無視して復号すると 0 になる（= offset ぶんずれる）
    assert spec_decode_datapng(rgb, factor=1.0, offset=0.0)[0, 0] == 0.0
    assert spec_decode_datapng(rgb, factor=1.0, offset=1000.0)[0, 0] == 1000.0


@pytest.mark.parametrize("raw", [RAW_MIN, RAW_MIN + 1, -1, 0, 1, RAW_MAX - 1, RAW_MAX])
def test_24ビット符号付き整数の全域を往復できる(raw):
    """rawValue は -8,388,608 〜 8,388,607 の全域を取りうる（仕様 §3.2）。"""
    enc = NumericalEncoding(factor=1.0, offset=0.0)
    rgb, valid = enc.encode(np.array([[float(raw)]]))
    assert valid.all()
    assert spec_decode_datapng(rgb)[0, 0] == float(raw)


def test_範囲外の値は既定でエラーになる():
    enc = NumericalEncoding(factor=1.0)
    with pytest.raises(ValueRangeError) as excinfo:
        enc.encode(np.array([[float(RAW_MAX + 1)]]))
    # 診断に必要な情報（実際の値と推奨 factor）がメッセージに含まれる
    assert "8388608" in str(excinfo.value)
    assert "factor" in str(excinfo.value)

    with pytest.raises(ValueRangeError):
        enc.encode(np.array([[float(RAW_MIN - 1)]]))


def test_範囲外の値をクランプできる():
    enc = NumericalEncoding(factor=1.0, on_overflow="clamp")
    rgb, valid = enc.encode(np.array([[1e9, -1e9]]))
    assert valid.all()
    decoded = spec_decode_datapng(rgb)
    assert decoded[0, 0] == float(RAW_MAX)
    assert decoded[0, 1] == float(RAW_MIN)


def test_範囲外の値を無効値にできる():
    enc = NumericalEncoding(factor=1.0, on_overflow="nodata")
    _, valid = enc.encode(np.array([[1e9, 0.0, -1e9]]))
    assert valid.tolist() == [[False, True, False]]


def test_無効画素は符号化の対象外():
    """無効画素の RGB は 0 で埋め、値の範囲検査にも掛けない（範囲外でもエラーにしない）。"""
    enc = NumericalEncoding(factor=1.0)
    values = np.array([[1.0, 1e9]], dtype=np.float64)
    valid = np.array([[True, False]])
    rgb, out_valid = enc.encode(values, valid=valid)
    assert out_valid.tolist() == [[True, False]]
    assert rgb[0, 1].tolist() == [0, 0, 0]


def test_nan_は無効画素として扱われる():
    enc = NumericalEncoding(factor=1.0)
    rgb, valid = enc.encode(np.array([[1.0, math.nan]]))
    assert valid.tolist() == [[True, False]]
    assert rgb[0, 1].tolist() == [0, 0, 0]


def test_factor_が0以下ならエラー():
    with pytest.raises(ValueError, match="factor"):
        NumericalEncoding(factor=0.0)
    with pytest.raises(ValueError, match="factor"):
        NumericalEncoding(factor=-0.01)


def test_未知の特殊エンコードはエラー():
    with pytest.raises(ValueError, match="specialEncoding"):
        NumericalEncoding(special="unknown")


# --- 特殊なエンコード（互換） --------------------------------------------------------


def test_mapbox_エンコードが仕様の復号式と一致する():
    enc = NumericalEncoding(special="mapbox")
    values = np.array([[-10000.0, 0.0, 1234.5, 8848.1]], dtype=np.float64)
    rgb, _ = enc.encode(values)
    got = spec_decode_mapbox(rgb)
    assert np.all(np.abs(got - values) <= 0.05 + 1e-9)  # 分解能 0.1m の半幅
    assert np.allclose(enc.decode(rgb), got)


def test_terrarium_エンコードが仕様の復号式と一致する():
    enc = NumericalEncoding(special="terrarium")
    values = np.array([[-32768.0, 0.0, 1234.5, 8848.0]], dtype=np.float64)
    rgb, _ = enc.encode(values)
    got = spec_decode_terrarium(rgb)
    assert np.all(np.abs(got - values) <= (1 / 256) / 2 + 1e-9)
    assert np.allclose(enc.decode(rgb), got)


def test_特殊エンコードは_factor_と_offset_を無視する():
    """仕様 §3.2.1: specialEncoding 指定時、クライアントは factor・offset を無視する（MUST）。

    したがって符号化側も factor・offset を反映してはならない。
    """
    plain = NumericalEncoding(special="mapbox")
    with_params = NumericalEncoding(special="mapbox", factor=0.01, offset=500.0)
    values = np.array([[100.0, 200.0]], dtype=np.float64)
    assert np.array_equal(plain.encode(values)[0], with_params.encode(values)[0])


def test_特殊エンコードの表現範囲外はエラー():
    enc = NumericalEncoding(special="terrarium")
    with pytest.raises(ValueRangeError):
        enc.encode(np.array([[-32769.0]]))


# --- TileJSON へ載せるフィールド -----------------------------------------------------


def test_datapng_フィールドは正式エンコードで_factor_と_offset_を出す():
    enc = NumericalEncoding(factor=0.01, offset=0.0)
    assert enc.datapng_fields() == {"factor": 0.01}

    enc = NumericalEncoding(factor=0.5, offset=100.0)
    assert enc.datapng_fields() == {"factor": 0.5, "offset": 100.0}


def test_datapng_フィールドは特殊エンコードで_factor_と_offset_を出さない():
    enc = NumericalEncoding(special="mapbox", factor=0.01, offset=100.0)
    assert enc.datapng_fields() == {"specialEncoding": "mapbox"}


# --- 補助 ---------------------------------------------------------------------------


def test_推奨_factor_はデータ範囲が収まる最小の桁を返す():
    # ±8000 は factor=0.001 で収まる（8,000,000 ≤ 8,388,607）
    assert suggest_factor(-8000.0, 8000.0) == pytest.approx(0.001)
    # 標高 -500〜9000m は 0.001 だと 9,000,000 で溢れるので 0.01
    assert suggest_factor(-500.0, 9000.0) == pytest.approx(0.01)
    # 極端に大きい範囲でも溢れない桁まで上げる
    assert suggest_factor(0.0, 1e12) == pytest.approx(1e6)


def test_入力の_dtype_で結果が変わらない():
    """float32 の配列でも量子化は float64 で行う（NumPy の弱い型付けへの対策）。

    素直に `values / factor` と書くと、float32 の配列 + Python の float は float32 の
    まま計算される。factor が小さいと仮数が足りず、丸めが 1 整数ぶんずれて
    「同じデータなのに入力の dtype で生成タイルが変わる」ことになる。
    """
    enc = NumericalEncoding(factor=0.001)
    # float32 で表せる値だけを使い、float64 へは無損失に広げる
    # （linspace を後から丸めると入力そのものが変わってしまう）
    values32 = np.linspace(-8000.0, 8000.0, 4096).astype(np.float32)
    values64 = values32.astype(np.float64)
    as32, _ = enc.encode(values32)
    as64, _ = enc.encode(values64)
    assert np.array_equal(as32, as64)


def test_encode_raw_と_encode_は同じ結果になる():
    enc = NumericalEncoding(factor=0.01, offset=5.0)
    values = np.array([[1.0, -2.5], [300.0, 0.0]])
    valid = np.array([[True, True], [True, False]])
    raw, raw_valid = enc.encode_raw(values, valid=valid)
    rgb, rgb_valid = enc.encode(values, valid=valid)
    assert np.array_equal(enc.raw_to_rgb(raw), rgb)
    assert np.array_equal(raw_valid, rgb_valid)


def test_raw_to_rgba_は無効画素を完全透明にする():
    enc = NumericalEncoding(factor=1.0)
    raw = np.array([[1234, -5678]], dtype=np.int64)
    valid = np.array([[True, False]])
    rgba = enc.raw_to_rgba(raw, valid)
    assert rgba.shape == (1, 2, 4)
    assert rgba[0, 0, 3] == 255
    assert rgba[0, 1, 3] == 0
    # 透明画素の RGB は意味を持たないので 0 に潰す
    assert tuple(rgba[0, 1, :3]) == (0, 0, 0)
    assert spec_decode_datapng(rgba[..., :3])[0, 0] == 1234


@pytest.mark.parametrize("bad", [math.inf, -math.inf])
def test_無限大は無効画素として扱われる(bad):
    """inf は「大きすぎる値」ではなく表現できない値。丸めて整数化すると未定義になる。"""
    enc = NumericalEncoding(factor=1.0)
    rgb, valid = enc.encode(np.array([[1.0, bad]]))
    assert valid.tolist() == [[True, False]]
    assert rgb[0, 1].tolist() == [0, 0, 0]
