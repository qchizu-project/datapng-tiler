"""タイル画像の読み書きのテスト。

とくに「無効値をアルファで表す／無効色で表す」の使い分けが、形式をまたいで
仕様どおりに成り立つかを確かめる（仕様 §2.2 / §3.2.2）。
"""

from __future__ import annotations

import numpy as np
import pytest

from datapng_tiler.imageio import (
    TileFormat,
    detect_format,
    indexed_to_image,
    load_tile,
    rgb_to_image,
    save_indexed,
)

FORMATS = [TileFormat("webp"), TileFormat("png")]
FORMAT_IDS = ["webp", "png"]


@pytest.fixture
def rgb() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)


@pytest.mark.parametrize("fmt", FORMATS, ids=FORMAT_IDS)
def test_アルファ無しの_RGB_はバイト単位で保存される(tmp_path, fmt, rgb):
    path = tmp_path / f"tile{fmt.extension}"
    fmt.save(rgb_to_image(rgb), path)

    loaded, alpha = load_tile(path)
    assert alpha is None, "アルファチャンネルを持たないことが読み手に伝わる必要がある"
    assert np.array_equal(loaded, rgb)


@pytest.mark.parametrize("fmt", FORMATS, ids=FORMAT_IDS)
def test_無効色がアルファ無しで完全に保存される(tmp_path, fmt):
    """--no-alpha 出力では invalidColor が実タイルにそのまま入っている必要がある。"""
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[1, 1] = (128, 0, 0)
    path = tmp_path / f"tile{fmt.extension}"
    fmt.save(rgb_to_image(rgb), path)

    loaded, alpha = load_tile(path)
    assert alpha is None
    assert tuple(loaded[1, 1]) == (128, 0, 0)


@pytest.mark.parametrize("fmt", FORMATS, ids=FORMAT_IDS)
def test_アルファ付きでは不透明画素の_RGB_が保存される(tmp_path, fmt, rgb):
    alpha = np.full((8, 8), 255, dtype=np.uint8)
    alpha[0, 0] = 0
    path = tmp_path / f"tile{fmt.extension}"
    fmt.save(rgb_to_image(rgb, alpha), path)

    loaded, loaded_alpha = load_tile(path)
    assert loaded_alpha is not None
    assert loaded_alpha[0, 0] == 0
    assert np.array_equal(loaded_alpha[1:], alpha[1:])
    # 不透明画素の RGB は可逆に保たれる（透明画素の RGB は仕様上意味を持たない）
    opaque = loaded_alpha == 255
    assert np.array_equal(loaded[opaque], rgb[opaque])


@pytest.mark.parametrize("fmt", FORMATS, ids=FORMAT_IDS)
def test_同じ入力から同じバイト列が出る(tmp_path, fmt, rgb):
    """タイル木の再生成で差分が出ないこと（冪等性の前提）。"""
    a = tmp_path / f"a{fmt.extension}"
    b = tmp_path / f"b{fmt.extension}"
    fmt.save(rgb_to_image(rgb), a)
    fmt.save(rgb_to_image(rgb), b)
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize("fmt", FORMATS, ids=FORMAT_IDS)
def test_形式は実バイト列から判定できる(tmp_path, fmt, rgb):
    path = tmp_path / f"tile{fmt.extension}"
    fmt.save(rgb_to_image(rgb), path)
    assert detect_format(path) == fmt.name

    # 拡張子を偽っても中身で判定する（仕様 §2.2 SHOULD）
    liar = tmp_path / "liar.png"
    liar.write_bytes(path.read_bytes())
    assert detect_format(liar) == fmt.name


# --- インデックスカラー（パレット型） -----------------------------------------------


def test_PNG_のインデックスカラーは色と透明を保つ(tmp_path):
    palette = [(0, 0, 0), (245, 245, 50), (255, 216, 0), (165, 0, 33)]
    indices = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    path = tmp_path / "tile.png"
    save_indexed(indexed_to_image(indices, palette, 0), path, TileFormat("png"), 0)

    rgb, alpha = load_tile(path)
    assert alpha is not None
    assert alpha.tolist() == [[0, 255], [255, 255]]
    assert tuple(rgb[0, 1]) == (245, 245, 50)
    assert tuple(rgb[1, 0]) == (255, 216, 0)
    assert tuple(rgb[1, 1]) == (165, 0, 33)


def test_WebP_ではインデックスカラーを_RGBA_へ展開する(tmp_path):
    palette = [(0, 0, 0), (245, 245, 50)]
    indices = np.array([[0, 1]], dtype=np.uint8)
    path = tmp_path / "tile.webp"
    save_indexed(indexed_to_image(indices, palette, 0), path, TileFormat("webp"), 0)

    rgb, alpha = load_tile(path)
    assert alpha is not None
    assert alpha.tolist() == [[0, 255]]
    assert tuple(rgb[0, 1]) == (245, 245, 50)


def test_透明を持たないインデックスカラー_PNG_はアルファ無しとして読める(tmp_path):
    palette = [(10, 20, 30), (40, 50, 60)]
    indices = np.array([[0, 1]], dtype=np.uint8)
    path = tmp_path / "tile.png"
    save_indexed(indexed_to_image(indices, palette, None), path, TileFormat("png"), None)

    rgb, alpha = load_tile(path)
    assert alpha is None
    assert tuple(rgb[0, 0]) == (10, 20, 30)
    assert tuple(rgb[0, 1]) == (40, 50, 60)


def test_パレットは256色まで():
    with pytest.raises(ValueError, match="256"):
        indexed_to_image(np.zeros((1, 1), dtype=np.uint8), [(0, 0, 0)] * 257, None)


# --- 設定の検証 ---------------------------------------------------------------------


def test_未知の形式はエラー():
    with pytest.raises(ValueError, match="未知の形式"):
        TileFormat("jpeg")


def test_圧縮設定の範囲を検証する():
    with pytest.raises(ValueError, match="webp_method"):
        TileFormat("webp", webp_method=7)
    with pytest.raises(ValueError, match="png_compress_level"):
        TileFormat("png", png_compress_level=10)


def test_WebP_は透明画素の_RGB_を保存しない(tmp_path):
    """仕様 §3.2.2 が「invalidColor で透明画素を指せない」と定める根拠を実測で固定する。

    ここが崩れる（保存されるようになる）としても仕様上の問題は起きないが、
    「なぜこの制約があるのか」を実行できる形で残しておく。
    """
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :] = (128, 0, 0)
    alpha = np.zeros((8, 8), dtype=np.uint8)  # 全画素が完全に透明
    path = tmp_path / "tile.webp"
    TileFormat("webp").save(rgb_to_image(rgb, alpha), path)

    loaded, loaded_alpha = load_tile(path)
    assert loaded_alpha is not None and not loaded_alpha.any()
    assert not (loaded == (128, 0, 0)).all(), (
        "透明画素の RGB が保存されている。仕様の前提が変わっていないか確認すること"
    )
