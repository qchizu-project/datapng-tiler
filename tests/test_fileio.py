"""アトミック書き込みとタイルパスのテスト。"""

from __future__ import annotations

import pytest

from datapng_tiler.fileio import (
    atomic_write,
    is_complete,
    remove_if_exists,
    sweep_temp_files,
    tile_path,
)


def test_正常終了で本来のパスへ差し替わる(tmp_path):
    target = tmp_path / "a" / "b" / "tile.webp"
    with atomic_write(target) as tmp:
        tmp.write_bytes(b"data")
    assert target.read_bytes() == b"data"
    assert not tmp.exists()


def test_例外時は本来のパスに触れない(tmp_path):
    target = tmp_path / "tile.webp"
    target.write_bytes(b"original")

    with pytest.raises(RuntimeError):
        with atomic_write(target) as tmp:
            tmp.write_bytes(b"partial")
            raise RuntimeError("途中で失敗")

    assert target.read_bytes() == b"original", "失敗した書き込みが既存を壊してはならない"
    assert not tmp.exists()


def test_存在しないファイルの削除は_False(tmp_path):
    assert remove_if_exists(tmp_path / "nope") is False
    path = tmp_path / "yes"
    path.write_bytes(b"x")
    assert remove_if_exists(path) is True
    assert not path.exists()


def test_0バイトのタイルは未生成として扱う(tmp_path):
    """中断で残った空ファイルを生成済みと誤認すると、レジュームが永久に飛ばす。"""
    empty = tmp_path / "empty.webp"
    empty.touch()
    assert is_complete(empty) is False

    filled = tmp_path / "filled.webp"
    filled.write_bytes(b"x")
    assert is_complete(filled) is True

    assert is_complete(tmp_path / "missing.webp") is False


def test_タイルパスの構成():
    assert (
        tile_path("/out", 14, 14552, 6451, ".webp").as_posix().endswith("/out/14/14552/6451.webp")
    )


def test_残った一時ファイルを掃除する(tmp_path):
    (tmp_path / "12" / "34").mkdir(parents=True)
    (tmp_path / "12" / "34" / "56.webp.tmp").write_bytes(b"x")
    (tmp_path / "12" / "34" / "57.webp").write_bytes(b"x")

    assert sweep_temp_files(tmp_path) == 1
    assert not (tmp_path / "12" / "34" / "56.webp.tmp").exists()
    assert (tmp_path / "12" / "34" / "57.webp").exists()
