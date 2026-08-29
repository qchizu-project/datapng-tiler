"""ファイル書き込みの下回り（アトミック書き込み・タイルパス）。"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_write(path: Path | str) -> Iterator[Path]:
    """一時ファイルのパスを yield し、正常終了時のみ本来のパスへ差し替える。

    途中で例外が起きたら一時ファイルを消し、本来のパスには一切触れない。中断しても
    「サイズは 0 でないが中身が途中まで」というタイルが残らないので、レジュームが
    「ファイルがあれば生成済み」という単純な判定で済む。

    各タイル座標は 1 プロセスが 1 回だけ書く（バッチは座標で重ならないよう分割する）ため、
    固定の ``.tmp`` 名で衝突しない。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        yield tmp
    except BaseException:
        # 後片付けの失敗で本来の例外を隠さない
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    else:
        os.replace(tmp, path)


def remove_if_exists(path: Path | str) -> bool:
    """ファイルがあれば削除して ``True`` を返す。無ければ ``False``。

    ``exists()`` で確かめてから消すと確認と削除の間に競合が入りうるので、
    直接 ``unlink()`` して ``FileNotFoundError`` を捕まえる。
    """
    try:
        Path(path).unlink()
        return True
    except FileNotFoundError:
        return False


def tile_path(root: Path | str, zoom: int, x: int, y: int, extension: str) -> Path:
    """``root/{z}/{x}/{y}.<ext>`` を返す。"""
    return Path(root) / str(zoom) / str(x) / f"{y}{extension}"


def is_complete(path: Path) -> bool:
    """タイルが生成済み（存在しサイズ > 0）か。

    0 バイトのファイルは未生成として扱う。中断時に残った空ファイルを「生成済み」と
    誤認すると、レジュームがそのタイルを永久に飛ばしてしまう。
    """
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def sweep_temp_files(root: Path | str) -> int:
    """タイル木に残った ``*.tmp`` を削除して件数を返す。

    プロセスが例外を出さずに kill された場合、`atomic_write` の後片付けが走らず
    一時ファイルが残る。次回実行の前に掃除する。
    """
    removed = 0
    for tmp in Path(root).rglob("*.tmp"):
        if remove_if_exists(tmp):
            removed += 1
    return removed
