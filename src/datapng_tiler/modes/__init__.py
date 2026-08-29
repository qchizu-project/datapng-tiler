"""タイル種別（mode）。

共通のスキャフォールド（再投影・タイル幾何・レジューム・base→overview のループ）は
`base.TileMode` と `engine` が持ち、種別ごとに差し替えるのは「どう読み、どう符号化し、
どう束ねるか」だけにしてある。
"""

from datapng_tiler.modes.base import ChildSlot, TileMode
from datapng_tiler.modes.numerical import NumericalMode
from datapng_tiler.modes.palette import PaletteMode

__all__ = ["ChildSlot", "NumericalMode", "PaletteMode", "TileMode"]
