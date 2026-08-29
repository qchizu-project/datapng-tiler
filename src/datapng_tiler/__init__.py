"""datapng-tiler — データPNGタイルと TileJSON を生成する。

[TileJSON DataPNG Extension](https://github.com/qchizu-project/tilejson-datapng-extension)
に準拠したグリッドデータタイル（数値型・パレット型）と、その TileJSON を生成する。
"""

__version__ = "0.1.0"

# 実装対象の仕様バージョン（`schema/datapng-<SPEC_VERSION>.schema.json` と対応）。
SPEC_VERSION = "0.7.0"

__all__ = ["__version__", "SPEC_VERSION"]
