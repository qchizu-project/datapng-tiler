#!/usr/bin/env python3
"""同梱している JSON Schema が仕様リポジトリの配布物と一致するかを確認する。

仕様（tilejson-datapng-extension）が更新されたのに実装側のスキーマが古いまま、を防ぐ。
ネットワークが使えない環境（オフラインの CI・ローカル）ではスキップして 0 で終了する
——このチェックは「食い違いを見つける」ためのもので、取得できないこと自体を失敗に
したいわけではない（取得できなければ食い違いも主張できない）。
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from datapng_tiler import SPEC_VERSION

RAW_URL = (
    "https://raw.githubusercontent.com/qchizu-project/tilejson-datapng-extension"
    f"/main/schema/datapng-{SPEC_VERSION}.schema.json"
)
LOCAL = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "datapng_tiler"
    / "schema"
    / f"datapng-{SPEC_VERSION}.schema.json"
)


def main() -> int:
    local = json.loads(LOCAL.read_text(encoding="utf-8"))
    try:
        with urllib.request.urlopen(RAW_URL, timeout=30) as res:  # noqa: S310 (固定の https URL)
            remote = json.loads(res.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"スキップ: 仕様リポジトリのスキーマを取得できませんでした（{exc}）")
        return 0

    if local == remote:
        print(f"一致: datapng-{SPEC_VERSION}.schema.json")
        return 0

    print(f"不一致: 同梱スキーマが {RAW_URL} と食い違っています", file=sys.stderr)
    print("  仕様の更新を取り込むか、SPEC_VERSION を上げてください", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
