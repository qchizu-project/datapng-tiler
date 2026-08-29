"""生成物を目視確認するためのプレビュー HTML。

タイル木のルートに置いてブラウザで開けば、生成したタイルをそのまま確認できる。
単に画像として並べるのではなく、**TileJSON の宣言どおりに復号して値を表示する**
——「絵としては出ているが値が違う」を見つけられるようにするため。

- 数値型: カーソル位置の画素を復号し、値と単位を表示する
- パレット型: 凡例を並べ、カーソル位置の色に対応する項目を強調する

**背景地図は既定で無し。** 地理院タイルや OpenStreetMap を既定にすると、このツールを
使うすべての人に第三者サービスの利用規約を負わせることになる。必要な人が
`--basemap gsi|osm`（または `--basemap-url`）で明示的に選ぶ。

Leaflet は CDN から SRI 付きで読み込む（同梱しない）。値の読み取りはタイル画像を
canvas に描いて画素を取るため、**タイルが HTML と同一オリジンにある**必要がある
（別ドメインのタイルは canvas が汚染され読み取れない。地図の表示自体はできる）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LEAFLET_VERSION = "1.9.4"
LEAFLET_CSS = f"https://unpkg.com/leaflet@{LEAFLET_VERSION}/dist/leaflet.css"
LEAFLET_JS = f"https://unpkg.com/leaflet@{LEAFLET_VERSION}/dist/leaflet.js"
# 実ファイルから計算した SRI（CDN が差し替えられても改ざんを検知できる）
LEAFLET_CSS_SRI = "sha384-sHL9NAb7lN7rfvG5lfHpm643Xkcjzp4jFvuavGOndn6pjVqS6ny56CAt3nsEVT4H"
LEAFLET_JS_SRI = "sha384-cxOPjt7s7Iz04uaHJceBmS+qpjv2JkIHNVcuOrM+YHwZOmJGBXI00mdUXEq65HTH"

# 選べる背景地図。**既定は none**（§モジュール docstring の理由）。
# 選んだ人が規約を確認できるよう、URL と併せて出典を持つ。
BASEMAPS: dict[str, dict[str, str]] = {
    "gsi": {
        "url": "https://cyberjapandata.gsi.go.jp/xyz/pale/{z}/{x}/{y}.png",
        "attribution": (
            '<a href="https://maps.gsi.go.jp/development/ichiran.html" '
            'target="_blank" rel="noopener">国土地理院</a>'
        ),
        "terms": "https://maps.gsi.go.jp/development/ichiran.html",
        "max_zoom": "18",
    },
    "osm": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": (
            '&copy; <a href="https://www.openstreetmap.org/copyright" '
            'target="_blank" rel="noopener">OpenStreetMap</a> contributors'
        ),
        "terms": "https://operations.osmfoundation.org/policies/tiles/",
        "max_zoom": "19",
    },
}
BASEMAP_CHOICES = ("none", *BASEMAPS)

# maxzoom を超えて引き伸ばして表示できる段数
_OVERZOOM = 1

_VIEWER_JS = """
const tj = CONFIG.tilejson;
const dp = tj.datapng || {};
const bounds = tj.bounds;
const leafletBounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]];

// Leaflet の地図ズームは 256px タイルを前提にしている。tileSize が 256 でない場合、
// 地図ズーム Z に対応するタイル z は Z - scale（scale = log2(tileSize/256)）になる。
// これを zoomOffset に反映しないと、1 段ずれたタイルを取りに行って何も表示されない。
const scale = Math.log2(tj.tileSize / 256);
const mapMinZoom = tj.minzoom + scale;
const mapMaxZoom = tj.maxzoom + scale;

const map = L.map('map', { minZoom: mapMinZoom, maxZoom: mapMaxZoom + CONFIG.overzoom });

if (CONFIG.basemap) {
  L.tileLayer(CONFIG.basemap.url, {
    maxNativeZoom: Number(CONFIG.basemap.max_zoom),
    maxZoom: mapMaxZoom + CONFIG.overzoom,
    attribution: CONFIG.basemap.attribution,
  }).addTo(map);
}

// 生成タイル。読み取りのために各タイルの画素を控えておく。
const pixels = new Map();
const DataLayer = L.TileLayer.extend({
  createTile: function (coords, done) {
    const tile = document.createElement('img');
    tile.crossOrigin = 'anonymous';
    tile.alt = '';
    tile.onload = () => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = tile.naturalWidth;
        canvas.height = tile.naturalHeight;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(tile, 0, 0);
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
        pixels.set(`${coords.z}/${coords.x}/${coords.y}`, data);
      } catch (err) {
        // 別オリジンのタイルは canvas が汚染されて読み取れない（表示は可能）
        pixels.set(`${coords.z}/${coords.x}/${coords.y}`, null);
      }
      done(null, tile);
    };
    tile.onerror = () => {
      // createTile を差し替えると Leaflet 既定のエラー処理も置き換わるため、
      // errorTileUrl（透明画像）への差し替えを自前で行う。
      // やらないと、生成されなかった空タイルの位置に壊れた画像の枠が出る。
      const fallback = this.options.errorTileUrl;
      if (fallback && tile.src !== fallback) {
        tile.src = fallback;
        return;
      }
      done(null, tile);
    };
    tile.src = this.getTileUrl(coords);
    return tile;
  },
});

new DataLayer(tj.tiles[0], {
  minNativeZoom: mapMinZoom,
  maxNativeZoom: mapMaxZoom,
  tileSize: tj.tileSize,
  zoomOffset: -scale,
  bounds: leafletBounds,
  attribution: tj.attribution || '',
  // 空タイル（生成されなかった領域）で壊れた画像アイコンを出さない
  errorTileUrl:
    'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
}).addTo(map);

map.fitBounds(leafletBounds);

// --- 仕様どおりの復号（tilejson-datapng-extension §3.2） ---
function decodeNumerical(r, g, b) {
  switch (dp.specialEncoding) {
    case 'mapbox':
      return -10000 + (r * 65536 + g * 256 + b) * 0.1;
    case 'terrarium':
      return (r * 256 + g + b / 256) - 32768;
    default: {
      const rp = r < 128 ? r : r - 256;
      const raw = rp * 65536 + g * 256 + b;
      return (dp.factor === undefined ? 1 : dp.factor) * raw +
             (dp.offset === undefined ? 0 : dp.offset);
    }
  }
}

function sample(latlng) {
  // タイル座標は「地図ズーム」で数える（createTile が受け取る coords と同じ基準）。
  const zoom = Math.min(Math.round(map.getZoom()), mapMaxZoom);
  const size = tj.tileSize;
  const projected = map.project(latlng, zoom);
  const point = projected.divideBy(size).floor();
  const image = pixels.get(`${zoom}/${point.x}/${point.y}`);
  if (!image) return null;

  const inTile = projected.subtract(point.multiplyBy(size));
  const col = Math.min(size - 1, Math.max(0, Math.floor(inTile.x)));
  const row = Math.min(size - 1, Math.max(0, Math.floor(inTile.y)));
  const offset = (row * image.width + col) * 4;
  return {
    r: image.data[offset],
    g: image.data[offset + 1],
    b: image.data[offset + 2],
    a: image.data[offset + 3],
  };
}

// --- 読み取りパネル ---
const readout = L.control({ position: 'bottomleft' });
readout.onAdd = function () {
  const div = L.DomUtil.create('div', 'readout');
  div.innerHTML = '<span class="hint">地図上にカーソルを合わせると値を表示します</span>';
  this._div = div;
  return div;
};
readout.addTo(map);

function invalidByAlpha(px) {
  // 仕様 §3.2.2: アルファチャンネルを持つタイルはアルファ 0 のみが無効
  return px.a === 0;
}

function invalidByColor(px) {
  const c = dp.invalidColor;
  return Array.isArray(c) && px.r === c[0] && px.g === c[1] && px.b === c[2];
}

map.on('mousemove', (event) => {
  const px = sample(event.latlng);
  if (!px) {
    readout._div.innerHTML =
      '<span class="hint">タイルなし（別オリジンだと読み取れません）</span>';
    return;
  }
  // canvas は常にアルファを返すため「タイルがアルファチャンネルを持つか」は区別できない。
  // ただし仕様 §3.2.2 により両者は排他なので、どちらの判定も安全に併用できる
  // （アルファ無しのタイルは canvas 上で常に a=255、invalidColor 付きのタイルは
  //  アルファチャンネルを持たない）。
  const invalid = invalidByAlpha(px) || invalidByColor(px);
  if (invalid) {
    readout._div.innerHTML = '<b>無効値</b>';
    highlightLegend(null);
    return;
  }
  if (dp.type === 'palette') {
    const item = legendItems.find((i) => i.r === px.r && i.g === px.g && i.b === px.b);
    // description も必ずエスケープする。legend は外部 URL から取ってくることがあり、
    // その内容をこのページの作者が書いたとは限らない。
    readout._div.innerHTML = item
      ? `<b>${escapeHtml(item.title)}</b>` +
        (item.description ? `<br>${escapeHtml(item.description)}` : '')
      : `<span class="hint">凡例に無い色 (${px.r}, ${px.g}, ${px.b})</span>`;
    highlightLegend(item || null);
    return;
  }
  const value = decodeNumerical(px.r, px.g, px.b);
  const unit = dp.unit ? ` ${escapeHtml(dp.unit)}` : '';
  readout._div.innerHTML =
    `<b>${value.toFixed(3)}${unit}</b><br>` +
    `<span class="hint">RGB (${px.r}, ${px.g}, ${px.b})</span>`;
});

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

// --- 凡例（パレット型） ---
let legendItems = [];
function highlightLegend(active) {
  document.querySelectorAll('.legend li').forEach((li, index) => {
    li.classList.toggle('active', active !== null && legendItems[index] === active);
  });
}

if (dp.type === 'palette') {
  const legend = dp.legend;
  if (typeof legend === 'string') {
    fetch(legend).then((res) => res.json()).then(renderLegend).catch(() => {});
  } else if (legend) {
    renderLegend(legend);
  }
}

function renderLegend(legend) {
  legendItems = legend.items || [];
  const control = L.control({ position: 'topright' });
  control.onAdd = function () {
    const div = L.DomUtil.create('div', 'legend');
    const title = legend.title ? `<h2>${escapeHtml(legend.title)}</h2>` : '';
    const rows = legendItems.map((item) =>
      `<li><span class="swatch" style="background: rgb(${item.r},${item.g},${item.b})"></span>` +
      `${escapeHtml(item.title)}</li>`).join('');
    div.innerHTML = `${title}<ul>${rows}</ul>`;
    L.DomEvent.disableClickPropagation(div);
    return div;
  };
  control.addTo(map);
}
"""

_STYLE = """
  html, body { margin: 0; height: 100%; font-family: system-ui, sans-serif; }
  #map { height: 100%; background: #1a1a1a; }
  .readout, .legend {
    background: rgba(255, 255, 255, 0.92);
    padding: 8px 10px;
    border-radius: 4px;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
    font-size: 13px;
    line-height: 1.5;
  }
  .readout b { font-size: 15px; }
  .readout .hint, .legend .hint { color: #555; font-size: 12px; }
  .legend { max-height: 60vh; overflow-y: auto; }
  .legend h2 { margin: 0 0 6px; font-size: 13px; }
  .legend ul { list-style: none; margin: 0; padding: 0; }
  .legend li { display: flex; align-items: center; gap: 6px; padding: 1px 2px; }
  .legend li.active { background: #ffe9a8; border-radius: 3px; }
  .legend .swatch {
    width: 14px; height: 14px; border: 1px solid #666; display: inline-block; flex: none;
  }
"""


def build_viewer_html(
    tilejson: dict[str, Any],
    *,
    basemap: str = "none",
    basemap_url: str | None = None,
    basemap_attribution: str | None = None,
) -> str:
    """TileJSON からプレビュー HTML（単一ファイル）を組み立てる。

    Args:
        tilejson: 生成した TileJSON（タイル URL・ズーム範囲・`datapng` をここから読む）
        basemap: ``none``（既定）/ ``gsi`` / ``osm``
        basemap_url: 独自の背景地図 URL テンプレート（`basemap` より優先）
        basemap_attribution: 独自背景地図の帰属表示
    """
    if basemap not in BASEMAP_CHOICES:
        raise ValueError(f"未知の basemap: {basemap!r}（利用可能: {', '.join(BASEMAP_CHOICES)}）")
    if not tilejson.get("bounds"):
        raise ValueError("TileJSON に bounds がありません（初期表示範囲を決められません）")

    if basemap_url:
        base: dict[str, str] | None = {
            "url": basemap_url,
            "attribution": basemap_attribution or "",
            "max_zoom": "22",
        }
    elif basemap != "none":
        base = BASEMAPS[basemap]
    else:
        base = None

    config = {"tilejson": tilejson, "basemap": base, "overzoom": _OVERZOOM}
    # インライン <script> 内に埋め込むため `</` を潰す（`</script>` による早期終了を防ぐ）
    config_json = json.dumps(config, ensure_ascii=False).replace("</", "<\\/")
    title = json.dumps(tilejson.get("name") or "datapng-tiler", ensure_ascii=False).replace(
        "</", "<\\/"
    )

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>プレビュー</title>
<link rel="stylesheet" href="{LEAFLET_CSS}"
      integrity="{LEAFLET_CSS_SRI}" crossorigin="anonymous">
<style>{_STYLE}</style>
</head>
<body>
<div id="map"></div>
<script src="{LEAFLET_JS}"
        integrity="{LEAFLET_JS_SRI}" crossorigin="anonymous"></script>
<script>
const CONFIG = {config_json};
document.title = "プレビュー: " + {title};
{_VIEWER_JS}
</script>
</body>
</html>
"""


def write_viewer_html(html: str, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def basemap_notice(basemap: str) -> str | None:
    """背景地図を選んだときに利用者へ伝える一文（規約の所在）。"""
    info = BASEMAPS.get(basemap)
    if info is None:
        return None
    return (
        f"背景地図 {basemap} を有効にしました。表示・再配布の条件は提供元の規約に従ってください: "
        f"{info['terms']}"
    )
