# datapng-tiler

A CLI and Python library that turns raster data into [data PNG](https://gsj-seamless.jp/labs/datapng/) tiles (**numerical PNG** and **palette PNG**) plus a **TileJSON** describing them.

Conforms to the [TileJSON DataPNG Extension](https://github.com/qchizu-project/tilejson-datapng-extension) v0.7.0.

> **言語**: [日本語版 README](./README.md)（詳しい説明はこちら）

## What it does

Use it to serve **continuous values** (elevation, depth, temperature) or **categories** (land use, flood-depth classes) as map tiles. It writes tiles that carry the values losslessly in RGB, together with the TileJSON metadata a client needs to decode them — factor, unit, invalid value, legend.

| | numerical PNG | palette PNG |
|---|---|---|
| Input | continuous-value raster | class-value raster, or RGB raster + legend |
| Storage | value quantized to a signed 24-bit integer, packed into RGB | legend colors, byte for byte |
| Typical use | elevation, depth, temperature | land use, hazard classes |

Tiles are **WebP (lossless) by default**; PNG is also available. Both are lossless, so values are never degraded.

## Install

```sh
uvx datapng-tiler --help          # run without installing
pipx install datapng-tiler        # install as a command
pip install datapng-tiler         # use as a library too
```

Requires Python 3.12+. GDAL ships inside the rasterio wheel, so there is nothing else to install.

## Quick start

```sh
# Numerical tiles (e.g. elevation, 1 cm resolution)
datapng-tiler tile dem.tif -o tiles/ --factor 0.01 --unit m

# Palette tiles from a class-value raster plus a legend
datapng-tiler tile flood.tif -o tiles/ --type palette --legend legend.yaml

# Migrate existing Terrain-RGB tiles to the canonical encoding
datapng-tiler convert ./terrain-rgb/ -o tiles/ --from mapbox --factor 0.01 --unit m

# Check the result against the specification
datapng-tiler validate tiles/tiles.json --tiles tiles/
```

`tile` writes the tile tree, `tiles.json` and a preview `index.html` that decodes and shows values under the cursor.

## Correctness notes

- **No threshold-based nodata detection.** Treating "values close to nodata" as invalid destroys real data when nodata is 0 or positive. Validity comes from the GDAL mask; for sources without a declared nodata, the warp's coverage mask (`add_alpha`) marks everything outside the source as invalid.
- **Values that do not fit in 24 bits are an error**, not a silent wraparound. Use `--on-overflow clamp|nodata` to choose otherwise.
- **The declared `support` always matches how tiles were produced.** `--support point` (default) means top-left alignment and top-left propagation into overviews; `--support block` means center alignment and integer averaging. They cannot be set independently.
- **Reprojection uses a near-exact transformer.** GDAL's default approximate transformer has an error that varies with the output resolution, which makes the same ground point sample slightly differently at different zoom levels.
- **Overviews are combined in the raw integer domain**, so quantization error does not accumulate across zoom levels.
- **Output is deterministic**: the same input and settings produce byte-identical tiles regardless of `--jobs`.

## Invalid values

Invalid pixels are marked either by alpha 0 or by a designated color — one or the other:

| | invalid pixels | `invalidColor` |
|---|---|---|
| default | alpha 0 | not declared |
| `--no-alpha --invalid-color R G B` | that color | declared |

Per specification §3.2.2, `invalidColor` **cannot designate a fully transparent pixel** (lossless WebP does not preserve the RGB of such pixels). In the default output invalid pixels are fully transparent, so declaring a color would have no effect; the CLI rejects `--invalid-color` on its own.

## Preview

`index.html` sits at the root of the tile tree. Serve it (`python -m http.server -d tiles/`) and hover the map to read decoded values — this catches "it renders, but the numbers are wrong".

**No basemap is loaded by default.** Defaulting to a third-party tile service would impose that service's terms on everyone who uses this tool. Opt in with `--basemap gsi|osm`.

## Development

```sh
uv sync
uv run pytest -v
uv run ruff check . && uv run ruff format .
```

Tests need no external data or network. Conformance is checked against a decoder transcribed literally from the specification text, not against this project's own decoder.

## License

MIT — see [LICENSE](./LICENSE) and [NOTICE](./NOTICE). The specification it implements is published under CC0 1.0.
