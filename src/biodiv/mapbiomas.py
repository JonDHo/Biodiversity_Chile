"""MapBiomas Chile Collection 2 land cover: legend, native mask, nearest-year lookup.

The single authority on MapBiomas in this project. Nothing else may hardcode class integers:
the rasters ship **without** an embedded legend (no colormap, no category names, no .aux.xml)
and `MapBiomas/metadata.txt` documents the hierarchy — `1.1 Forest`, `2.1 Wetland`, ... —
without ever stating which integer the raster stores. Those two code systems are not the same,
and a sample of "native vegetation" quietly becomes a sample of orchards if they are confused.

HOW THE INTEGERS WERE ANCHORED. Not by assumption. The 1,082 Parcelas-CL plots are native
vegetation by design, so sampling MapBiomas at their coordinates says what native looks like:

    66 -> 48.3 %   60 -> 22.9 %   59 -> 7.6 %   12 -> 7.5 %

which puts shrubland at 66 (matorral dominates the Chilean Mediterranean zone) and the forest
subclasses at 59/60/61. Point probes at known ground confirm the anthropic side: Santiago and
Vina give 24, the Constitucion pine plantation gives 9. The rest follow MapBiomas convention
and are marked as such in ``MapBiomas/legend.csv``.

TWO THINGS THE LEGEND DOES NOT SETTLE, both harmless here and neither to be reported as fact:

1. **Which of 59/60/61 is primary, secondary or dwarf** is inference. For the mask it does not
   matter -- all three are native forest -- but the subclass must not be reported without
   confirmation.
2. **67, 79 and 80 are unassigned** (together < 0.2 % of the study envelope). They are excluded
   by default, which is the safe direction: what is not recognised does not enter.

STABILITY IS PER-WINDOW, NOT PER-COLLECTION. ``window_native`` requires native cover across the
three years of a sample's own causal window, not across 1999-2024. Requiring it across the whole
collection drops the study envelope from 112,241 km2 to a projected ~41,191 km2 -- it lets one
misclassified year in 26 kill a pixel, and biases the pool towards dense-forest cores, which is
the opposite of the coverage this sampling exists to buy. The window rule also makes every
figure immune to how many annual maps have finished uploading: measured over three different
windows it gives 84,301 / 83,747 / 83,153 km2.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[2]
MB_DIR = ROOT / "MapBiomas"
LEGEND_CSV = MB_DIR / "legend.csv"

#: Environment override for where the annual rasters live. It exists because tile inference
#: runs on dask-gateway workers and those cannot see the home directory (measured,
#: ``logs/gw_probe.log``): the 3.6 GB of rasters are copied once to the scratch bucket and the
#: workers are pointed at that prefix. Accepts a local directory or an ``s3://`` prefix -- GDAL
#: reads a window straight out of S3, and the rasters are tiled 512x512 with overviews, so a
#: tile-sized window fetches only its own blocks (measured: 0.68 s for a 557 x 557 window).
#: ``legend.csv`` is deliberately not moved: it names classes, it never builds the mask.
RASTERS_DIR_ENV = "BIODIV_MAPBIOMAS_DIR"

#: Where the rasters are when nothing says otherwise: the team prefix, read windowed over
#: the network. The 3.6 GB of annual maps are **not** in the repo -- ``MapBiomas/`` holds
#: only ``legend.csv`` and ``metadata.txt`` -- so a default pointing at it made every caller
#: that did not set ``BIODIV_MAPBIOMAS_DIR`` fail with "no MapBiomas rasters in ...". The
#: prefix is the one `scripts/argo/upload_assets.sh` uploads to and `process_argo.yaml`
#: hands the pod, so pod, gateway worker and notebook now resolve the same rasters with no
#: configuration. A local copy still wins over it, see ``rasters_dir``.
#:
#: This default is only affordable because the rasters are **COGs**: LZW-compressed, 512x512
#: internal tiles, full overview pyramid (2 ... 516), so ``rasterio.open`` range-requests
#: only the blocks a tile window touches, never the 144,896 x 34,599 px whole. Measured on
#: this prefix: 0.79 s for a 10 km tile mask, 0.51 s for a 371 x 371 window. Replacing them
#: with striped or uncompressed TIFFs would turn each of those into a ~150 MB download and
#: this default into a mistake.
DEFAULT_RASTERS_DIR = "s3://easido-prod-dc-data-projects/easi-workflows-team/biodiv/MapBiomas"

#: Native vegetation. Forest (3 and its three subclasses), wetland, grassland, steppe,
#: shrubland. Rocky outcrop (29) is a natural non-forest formation but is **not vegetation**
#: and is excluded: only 1 of 1,082 plots falls on it, so including it would add an object the
#: labelled set barely contains.
NATIVE = frozenset({3, 59, 60, 61, 11, 12, 63, 66})

#: MapBiomas is delivered in geographic coordinates; the project works in UTM 19S.
MB_CRS = "EPSG:4326"
PROJECT_CRS = "EPSG:32719"


@lru_cache(maxsize=1)
def legend() -> pd.DataFrame:
    """Class code -> hierarchy and name, indexed by the raster integer."""
    return pd.read_csv(LEGEND_CSV).set_index("code")


def class_name(code: int) -> str:
    lg = legend()
    return str(lg.loc[code, "name"]) if code in lg.index else f"UNKNOWN_{code}"


def rasters_dir() -> str:
    """Local directory or ``s3://`` prefix holding the annual rasters.

    In order: ``BIODIV_MAPBIOMAS_DIR`` if set (honoured as given -- an empty override is an
    error to be reported, not something to silently fall back from), then a local
    ``MapBiomas/`` that actually holds ``.tif`` files, then ``DEFAULT_RASTERS_DIR``. The
    local check is on the rasters and not on the directory, which exists in every checkout
    for ``legend.csv``.
    """
    env = os.environ.get(RASTERS_DIR_ENV)
    if env:
        return env
    return str(MB_DIR) if any(MB_DIR.glob("*.tif")) else DEFAULT_RASTERS_DIR


@lru_cache(maxsize=4)
def _scan(directory: str) -> tuple[tuple[int, str], ...]:
    """``(year, path)`` for every annual raster under ``directory``.

    Cached on the directory rather than on nothing, so that pointing the process at the
    scratch prefix mid-run re-scans instead of serving the local listing.
    """
    if directory.startswith("s3://"):
        import boto3
        bucket, _, prefix = directory[len("s3://"):].partition("/")
        prefix = prefix.rstrip("/") + "/"
        pages = boto3.client("s3").get_paginator("list_objects_v2")
        names = [o["Key"][len(prefix):]
                 for page in pages.paginate(Bucket=bucket, Prefix=prefix)
                 for o in page.get("Contents", [])]
        base = directory.rstrip("/")
        found = {n: f"{base}/{n}" for n in names if n.endswith(".tif") and "/" not in n}
    else:
        found = {f.name: str(f) for f in sorted(Path(directory).glob("*.tif"))}
    out = {}
    for name, path in sorted(found.items()):
        m = re.match(r"(\d{4})_", name)
        if m:
            out[int(m.group(1))] = path
    return tuple(out.items())


def available_years() -> dict[int, str]:
    """Year -> raster path, for the annual maps actually present.

    Files are named ``<year>_coverage_lclu_<version>_<uuid>.tif``; the uuid differs per year,
    so the year has to be parsed rather than formatted into a template.
    """
    return dict(_scan(rasters_dir()))


def year_map(year: int) -> tuple[str, int, int]:
    """``(path, year_used, delta)`` for ``year``, falling back to the nearest available map.

    ``delta`` is signed years of displacement and is recorded per sample, so that how far a
    match had to stretch stays auditable instead of being silently absorbed. Ties go to the
    earlier year, which keeps the choice deterministic.
    """
    have = available_years()
    if not have:
        raise FileNotFoundError(f"no MapBiomas rasters in {rasters_dir()}")
    if year in have:
        return have[year], year, 0
    nearest = min(have, key=lambda y: (abs(y - year), y))
    return have[nearest], nearest, nearest - year


def _read(path: str | Path, bounds_ll: tuple[float, float, float, float]) -> np.ndarray:
    """Windowed read of one annual map.

    Always windowed: the rasters are 144,896 x 34,599 px, so a full read is ~5e9 pixels and
    will take the machine down rather than fail.
    """
    west, south, east, north = bounds_ll
    with rasterio.open(path) as src:
        return src.read(1, window=from_bounds(west, south, east, north, src.transform),
                        boundless=True, fill_value=0)


def window_native(bounds_ll: tuple[float, float, float, float],
                  win_start: int, win_end: int) -> tuple[np.ndarray, list[dict]]:
    """Native cover across **every** year of the causal window ``win_start..win_end``.

    Returns ``(mask, used)`` where ``mask`` is boolean over the requested bounds and ``used``
    records, per window year, which map answered for it and by how much it was displaced.

    The AND across the window is the point: a pixel that was plantation and got cleared, or
    cropland that went fallow, is not a native time series even if it looks native in the
    census year. The three-year window is exactly what has to be homogeneous, because it is
    exactly what the extracted series covers.
    """
    mask, used = None, []
    for y in range(win_start, win_end + 1):
        path, used_year, delta = year_map(y)
        a = _read(path, bounds_ll)
        m = np.isin(a, list(NATIVE))
        mask = m if mask is None else (mask & m)
        used.append(dict(year=y, year_used=used_year, delta=delta))
    return mask, used


def sample_at(lon, lat, year: int) -> tuple[np.ndarray, int, int]:
    """Class code at each ``(lon, lat)`` for ``year``. Returns ``(codes, year_used, delta)``."""
    path, used_year, delta = year_map(year)
    with rasterio.open(path) as src:
        codes = np.array([v[0] for v in src.sample(zip(np.atleast_1d(lon),
                                                       np.atleast_1d(lat)))])
    return codes, used_year, delta


def is_native(codes) -> np.ndarray:
    return np.isin(np.asarray(codes), list(NATIVE))


@lru_cache(maxsize=1)
def _transformers():
    from pyproj import Transformer
    return (Transformer.from_crs(PROJECT_CRS, MB_CRS, always_xy=True),
            Transformer.from_crs(MB_CRS, PROJECT_CRS, always_xy=True))


def utm_to_ll(x, y):
    return _transformers()[0].transform(x, y)


def ll_to_utm(lon, lat):
    return _transformers()[1].transform(lon, lat)


def grid_lonlat(bounds_ll: tuple[float, float, float, float],
                shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre lon/lat for an array read with ``_read`` over ``bounds_ll``.

    Kept next to ``_read`` on purpose: the two have to agree on pixel-centre convention, and
    a half-pixel disagreement here shifts every sampled coordinate by 15 m without any error.
    """
    west, south, east, north = bounds_ll
    ny, nx = shape
    lon = west + (np.arange(nx) + 0.5) * (east - west) / nx
    lat = north - (np.arange(ny) + 0.5) * (north - south) / ny
    return lon, lat
