"""Materialise one tile's kNDVI as a Zarr store. (docs/24 Fase 0, S2.1)

Two sources, and the distinction matters for what the gate proves:

``--from-cache``  convert a `BIODIV_TILE_CACHE` directory written by a previous `dc.load`.
                  The dev cache is already verified bit-identical to `dc.load` (`docs/21`
                  section 8.8), so converting it tests the Zarr layer without paying the
                  ~700 s load a second time.
``--from-dc``     load from the datacube directly. Slower, and what a real build would do.

Either way the store is written by `maptask.zarr_write`, so this exercises the same code the
production read path would consume -- the point is to test that function, not to reimplement
it here.

Usage:
    PYTHONPATH=src python scripts/bench/build_tile_zarr.py \
        --tiles-file tile.csv --years 2005,2015,2024 \
        --from-cache /path/to/tilecache --out /path/to/zarr
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-file", type=Path, required=True)
    ap.add_argument("--years", required=True, help="comma list, e.g. 2005,2015,2024")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--from-cache", type=Path)
    ap.add_argument("--from-dc", action="store_true")
    ap.add_argument("--resolution", type=int, default=30)
    ap.add_argument("--clevel", type=int, default=5)
    a = ap.parse_args()
    if bool(a.from_cache) == bool(a.from_dc):
        ap.error("need exactly one of --from-cache or --from-dc")

    from biodiv import maptask as mt

    years = [int(v) for v in a.years.split(",")]
    cfg = mt.TileConfig(years=years, dest="", tags={}, resolution=a.resolution)
    a.out.mkdir(parents=True, exist_ok=True)

    dc = None
    if a.from_dc:
        import datacube
        from datacube.utils.aws import configure_s3_access
        configure_s3_access(aws_unsigned=False, requester_pays=True)
        dc = datacube.Datacube(app="biodiv-build-tile-zarr")

    for tile in pd.read_csv(a.tiles_file).to_dict("records"):
        t = time.perf_counter()
        if a.from_cache:
            da = mt._cache_read(str(a.from_cache), tile, cfg)
            if da is None:
                raise SystemExit(f"no cache for {tile['tile_id']} at {a.from_cache}")
            src = "cache"
        else:
            da = mt.load_tile(dc, tile, cfg)
            if da is None:
                raise SystemExit(f"no data for {tile['tile_id']}")
            src = "dc.load"
        read = time.perf_counter() - t

        t = time.perf_counter()
        p = mt.zarr_write(str(a.out), tile, cfg, da, clevel=a.clevel)
        wrote = time.perf_counter() - t
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        raw = da.values.nbytes
        print(f"{tile['tile_id']}: {da.shape} from {src} in {read:.1f}s -> "
              f"{size / 1e6:.1f} MB ({raw / size:.2f}x) in {wrote:.1f}s  {p.name}",
              flush=True)


if __name__ == "__main__":
    main()
