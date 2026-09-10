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


def store_bytes(p) -> int:
    """Size of one store, whether it landed on disk or on S3."""
    if isinstance(p, Path):
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    import s3fs
    return sum(o["size"] for o in s3fs.S3FileSystem().find(p, detail=True).values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-file", type=Path, required=True)
    ap.add_argument("--years", required=True, help="comma list, e.g. 2005,2015,2024")
    ap.add_argument("--out", required=True,
                    help="local directory or s3:// prefix for the stores. Left as a string on "
                         "purpose: `Path` collapses the double slash of an s3:// URI.")
    ap.add_argument("--from-cache", type=Path)
    ap.add_argument("--from-dc", action="store_true")
    ap.add_argument("--resolution", type=int, default=30)
    ap.add_argument("--load-threads", type=int, default=4, dest="load_threads",
                    help="dask threaded-scheduler threads for the `--from-dc` load. The "
                         "`TileConfig` default is 0, which is the synchronous path and costs "
                         "~1,900 s a tile; `scripts/73` defaults to 4, the measured knee "
                         "(docs/21 section 8.1). Ignored by `--from-cache`.")
    ap.add_argument("--clevel", type=int, default=1,
                    help="zstd level. 1 is the measured choice: 3,2x faster than 5 and only "
                         "3 %% bigger (docs/24 section 3).")
    ap.add_argument("--tchunk", type=int, default=-1,
                    help="dates per time chunk; -1 uses `maptask._ZARR_TCHUNK`, 0 writes the "
                         "time axis as one chunk (the Fase 0 layout).")
    ap.add_argument("--skip-existing", action="store_true", dest="skip_existing",
                    help="leave tiles that already have a readable store alone, which is what "
                         "makes the build pass resumable on spot capacity (docs/24 section 7).")
    a = ap.parse_args()
    if bool(a.from_cache) == bool(a.from_dc):
        ap.error("need exactly one of --from-cache or --from-dc")

    from biodiv import maptask as mt

    years = [int(v) for v in a.years.split(",")]
    cfg = mt.TileConfig(years=years, dest="", tags={}, resolution=a.resolution,
                        load_threads=a.load_threads)
    out = a.out
    if not out.startswith("s3://"):
        Path(out).mkdir(parents=True, exist_ok=True)

    dc = None
    if a.from_dc:
        import datacube
        from datacube.utils.aws import configure_s3_access
        configure_s3_access(aws_unsigned=False, requester_pays=True)
        dc = datacube.Datacube(app="biodiv-build-tile-zarr")

    for tile in pd.read_csv(a.tiles_file).to_dict("records"):
        if a.skip_existing and mt._zarr_exists(out, tile, cfg):
            print(f"{tile['tile_id']}: already built, skipped", flush=True)
            continue
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
        kw = {} if a.tchunk < 0 else {"tchunk": a.tchunk}
        p = mt.zarr_write(out, tile, cfg, da, clevel=a.clevel, **kw)
        wrote = time.perf_counter() - t
        size = store_bytes(p)
        raw = da.values.nbytes
        print(f"{tile['tile_id']}: {da.shape} from {src} in {read:.1f}s -> "
              f"{size / 1e6:.1f} MB ({raw / size:.2f}x) in {wrote:.1f}s  {p}",
              flush=True)


if __name__ == "__main__":
    main()
