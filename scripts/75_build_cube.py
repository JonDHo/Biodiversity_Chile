#!/usr/bin/env python3
"""Build the materialised kNDVI cube: dc.load -> compute -> one Zarr store per tile in S3.

This is `docs/24` paso 6a, the construction pass of Option A. It carries no model, no
checkpoints and no GPU: it exists only to turn ~1.800 COG window reads per tile into one Zarr
store, so that every later inference run reads the cube instead (`docs/21` 8.14, 8.16).

    python scripts/75_build_cube.py \
        --tiles-file missing_tiles.csv --years 2000-2026 \
        --cube s3://BUCKET/PREFIX/cube/chile_30m --workers 7 --resume

**The scheduler is the one decision that matters here, and it is not the one `scripts/73`
makes.** `scripts/73` loads with a threaded scheduler because a `LocalCluster`'s workers stay
alive through the CNN and tax it 19 %, which at 27 years costs more than the load saves
(`docs/21` 8.10). This pass has no CNN, so that penalty does not exist and the cluster wins
outright: measured on a full 22-year tile, 688,8 s with `--load-threads 4` against 292,2 s with
`--workers 7`, **2,36x**, with the rasters bit-identical between the two (`docs/21` 8.17).
Hence `--workers 7` is the default here and `--load-threads` is the fallback.

Resumability is "does a complete store exist", the same question `scripts/argo/cube_progress.py`
asks of the whole list -- and a *complete* store, not merely a present one: `maptask.zarr_write`
writes the group attributes last, in one atomic put, so a build killed mid-write leaves
something that reads as absent rather than as a short tile. S3 has no rename, so that ordering
is what stands in for the local `.part`-and-move (`docs/21` 8.16).

A tile that fails is recorded and the pass moves on. Over thousands of tiles on spot capacity a
single bad tile must not take the pod with it -- the manifest is what says which ones to revisit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_years(s: str) -> list[int]:
    if "-" in s:
        a, b = (int(v) for v in s.split("-"))
        return list(range(a, b + 1))
    return [int(v) for v in s.split(",")]


def store_bytes(p) -> int:
    """Size of one store, whether it landed on disk or on S3."""
    if isinstance(p, Path):
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    import s3fs
    return sum(o["size"] for o in s3fs.S3FileSystem().find(str(p), detail=True).values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles-file", required=True, help="tile_id,xmin,ymin,xmax,ymax")
    ap.add_argument("--cube", required=True,
                    help="local directory or s3:// prefix for the stores. A string, not a "
                         "Path: Path collapses the double slash of an s3:// URI.")
    ap.add_argument("--years", default="2000-2026",
                    help="the span actually loaded is years[0]-2 .. years[-1], the same "
                         "convention scripts/73 uses, so the cube matches what inference wants")
    ap.add_argument("--resolution", type=int, default=30)
    ap.add_argument("--workers", type=int, default=7,
                    help="dask LocalCluster workers (0 = threaded scheduler). 7 is the "
                         "measured choice, 2,36x over --load-threads 4 (docs/21 8.17)")
    ap.add_argument("--load-threads", type=int, default=4, dest="load_threads",
                    help="threads for the fallback path, used only when --workers 0")
    ap.add_argument("--clevel", type=int, default=1,
                    help="zstd level; 1 is the measured choice (docs/24 sec. 3)")
    ap.add_argument("--tchunk", type=int, default=-1,
                    help="dates per time chunk; -1 uses maptask._ZARR_TCHUNK")
    ap.add_argument("--resume", action="store_true",
                    help="skip tiles that already have a complete store")
    ap.add_argument("--out", default="", help="directory for manifest.json / manifest.csv")
    ap.add_argument("--tag", default="", help="stamped into the manifest for provenance")
    a = ap.parse_args()

    from biodiv import maptask as mt

    years = parse_years(a.years)
    cfg = mt.TileConfig(years=years, dest="", tags={}, resolution=a.resolution,
                        load_threads=a.load_threads, load_client=a.workers > 0)
    cube = a.cube
    if not cube.startswith("s3://"):
        Path(cube).mkdir(parents=True, exist_ok=True)

    import datacube
    from datacube.utils.aws import configure_s3_access

    client = cluster = None
    if a.workers > 0:
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster(n_workers=a.workers, processes=True, threads_per_worker=1)
        client = Client(cluster)
        print(f"dask: local cluster, {a.workers} workers", flush=True)
    else:
        print(f"load: dask threaded scheduler, {a.load_threads} threads", flush=True)

    # Boot order is not negotiable, same as `scripts/73`: cluster, then
    # configure_s3_access(client=...), then the Datacube. `usgs-landsat` is requester-pays and
    # passing `client` is what propagates that to the workers; setting it in the driver alone
    # leaves every worker read failing with AccessDenied.
    configure_s3_access(aws_unsigned=False, requester_pays=True, client=client)
    dc = datacube.Datacube(app="biodiv-build-cube")

    tiles = pd.read_csv(a.tiles_file).to_dict("records")
    rows: list[dict] = []
    t_run = time.perf_counter()
    try:
        for i, tile in enumerate(tiles, 1):
            tid = tile["tile_id"]
            if a.resume and mt._zarr_exists(cube, tile, cfg):
                print(f"[{i}/{len(tiles)}] {tid}: already built, skipped", flush=True)
                rows.append(dict(tile_id=tid, status="skipped"))
                continue
            try:
                # Per tile, not once before the loop. GDAL holds the credential values it was
                # given and cannot renew them from the projected service-account token the way
                # boto3 does, so a pass outliving its role session starts failing every read
                # with RasterioIOError('The provided token has expired') -- the failure that
                # killed all five chunks of the 2026-09-08 run at the 63-minute mark. It only
                # reaches STS near expiry.
                configure_s3_access(aws_unsigned=False, requester_pays=True, client=client)

                t = time.perf_counter()
                da = mt.load_tile(dc, tile, cfg)
                load_s = time.perf_counter() - t
                if da is None:
                    raise RuntimeError("no data for this tile and span")

                t = time.perf_counter()
                kw = {} if a.tchunk < 0 else {"tchunk": a.tchunk}
                p = mt.zarr_write(cube, tile, cfg, da, clevel=a.clevel, **kw)
                write_s = time.perf_counter() - t
                size = store_bytes(p)

                rows.append(dict(tile_id=tid, status="ok", n_dates=int(da.shape[0]),
                                 ny=int(da.shape[1]), nx=int(da.shape[2]),
                                 load_seconds=round(load_s, 1),
                                 write_seconds=round(write_s, 1),
                                 bytes=size, path=str(p)))
                print(f"[{i}/{len(tiles)}] {tid}: {da.shape} load {load_s:.1f}s -> "
                      f"{size / 1e6:.1f} MB write {write_s:.1f}s", flush=True)
            except Exception as e:                                  # noqa: BLE001
                # One bad tile must not take the pod with it; the manifest says which to revisit.
                rows.append(dict(tile_id=tid, status="error",
                                 error=f"{type(e).__name__}: {str(e)[:300]}"))
                print(f"[{i}/{len(tiles)}] {tid}: ERROR {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
    finally:
        for x in (client, cluster):
            if x is not None:
                x.close()

    ok = sum(r["status"] == "ok" for r in rows)
    err = sum(r["status"] == "error" for r in rows)
    skip = sum(r["status"] == "skipped" for r in rows)
    wall = time.perf_counter() - t_run
    print(f"built {ok}, skipped {skip}, errors {err} of {len(tiles)} in {wall / 60:.1f} min",
          flush=True)

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
        (out / "run.json").write_text(json.dumps({
            "tag": a.tag, "cube": cube, "years": a.years, "resolution": a.resolution,
            "workers": a.workers, "load_threads": a.load_threads, "clevel": a.clevel,
            "tchunk": a.tchunk, "tiles": len(tiles), "ok": ok, "skipped": skip, "errors": err,
            "wall_seconds": round(wall, 1), "argv": sys.argv,
            "zarr_tchunk_default": mt._ZARR_TCHUNK, "cache_version": mt._CACHE_VERSION,
        }, indent=2))

    # A pod that built nothing and errored on everything should fail, so Argo retries it.
    if err and not ok:
        raise SystemExit(f"every tile failed ({err})")


if __name__ == "__main__":
    main()
