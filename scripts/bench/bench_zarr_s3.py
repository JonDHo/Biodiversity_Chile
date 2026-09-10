"""Read the materialised cube from S3, and time it. (docs/24 §7 paso 6b)

This is the last unmeasured piece of Fase 0. Everything in `docs/21` §8.14 and §8.15 read
stores on **local disk**; the production shape is a build pass writing to S3 and an inference
pass reading from it, so the read has to be measured against a real ``s3://`` prefix.

Three modes, because there are three ways to get the bytes onto the GPU node and they have
different shapes:

``direct``    `_zarr_read` straight off ``s3://``. One GET per chunk, in the year loop.
``copy``      bulk-copy the store to local disk first, then read it there. This is what an
              on-node SSD buys: a sequential transfer that can overlap the previous tile's
              forward pass, which a chunk read inside the year loop cannot.
``local``     the baseline already measured (~2,5 s), re-run here so the three numbers come
              off the same machine on the same day.

`--tchunk` re-writes the store with the time axis split into blocks instead of held as one
chunk. The single chunk is the right shape on local disk and gives an S3 read nothing to
overlap, so whether splitting it pays is a question only S3 can answer.

Usage:
    # stage the fixtures (one-off)
    PYTHONPATH=src python scripts/bench/bench_zarr_s3.py upload \
        --from /home/jovyan/biodiv_cube_tiles --to s3://BUCKET/PREFIX/cube_bench

    # measure
    PYTHONPATH=src python scripts/bench/bench_zarr_s3.py read \
        --root s3://BUCKET/PREFIX/cube_bench --mode direct --jobs 1
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TILES = ("t17_599", "t17_600", "t18_599", "t18_600")


def tile_records(csv: Path, ids) -> list[dict]:
    df = pd.read_csv(csv)
    recs = {r["tile_id"]: r for r in df.to_dict("records")}
    missing = [t for t in ids if t not in recs]
    if missing:
        raise SystemExit(f"tiles not in {csv}: {missing}")
    return [recs[t] for t in ids]


def store_bytes(root: str, tile: dict, cfg) -> int:
    """Size of one store, local or on S3."""
    from biodiv import maptask as mt
    p = mt._zarr_path(root, tile, cfg)
    if isinstance(p, Path):
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    import s3fs
    fs = s3fs.S3FileSystem()
    return sum(o["size"] for o in fs.find(str(p), detail=True).values())


def cmd_upload(a, cfg, tiles) -> None:
    """Stage the fixtures on S3, optionally re-chunking on the way."""
    from biodiv import maptask as mt
    for tile in tiles:
        da = mt._zarr_read(str(a.src), tile, cfg)
        if da is None:
            raise SystemExit(f"no local store for {tile['tile_id']} under {a.src}")
        t = time.perf_counter()
        p = mt.zarr_write(a.to, tile, cfg, da, clevel=a.clevel, tchunk=a.tchunk)
        wrote = time.perf_counter() - t
        size = store_bytes(a.to, tile, cfg)
        print(f"{tile['tile_id']}: {da.shape} -> {size / 1e6:.1f} MB in {wrote:.1f}s  "
              f"tchunk={a.tchunk or da.shape[0]}  {p}", flush=True)


def read_one(root: str, tile: dict, cfg, mode: str, stage: Path | None):
    """Return (seconds, array). `seconds` is everything it takes to have the array in hand."""
    from biodiv import maptask as mt
    if mode != "copy":
        t = time.perf_counter()
        da = mt._zarr_read(root, tile, cfg)
        return time.perf_counter() - t, da, 0.0

    # `copy`: bulk transfer first, then read locally. Timed apart, because the copy is the
    # part that can overlap the previous tile's forward pass and the read is not.
    src = mt._zarr_path(root, tile, cfg)
    dst = stage / Path(str(src)).name
    shutil.rmtree(dst, ignore_errors=True)
    t = time.perf_counter()
    subprocess.run(["aws", "s3", "cp", "--recursive", "--quiet", str(src), str(dst)],
                   check=True)
    copied = time.perf_counter() - t
    t = time.perf_counter()
    da = mt._zarr_read(str(stage), tile, cfg)
    return time.perf_counter() - t, da, copied


def cmd_read(a, cfg, tiles) -> None:
    stage = Path(a.stage) if a.stage else None
    if stage:
        stage.mkdir(parents=True, exist_ok=True)

    if a.jobs > 1:
        # One process per tile, which is what `--jobs N` does in `scripts/73`. The point is
        # whether N readers share the link or contend for it.
        import multiprocessing as mp
        t0 = time.perf_counter()
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            out = pool.starmap(_worker, [(a.root, t, cfg, a.mode, stage) for t in tiles])
        wall = time.perf_counter() - t0
        for tid, (sec, cop, mb) in zip([t["tile_id"] for t in tiles], out):
            print(f"  {tid}: read {sec:.2f}s  copy {cop:.2f}s  {mb:.0f} MB", flush=True)
        tot = sum(s + c for s, c, _ in out)
        print(f"{a.mode} --jobs {a.jobs}: wall {wall:.2f}s for {len(tiles)} tiles, "
              f"{wall / len(tiles):.2f}s/tile, sum of per-tile {tot:.2f}s", flush=True)
        return

    ref = None
    for tile in tiles:
        sec, da, copied = read_one(a.root, tile, cfg, a.mode, stage)
        if da is None:
            raise SystemExit(f"miss for {tile['tile_id']} under {a.root}")
        mb = da.values.nbytes / 1e6
        print(f"  {tile['tile_id']}: read {sec:.2f}s  copy {copied:.2f}s  "
              f"{mb:.0f} MB raw  {mb / max(sec + copied, 1e-9):.0f} MB/s", flush=True)
        if a.verify:
            from biodiv import maptask as mt
            local = mt._zarr_read(a.verify, tile, cfg)
            if local is None:
                raise SystemExit(f"no local store for {tile['tile_id']} to verify against")
            same = (np.array_equal(da.values, local.values, equal_nan=True)
                    and np.array_equal(da.time.values, local.time.values)
                    and np.array_equal(da.x.values, local.x.values)
                    and np.array_equal(da.y.values, local.y.values))
            print(f"    vs local: {'BIT-IDENTICAL' if same else 'DIFFERS'}", flush=True)
            if not same:
                ref = False
    if a.verify and ref is False:
        raise SystemExit("verification failed")


def _worker(root, tile, cfg, mode, stage):
    sec, da, copied = read_one(root, tile, cfg, mode, stage)
    if da is None:
        raise SystemExit(f"miss for {tile['tile_id']}")
    return sec, copied, da.values.nbytes / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("upload", "read"))
    ap.add_argument("--from", dest="src", help="local root holding the stores (upload)")
    ap.add_argument("--to", help="s3:// prefix to write (upload)")
    ap.add_argument("--root", help="root to read from, local or s3:// (read)")
    ap.add_argument("--mode", choices=("direct", "copy", "local"), default="direct")
    ap.add_argument("--stage", default="", help="local directory for --mode copy")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--tchunk", type=int, default=0, help="dates per time chunk, 0 = one chunk")
    ap.add_argument("--clevel", type=int, default=1)
    ap.add_argument("--verify", default="", help="local root to check the bytes against")
    ap.add_argument("--tiles", default=",".join(TILES))
    ap.add_argument("--tiles-csv", type=Path,
                    default=Path("results/figures/tiles_native_10km.csv"))
    ap.add_argument("--years", default="2005,2015,2024",
                    help="only the span matters here: it names the store")
    a = ap.parse_args()

    from biodiv import maptask as mt
    cfg = mt.TileConfig(years=[int(v) for v in a.years.split(",")], dest="", tags={})
    tiles = tile_records(a.tiles_csv, a.tiles.split(","))

    if a.cmd == "upload":
        if not (a.src and a.to):
            ap.error("upload needs --from and --to")
        cmd_upload(a, cfg, tiles)
    else:
        if not a.root:
            ap.error("read needs --root")
        if a.mode == "copy" and not a.stage:
            ap.error("--mode copy needs --stage")
        cmd_read(a, cfg, tiles)


if __name__ == "__main__":
    main()
