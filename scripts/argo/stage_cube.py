#!/usr/bin/env python3
"""Copy one chunk's Zarr stores from the cube in S3 down to local disk.

The inference pass reads the cube through `BIODIV_TILE_ZARR`, which takes a local path or an
``s3://`` prefix indifferently -- so this changes nothing about the run except where the bytes
come from. It exists because those two are not equally fast on a GPU node (`docs/21` 8.16,
22-year tile):

    read direct from S3   3,1 s          (8,3 s with four readers)
    copy S3 -> local      4,0 s
    read from local disk  0,49 s

Reading direct costs ~13 % of a tile against the ~70 s of GPU work; staging first and reading
locally costs ~7 %, because the copy is one bulk sequential transfer -- the shape S3 is good at
-- instead of chunk reads interleaved with the year loop. Neither is free, and staging only
wins because it is done once per pod rather than once per year.

What it does NOT do is overlap the copy with compute. That would take the visible cost to ~1 %
and needs a prefetch thread inside the tile loop; this is the version that requires no change
to `scripts/73`. Run it before inference, not during.

Usage:
    python stage_cube.py --tiles-file batch_0.csv \
        --s3-cube s3://BUCKET/PREFIX/cube/chile_30m \
        --years 2000-2026 --dest /scratch/cube [--threads 16]

Skips stores already present and complete, so a retried pod does not re-download what the
previous attempt already got.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import boto3

CACHE_VERSION = 1


def store_name(tile_id: str, resolution: int, y0: int, y1: int) -> str:
    return f"v{CACHE_VERSION}_{tile_id}_{resolution}m_{y0 - 2}_{y1}.zarr"


def is_complete(local: Path) -> bool:
    """Same completion marker the writer uses: group attributes, written last."""
    try:
        meta = json.loads((local / "zarr.json").read_text())
        return bool(meta.get("attributes", {}).get("bbox"))
    except Exception:                                               # noqa: BLE001
        return False


def fetch_store(bucket: str, prefix: str, name: str, dest: Path) -> tuple[str, int, str]:
    """Download every object under one store. Returns (name, bytes, status)."""
    local = dest / name
    if is_complete(local):
        return name, 0, "present"
    s3 = boto3.client("s3")                       # one client per thread; they are not shared
    total = 0
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=f"{prefix}{name}/"):
        keys.extend((o["Key"], o["Size"]) for o in page.get("Contents", []))
    if not keys:
        return name, 0, "absent"
    # `zarr.json` last, for the same reason the writer puts the attributes last: an interrupted
    # download then leaves something `is_complete` calls unfinished rather than something the
    # reader would accept as a short tile.
    keys.sort(key=lambda kv: kv[0].endswith("/zarr.json"))
    for key, size in keys:
        target = local / key[len(prefix) + len(name) + 1:]
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(target))
        total += size
    return name, total, "fetched"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles-file", required=True)
    p.add_argument("--s3-cube", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--years", default="2000-2026")
    p.add_argument("--resolution", type=int, default=30)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--allow-missing", action="store_true",
                   help="carry on when a tile has no store in the cube; without it a missing "
                        "store fails the pod, which is what you want when the build pass is "
                        "supposed to have finished already")
    a = p.parse_args()

    y0, y1 = (int(v) for v in a.years.split("-"))
    u = urlparse(a.s3_cube)
    bucket, prefix = u.netloc, u.path.lstrip("/")
    if not prefix.endswith("/"):
        prefix += "/"
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)

    with open(a.tiles_file) as f:
        names = [store_name(r["tile_id"], a.resolution, y0, y1) for r in csv.DictReader(f)]

    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, a.threads)) as ex:
        out = list(ex.map(lambda n: fetch_store(bucket, prefix, n, dest), names))
    wall = time.perf_counter() - t

    got = sum(b for _, b, _ in out)
    absent = [n for n, _, s in out if s == "absent"]
    n_present = sum(s == "present" for _, _, s in out)
    print(f"staged {len(out) - len(absent) - n_present} store(s), {n_present} already present, "
          f"{got / 1e9:.2f} GB in {wall:.1f}s ({got / 1e6 / max(wall, 1e-9):.0f} MB/s) -> {dest}",
          file=sys.stderr)
    if absent:
        print(f"no store in the cube for: {', '.join(absent)}", file=sys.stderr)
        if not a.allow_missing:
            raise SystemExit(f"{len(absent)} tile(s) missing from the cube")


if __name__ == "__main__":
    main()
