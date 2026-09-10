#!/usr/bin/env python3
"""Compare the full tile list against the Zarr cube already in S3, tile by tile.

The build-pass twin of `tile_progress.py`, and the same single source of truth for "already
done": there is no separate progress tracker, because a tracker can disagree with S3 and S3
cannot disagree with itself. A tile counts as built only when its store carries the completion
marker -- `zarr_write` puts the group attributes last precisely so an interrupted build reads
as absent rather than as a short tile (`docs/21` 8.16).

Why this is two passes and not one. A LIST over the prefix is cheap and says which stores
*exist*, but it cannot see attributes, so it cannot tell a finished store from one whose chunks
landed before the pod died. The second pass GETs only the root `zarr.json` of the stores the
LIST found -- a few hundred bytes each, concurrent -- so the cost tracks the number of tiles
already built rather than the 5.769 in the list. Checking every tile with a full `_zarr_read`
instead would move the whole cube to answer the question (measured 6,6 s against 0,071 s a
tile).

Usage:
    python cube_progress.py \
        --tiles-csv tiles_native_10km.csv \
        --s3-cube s3://easido-prod-dc-data-projects/easi-workflows-team/biodiv/cube/chile_30m \
        --years 2000-2026 \
        [--limit N] [--out missing_tiles.csv] [--threads 32]

Always prints a one-line summary to stderr (built/total/pct). With --out, writes the tiles
still to build as tile_id,xmin,ymin,xmax,ymax -- the same columns `tile_progress.py` emits, so
the Argo split step is identical for both passes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import boto3

#: Must track `maptask._CACHE_VERSION`. Kept as a literal so this script stays importable
#: without the package on the path -- the Argo list step runs it from a bare checkout.
CACHE_VERSION = 1


def store_name(tile_id: str, resolution: int, y0: int, y1: int) -> str:
    """The store basename `maptask._zarr_path` builds. Same span convention: years[0]-2."""
    return f"v{CACHE_VERSION}_{tile_id}_{resolution}m_{y0 - 2}_{y1}.zarr"


def list_candidates(s3, bucket: str, prefix: str) -> set[str]:
    """Store names that have at least a root `zarr.json` under the prefix."""
    seen: set[str] = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rest = obj["Key"][len(prefix):]
            name, _, tail = rest.partition("/")
            if name.endswith(".zarr") and tail == "zarr.json":
                seen.add(name)
    return seen


def is_complete(s3, bucket: str, prefix: str, name: str) -> bool:
    """Whether the store carries the completion marker `zarr_write` writes last."""
    try:
        body = s3.get_object(Bucket=bucket, Key=f"{prefix}{name}/zarr.json")["Body"].read()
        return bool(json.loads(body).get("attributes", {}).get("bbox"))
    except Exception:                                               # noqa: BLE001
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles-csv", required=True, help="tile_id,xmin,ymin,xmax,ymax (UTM 19S)")
    p.add_argument("--s3-cube", required=True, help="s3://bucket/prefix holding the stores")
    p.add_argument("--years", default="2000-2026")
    p.add_argument("--resolution", type=int, default=30)
    p.add_argument("--limit", type=int, default=0,
                   help="cap how many still-to-build tiles to report/write (0 = all)")
    p.add_argument("--out", default=None, help="write the tiles still to build here")
    p.add_argument("--threads", type=int, default=32, help="concurrent zarr.json GETs")
    args = p.parse_args()

    y0, y1 = (int(v) for v in args.years.split("-"))
    u = urlparse(args.s3_cube)
    bucket, prefix = u.netloc, u.path.lstrip("/")
    if not prefix.endswith("/"):
        prefix += "/"

    s3 = boto3.client("s3")
    candidates = list_candidates(s3, bucket, prefix)

    with open(args.tiles_csv) as f:
        rows = list(csv.DictReader(f))

    wanted = {r["tile_id"]: store_name(r["tile_id"], args.resolution, y0, y1) for r in rows}
    check = [n for n in wanted.values() if n in candidates]
    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as ex:
        complete = {n for n, ok in zip(check, ex.map(
            lambda n: is_complete(s3, bucket, prefix, n), check)) if ok}

    missing = [r for r in rows if wanted[r["tile_id"]] not in complete]
    n_done = len(rows) - len(missing)
    torn = len(check) - len(complete)
    print(f"cube: {n_done}/{len(rows)} tiles built "
          f"({100 * n_done / len(rows):.2f}%), {len(missing)} to build"
          + (f"; {torn} store(s) present but incomplete, they will be rebuilt" if torn else ""),
          file=sys.stderr)

    if args.limit:
        missing = missing[:args.limit]

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tile_id", "xmin", "ymin", "xmax", "ymax"])
            w.writeheader()
            w.writerows({k: r[k] for k in w.fieldnames} for r in missing)


if __name__ == "__main__":
    main()
