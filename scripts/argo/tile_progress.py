#!/usr/bin/env python3
"""Compare the full tile list against what's already in S3, tile by tile.

A tile counts as done only when every requested year exists for it in S3
(<tile_id>_<year>.tif under the destination prefix) — this is the single
source of truth for "already processed," shared by the Argo
list-missing-tiles step and any ad hoc progress check. It doesn't care who
wrote a tile: the existing pod's chile_gw/chile_pod halves and any Argo
pod write into the same prefix, so this is what keeps a rerun from
reprocessing a tile someone else already finished.

Usage:
    python tile_progress.py \
        --tiles-csv tiles_native_10km.csv \
        --s3-dest s3://easido-prod-user-scratch/<userid>/biodiv/outputs/maps/chile_30m_2000_2026 \
        --years 2000-2026 \
        [--s3-cube s3://.../work/cube/chile_30m] [--limit N] [--out missing_tiles.csv]

Always prints a one-line summary to stderr (done/total/pct). With --out,
also writes the tiles still missing at least one year as
tile_id,xmin,ymin,xmax,ymax — --limit caps how many of those it writes,
for a small test run instead of the full remaining backlog. With --s3-cube
the list is further restricted to tiles whose cube store is complete
(cube_progress.built_tiles), which is what the cube_argo infer stage can run.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from urllib.parse import urlparse

import boto3

TILE_RE = re.compile(r"^(?P<tile_id>t\d+_\d+)_(?P<year>\d{4})\.tif$")


def list_done_years(s3, bucket: str, prefix: str) -> dict[str, set[int]]:
    done: dict[str, set[int]] = defaultdict(set)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"].rsplit("/", 1)[-1]
            m = TILE_RE.match(key)
            if m:
                done[m["tile_id"]].add(int(m["year"]))
    return done


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles-csv", required=True, help="tile_id,xmin,ymin,xmax,ymax (UTM 19S)")
    p.add_argument("--s3-dest", required=True, help="s3://bucket/prefix")
    p.add_argument("--years", default="2000-2026")
    p.add_argument("--s3-cube", default=None,
                   help="s3://bucket/prefix of the materialised cube; with it, only tiles that "
                        "have a complete store are listed (the others cannot be inferred yet)")
    p.add_argument("--resolution", type=int, default=30, help="store resolution, with --s3-cube")
    p.add_argument("--limit", type=int, default=0, help="cap how many missing tiles to report/write (0 = all)")
    p.add_argument("--out", default=None, help="write missing tiles here as tile_id,xmin,ymin,xmax,ymax")
    args = p.parse_args()

    y0, y1 = (int(x) for x in args.years.split("-"))
    years = set(range(y0, y1 + 1))

    u = urlparse(args.s3_dest)
    bucket, prefix = u.netloc, u.path.lstrip("/")
    if not prefix.endswith("/"):
        prefix += "/"

    s3 = boto3.client("s3")
    done = list_done_years(s3, bucket, prefix)

    with open(args.tiles_csv) as f:
        rows = list(csv.DictReader(f))

    missing = [r for r in rows if years - done.get(r["tile_id"], set())]
    n_done = len(rows) - len(missing)
    print(
        f"tiles: {n_done}/{len(rows)} complete ({100 * n_done / len(rows):.2f}%), "
        f"{len(missing)} missing at least one of {len(years)} years",
        file=sys.stderr,
    )

    if args.s3_cube:
        from cube_progress import built_tiles
        built, torn = built_tiles(s3, args.s3_cube, [r["tile_id"] for r in rows],
                                  args.resolution, y0, y1)
        not_built = [r for r in missing if r["tile_id"] not in built]
        missing = [r for r in missing if r["tile_id"] in built]
        print(
            f"cube: {len(built)}/{len(rows)} tiles built; of the {len(missing) + len(not_built)} "
            f"missing maps, {len(missing)} can be inferred from the cube and {len(not_built)} "
            f"have no store yet"
            + (f"; {torn} store(s) present but incomplete, skipped" if torn else ""),
            file=sys.stderr,
        )

    if args.limit:
        missing = missing[: args.limit]

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tile_id", "xmin", "ymin", "xmax", "ymax"])
            w.writeheader()
            w.writerows(missing)


if __name__ == "__main__":
    main()
