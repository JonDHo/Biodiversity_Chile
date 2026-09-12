#!/usr/bin/env python3
"""Check that a finished Argo run actually produced usable rasters.

The workflow succeeding only means every pod exited 0. It does not mean the
rasters are right: a tile can be written with every facet band all-NaN (no
clear-sky observations, or the native mask excluded everything), with the
smearing correction silently disabled, or from the wrong checkpoint
directory -- all of which exit 0 and land a plausible-looking GeoTIFF in S3.
This reads the objects back and fails loudly on each of those.

Per (tile, year) raster it checks:
  * CRS EPSG:32719, pixel size == --resolution, grid aligned to the tile
    bounds in the tile CSV (when one is given)
  * band descriptions consistent across every file of the run, ending in the
    three quality layers n_obs / span_days / native
  * at least --min-valid-frac of the native-masked pixels carry a finite
    prediction in every facet band
  * GDAL tags record smearing="oof", the expected checkpoint dir and seed
    count -- i.e. the raster was produced by the deployed ensemble with the
    Duan correction on, not by a diagnostic run

With --manifest it also cross-checks the run's own manifest.csv: every
status="ok" row must have a matching object in S3, and no object may exist
without an "ok" row behind it.

Usage:
    python argo/verify_run.py \
        --s3-dest s3://easido-prod-dc-data-projects/easi-workflows-team/biodiv/outputs/maps/chile_30m_2000_2026 \
        --years 2020-2020 --tiles t123_456 \
        --manifest s3://.../work/runs/<workflow>/infer/chunk-0/argo-test1-0/manifest.csv

Exit status is 0 only when every check passed.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np

QUALITY_BANDS = ["n_obs", "span_days", "native"]
UTM = 32719


def split_s3(uri: str) -> tuple[str, str]:
    u = urlparse(uri)
    return u.netloc, u.path.lstrip("/")


def parse_years(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def read_manifest(s3, uri: str) -> list[dict]:
    bucket, key = split_s3(uri)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    return list(csv.DictReader(io.StringIO(body)))


def check_raster(path: Path, args, problems: list[str], band_names: list[str] | None,
                 tile_bounds: dict[str, tuple[float, float, float, float]]) -> list[str]:
    """Validate one GeoTIFF; returns its band names so later files can be compared."""
    import rasterio

    name = path.name
    tile_id = name.rsplit("_", 1)[0]
    with rasterio.open(path) as src:
        names = [src.descriptions[i] or f"band{i + 1}" for i in range(src.count)]

        if band_names is not None and names != band_names:
            problems.append(f"{name}: bands {names} differ from {band_names}")
        if names[-3:] != QUALITY_BANDS:
            problems.append(f"{name}: last three bands are {names[-3:]}, expected {QUALITY_BANDS}")
        facets = names[:-3]
        if not facets:
            problems.append(f"{name}: no facet bands, only quality layers")

        if src.crs is None or src.crs.to_epsg() != UTM:
            problems.append(f"{name}: CRS is {src.crs}, expected EPSG:{UTM}")
        res = (abs(src.transform.a), abs(src.transform.e))
        if not np.allclose(res, args.resolution):
            problems.append(f"{name}: pixel size {res}, expected {args.resolution} m")
        if tile_id in tile_bounds:
            xmin, ymin, xmax, ymax = tile_bounds[tile_id]
            got = src.bounds
            if not np.allclose([got.left, got.bottom, got.right, got.top],
                               [xmin, ymin, xmax, ymax], atol=args.resolution):
                problems.append(f"{name}: bounds {tuple(got)} do not match the tile CSV "
                                f"{(xmin, ymin, xmax, ymax)}")

        native = src.read(names.index("native") + 1)
        n_native = int(np.nansum(native > 0))
        if n_native == 0:
            problems.append(f"{name}: native mask is empty -- nothing was predictable here")

        for b in facets:
            arr = src.read(names.index(b) + 1)
            inside = np.isfinite(arr) & (native > 0)
            frac = inside.sum() / n_native if n_native else 0.0
            if frac < args.min_valid_frac:
                problems.append(f"{name}: band {b} finite on {frac:.1%} of native pixels "
                                f"(< {args.min_valid_frac:.0%})")
            if inside.any() and not np.isfinite(np.nanstd(arr[inside])):
                problems.append(f"{name}: band {b} has no finite spread")
            elif inside.any() and float(np.nanstd(arr[inside])) == 0.0:
                problems.append(f"{name}: band {b} is constant over the tile")

        tags = src.tags()
        if args.require_smearing and tags.get("smearing") != "oof":
            problems.append(f"{name}: smearing={tags.get('smearing')!r}, expected 'oof' "
                            "(diagnostic run, not publishable)")
        if args.expect_seeds and tags.get("n_seeds") != str(args.expect_seeds):
            problems.append(f"{name}: n_seeds={tags.get('n_seeds')!r}, "
                            f"expected {args.expect_seeds}")
        if args.expect_ckpt and args.expect_ckpt not in (tags.get("ckpt_dir") or ""):
            problems.append(f"{name}: ckpt_dir={tags.get('ckpt_dir')!r} does not contain "
                            f"{args.expect_ckpt!r}")
    return names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--s3-dest", required=True, help="s3:// prefix the rasters were written to")
    p.add_argument("--years", default="2000-2026")
    p.add_argument("--tiles", default="", help="comma-separated tile_ids to check (default: all found)")
    p.add_argument("--tiles-csv", default="", help="tile_id,xmin,ymin,xmax,ymax, to check georeferencing")
    p.add_argument("--manifest", default="", help="s3:// or local manifest.csv of the run")
    p.add_argument("--resolution", type=int, default=30)
    p.add_argument("--min-valid-frac", type=float, default=0.05,
                   help="minimum finite predictions per facet band, as a fraction of native pixels")
    p.add_argument("--require-smearing", action="store_true", default=True)
    p.add_argument("--no-require-smearing", action="store_false", dest="require_smearing")
    p.add_argument("--expect-seeds", type=int, default=5)
    p.add_argument("--expect-ckpt", default="", help="substring the raster's ckpt_dir tag must contain")
    args = p.parse_args()

    years = parse_years(args.years)
    want_tiles = {t for t in args.tiles.split(",") if t}
    problems: list[str] = []
    s3 = boto3.client("s3")

    tile_bounds: dict[str, tuple[float, float, float, float]] = {}
    if args.tiles_csv:
        with open(args.tiles_csv) as f:
            for r in csv.DictReader(f):
                tile_bounds[r["tile_id"]] = (float(r["xmin"]), float(r["ymin"]),
                                             float(r["xmax"]), float(r["ymax"]))

    bucket, prefix = split_s3(args.s3_dest)
    prefix = prefix.rstrip("/") + "/"
    found: dict[tuple[str, int], str] = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = key.rsplit("/", 1)[-1]
            if not name.endswith(".tif"):
                continue
            stem, _, year = name[:-4].rpartition("_")
            if not year.isdigit():
                continue
            if want_tiles and stem not in want_tiles:
                continue
            if int(year) not in years:
                continue
            if obj["Size"] == 0:
                problems.append(f"{name}: zero-byte object")
            found[(stem, int(year))] = key

    print(f"found {len(found)} raster(s) under {args.s3_dest} "
          f"for {len(want_tiles) or 'all'} tile(s), years {years[0]}-{years[-1]}", file=sys.stderr)
    if not found:
        print("FAIL: no rasters written", file=sys.stderr)
        sys.exit(1)

    if want_tiles:
        for t in sorted(want_tiles):
            missing = [y for y in years if (t, y) not in found]
            if missing:
                problems.append(f"{t}: no raster for year(s) {missing}")

    band_names: list[str] | None = None
    with tempfile.TemporaryDirectory() as td:
        for (tile_id, year), key in sorted(found.items()):
            local = Path(td) / f"{tile_id}_{year}.tif"
            s3.download_file(bucket, key, str(local))
            band_names = check_raster(local, args, problems, band_names, tile_bounds)
            local.unlink()
    print(f"bands: {band_names}", file=sys.stderr)

    if args.manifest:
        rows = (read_manifest(s3, args.manifest) if args.manifest.startswith("s3://")
                else list(csv.DictReader(open(args.manifest))))
        ok = {(r["tile_id"], int(r["year"])) for r in rows if r["status"] == "ok"}
        bad = [r for r in rows if r["status"] != "ok"]
        for r in bad:
            print(f"manifest: {r['tile_id']} {r['year']} status={r['status']} "
                  f"{r.get('error', '')}", file=sys.stderr)
        for k in sorted(ok - set(found)):
            problems.append(f"manifest says ok but no object in S3: {k[0]} {k[1]}")
        for k in sorted(set(found) - ok):
            problems.append(f"object in S3 with no ok manifest row: {k[0]} {k[1]}")
        print(f"manifest: {len(ok)} ok, {len(bad)} not ok", file=sys.stderr)

    if problems:
        print(f"\nFAIL: {len(problems)} problem(s)", file=sys.stderr)
        for m in problems:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)
    print("\nOK: every check passed", file=sys.stderr)


if __name__ == "__main__":
    main()
