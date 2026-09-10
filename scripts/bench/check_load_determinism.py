"""Is `dc.load` reproducible? The premise of the whole Zarr gate. (docs/24 S2.0)

The Zarr consistency gate is "bit-identical to `dc.load`". That is meaningless if `dc.load`
itself is not reproducible, so this runs first and cheaply.

Why it might not be. `mapinfer.load_kndvi` concatenates one `dc.load` per product and then
`.sortby("time")`. xarray's `sortby` uses `np.lexsort`, which **is** stable, so two acquisitions
sharing a solar day keep the order they arrived in -- but that only pins the tie order if the
index hands back datasets in the same order every time, which nobody has checked. The stable
argsort in `interp_common_grid` makes tie order **observable**, so an unstable index order would
change the curves.

Run twice in separate processes, then compare:

    PYTHONPATH=src python scripts/bench/check_load_determinism.py --out /tmp/a.npz
    PYTHONPATH=src python scripts/bench/check_load_determinism.py --out /tmp/b.npz
    PYTHONPATH=src python scripts/bench/check_load_determinism.py --compare /tmp/a.npz /tmp/b.npz

Separate processes on purpose: within one process a warm index cache could hide exactly the
instability being looked for.

If the two runs differ, that is a finding in its own right and not only about Zarr -- it would
mean the published rasters are not exactly reproducible either, and the Zarr gate has to become
a declared tolerance rather than bit-identity.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# t18_600 (Cauquenes): the tile with published baselines in docs/21 sections 6, 8.7 and 8.8.
DEFAULT_BBOX = (179820.0, 5994000.0, 189810.0, 6003990.0)


def capture(bbox, y0: int, y1: int, resolution: int, out: Path) -> None:
    import datacube
    from datacube.utils.aws import configure_s3_access

    from biodiv import mapinfer as mi

    configure_s3_access(aws_unsigned=False, requester_pays=True)
    dc = datacube.Datacube(app="biodiv-load-determinism")
    da = mi.load_kndvi(dc, bbox, y0, y1, resolution=resolution)
    if da is None:
        raise SystemExit("load_kndvi returned None -- no data for that window")
    if hasattr(da.data, "compute"):
        da = da.compute(scheduler="threads", num_workers=4)
    # `sensor` rides along as a coordinate: it is not used downstream, but it records which
    # product each acquisition came from, which is exactly what a tie-order change would move.
    sensor = da.coords["sensor"].values.astype(str) if "sensor" in da.coords else np.array([])
    np.savez(out, values=np.asarray(da.values, np.float32), times=da.time.values,
             x=da.x.values, y=da.y.values, sensor=sensor)
    print(f"wrote {out}: {da.sizes} -- {np.isfinite(da.values).mean():.1%} finite")


def compare(a: Path, b: Path) -> int:
    A, B = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    bad = 0
    for k in ("times", "x", "y", "sensor"):
        same = A[k].shape == B[k].shape and bool((A[k] == B[k]).all())
        print(f"  {k:8s} identical: {same}")
        bad += not same
    va, vb = A["values"], B["values"]
    if va.shape != vb.shape:
        print(f"  values   SHAPE DIFFERS {va.shape} vs {vb.shape}")
        return 1
    same = np.array_equal(va, vb, equal_nan=True)
    print(f"  values   identical: {same}  (shape {va.shape})")
    if not same:
        d = np.abs(np.nan_to_num(va) - np.nan_to_num(vb))
        nan_mismatch = int((np.isnan(va) != np.isnan(vb)).sum())
        print(f"           max abs diff {d.max():.3e}, differing cells {int((d > 0).sum()):,},"
              f" NaN-pattern mismatches {nan_mismatch:,}")
    bad += not same
    print("\nVERDICT:", "REPRODUCIBLE -- bit-identity is a valid gate" if bad == 0 else
          "NOT REPRODUCIBLE -- the Zarr gate must become a declared tolerance")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", type=Path, nargs=2, metavar=("A", "B"))
    ap.add_argument("--years", default="2018-2020",
                    help="short span by default: order instability shows up in ~200 datasets "
                         "just as well as in 1,650, and costs minutes instead of 25 of them")
    ap.add_argument("--resolution", type=int, default=30)
    ap.add_argument("--bbox", type=float, nargs=4, default=list(DEFAULT_BBOX))
    a = ap.parse_args()

    if a.compare:
        raise SystemExit(compare(*a.compare))
    if not a.out:
        ap.error("need --out or --compare")
    y0, y1 = (int(v) for v in a.years.split("-"))
    capture(tuple(a.bbox), y0, y1, a.resolution, a.out)


if __name__ == "__main__":
    main()
