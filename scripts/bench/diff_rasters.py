"""Are two map runs bit-identical? (the gate protocol from docs/21 section 8.8)

Compares every band of every GeoTIFF in two output directories with
``np.array_equal(equal_nan=True)`` -- NaN must match NaN, not merely "both non-finite" -- plus
the manifest rows, which carry ``n_pred`` and the per-year status and would catch a change that
somehow left the pixels alone.

Bitwise rather than "close": every numerical change in this path so far has been exactly
reproducible, and holding that line is what makes a diff meaningful. When a change *cannot* be
bit-identical (a GPU, a folded BatchNorm), ``--tol`` reports the worst deviation instead, and
the tolerance is meant to be declared in advance rather than discovered here.

``--tol-range`` is the same idea expressed per band instead of in absolute units, and it is the
form a whole-raster gate wants: the ten bands are different facets whose units differ by orders
of magnitude, so one absolute number is either vacuous for some of them or impossible for
others. The tolerance for a band is ``FRAC * (p99 - p1)`` of that band **in A**, i.e. a fraction
of its own dynamic range; p99/p1 rather than max/min so one clipped outlier cannot inflate the
allowance. The NaN pattern is never tolerated: it comes from the mask and the finite-check,
which are integer and logical, so a difference there is a bug and not rounding.

Usage:
    python scripts/bench/diff_rasters.py DIR_A DIR_B [--tol 1e-6 | --tol-range 1e-4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def read_bands(path: Path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read(), src.descriptions, dict(src.tags())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--tol", type=float, default=None,
                    help="allow this max absolute deviation instead of requiring bit-identity")
    ap.add_argument("--tol-range", type=float, default=None, dest="tol_range",
                    help="allow this fraction of each band's own p99-p1 range in A")
    args = ap.parse_args()
    if args.tol is not None and args.tol_range is not None:
        ap.error("--tol and --tol-range are two ways to say the same thing; pick one")

    ta = sorted(p.name for p in args.a.rglob("*.tif"))
    tb = sorted(p.name for p in args.b.rglob("*.tif"))
    if ta != tb:
        print(f"FAIL: different rasters\n  only in A: {set(ta) - set(tb)}\n"
              f"  only in B: {set(tb) - set(ta)}")
        return 1
    if not ta:
        print("FAIL: no rasters found")
        return 1

    worst, bad, checked = 0.0, 0, 0
    worst_frac = 0.0
    for name in ta:
        pa = next(args.a.rglob(name))
        pb = next(args.b.rglob(name))
        va, desc, _ = read_bands(pa)
        vb, _, _ = read_bands(pb)
        if va.shape != vb.shape:
            print(f"FAIL {name}: shape {va.shape} vs {vb.shape}")
            bad += 1
            continue
        for i in range(va.shape[0]):
            checked += 1
            label = (desc[i] if desc and desc[i] else f"band{i + 1}")
            if np.array_equal(va[i], vb[i], equal_nan=True):
                continue
            d = np.abs(np.nan_to_num(va[i]) - np.nan_to_num(vb[i]))
            nan_mismatch = int((np.isnan(va[i]) != np.isnan(vb[i])).sum())
            worst = max(worst, float(d.max()))
            allow, frac = args.tol, None
            if args.tol_range is not None:
                fin = va[i][np.isfinite(va[i])]
                span = float(np.percentile(fin, 99) - np.percentile(fin, 1)) if fin.size else 0.0
                allow = args.tol_range * span
                frac = float(d.max() / span) if span > 0 else float("inf")
                worst_frac = max(worst_frac, frac)
            ok = allow is not None and d.max() <= allow and nan_mismatch == 0
            note = f", {frac:.2e} of p99-p1 range" if frac is not None else ""
            if not ok:
                bad += 1
                print(f"  DIFF {name} [{label}]: max abs {d.max():.3e}{note}, "
                      f"cells {int((d > 0).sum()):,}, NaN-pattern mismatches {nan_mismatch:,}")
            elif frac is not None and d.max() > 0:
                print(f"  ok   {name} [{label}]: max abs {d.max():.3e}{note}")

    # The manifest carries n_pred and per-year status: a change that left every pixel alone
    # but altered which years ran would pass the raster check and fail here.
    ma, mb = next(args.a.rglob("manifest.csv"), None), next(args.b.rglob("manifest.csv"), None)
    if ma and mb:
        import csv

        # `seconds`/`load_seconds` are wall clock and `file` is the output directory: all three
        # differ between any two runs by construction and say nothing about correctness. Every
        # other column is substantive -- n_pred, the grid span, the per-year status -- and a
        # change that left the pixels alone but moved one of those must still fail.
        volatile = {"seconds", "load_seconds", "file"}

        def rows(path):
            with open(path, newline="") as fh:
                return [{k: v for k, v in r.items() if k not in volatile}
                        for r in csv.DictReader(fh)]

        ra, rb = rows(ma), rows(mb)
        same = ra == rb
        print(f"  manifest ({len(ra)} rows, ignoring {'/'.join(sorted(volatile))}): "
              f"{'identical' if same else 'DIFFERS'}")
        if not same:
            for x, y in zip(ra, rb):
                for k in x:
                    if x[k] != y[k]:
                        print(f"    {x.get('tile_id')} {x.get('year')} {k}: {x[k]} vs {y[k]}")
        bad += not same

    print(f"\n{len(ta)} rasters x {checked // len(ta)} bands checked")
    if bad == 0:
        if args.tol_range is not None:
            print(f"VERDICT: WITHIN DECLARED TOLERANCE {args.tol_range:.1e} of each band's "
                  f"p99-p1 range (worst {worst_frac:.2e} of range, {worst:.3e} absolute)")
        else:
            print("VERDICT: ALL BIT-IDENTICAL" if args.tol is None else
                  f"VERDICT: WITHIN TOLERANCE {args.tol:.1e} (worst {worst:.3e})")
        return 0
    print(f"VERDICT: {bad} MISMATCHES (worst abs {worst:.3e}"
          + (f", {worst_frac:.2e} of range)" if args.tol_range is not None else ")"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
