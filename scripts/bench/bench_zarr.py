"""What does a Zarr tile actually cost to store and read? (docs/24 S2.2)

Three numbers the Zarr plan is currently guessing at, and one of them is a decision point:

1. **Compressed size.** `docs/24` estimates ~180 MB per tile from an assumed ~55 % NaN. The
   NaN fraction is measurable, and on the first tile checked it is nearer 40 %, so the estimate
   needs testing rather than trusting.

2. **Read scaling with threads -- the decision point.** The performance case for Zarr in
   `docs/24` section 2 point 3 is that blosc releases the GIL where GDAL does not, so reads
   should scale with threads instead of plateauing at 3.4x the way COG header parsing does
   (`docs/21` section 8.1). **If reads plateau too, that argument collapses** and only the
   re-run economics survive.

3. **Chunk alignment.** Chunks that do not line up with the 333 px tile lattice make a tile
   pull data it does not use. This measures the amplification directly by reading the same
   region from an aligned and a deliberately misaligned store.

Input is an .npz written by `check_load_determinism.py`, so this reuses a real tile rather than
synthesising one -- compression ratios depend entirely on the actual NaN pattern and value
distribution.

Caveat carried into the write-up: this reads from local disk, not S3. The GIL question (does
decompression parallelise?) transfers, because that is CPU work either way; the absolute
latencies do not.

Usage:
    PYTHONPATH=src python scripts/bench/bench_zarr.py --npz /path/to/load_a.npz
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import xarray as xr


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def build(da: xr.DataArray, root: Path, chunks: tuple[int, int, int], level: int) -> tuple:
    """Write one Zarr store and return (path, bytes, seconds)."""
    path = root / f"t{chunks[0]}_y{chunks[1]}_x{chunks[2]}_z{level}.zarr"
    if path.exists():
        shutil.rmtree(path)
    try:                                    # zarr-python 3 wants its own codec objects
        from zarr.codecs import BloscCodec, BloscShuffle
        comp = {"compressors": [BloscCodec(cname="zstd", clevel=level,
                                           shuffle=BloscShuffle.shuffle)]}
    except ImportError:                     # zarr-python 2 takes a numcodecs codec
        from numcodecs import Blosc
        comp = {"compressor": Blosc(cname="zstd", clevel=level, shuffle=Blosc.SHUFFLE)}
    enc = {"kndvi": dict(chunks=chunks, **comp)}
    t = time.perf_counter()
    da.to_dataset(name="kndvi").to_zarr(path, encoding=enc, consolidated=True, mode="w")
    return path, dir_bytes(path), time.perf_counter() - t


def decompress_scaling(path: Path, threads: int, repeat: int = 4) -> float:
    """Seconds to decompress every chunk on ``threads`` threads, store opened ONCE.

    Opening the store, parsing consolidated metadata and building a dask graph are fixed
    per-read costs that do not parallelise, and they are comparable to the decompression
    itself for a single tile. Timing them inside the loop makes any codec look like it does
    not scale -- so the store is opened once outside the timer and only the chunk reads are
    measured. This is the question `docs/24` actually asks: does blosc release the GIL the way
    GDAL's header parsing does not (`docs/21` section 8.1)?
    """
    import zarr
    from concurrent.futures import ThreadPoolExecutor

    arr = zarr.open_array(str(path / "kndvi"), mode="r")
    cy, cx = arr.chunks[1], arr.chunks[2]
    blocks = [(slice(None), slice(y, min(y + cy, arr.shape[1])),
               slice(x, min(x + cx, arr.shape[2])))
              for y in range(0, arr.shape[1], cy) for x in range(0, arr.shape[2], cx)]

    def pull(sl):
        return int(arr[sl].size)

    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(pull, blocks))                      # warm the page cache
        best = float("inf")
        for _ in range(repeat):
            t = time.perf_counter()
            list(ex.map(pull, blocks))
            best = min(best, time.perf_counter() - t)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="scratch dir for the stores")
    ap.add_argument("--full-dates", type=int, default=1811,
                    help="production span, to extrapolate this short-span tile")
    a = ap.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)

    z = np.load(a.npz, allow_pickle=False)
    v = z["values"]
    da = xr.DataArray(v, coords={"time": z["times"], "y": z["y"], "x": z["x"]},
                      dims=("time", "y", "x"), name="kndvi")
    T, NY, NX = v.shape
    finite = float(np.isfinite(v).mean())
    raw = v.nbytes
    print(f"tile {NY}x{NX}, {T} dates, {finite:.1%} finite ({1 - finite:.1%} NaN), "
          f"raw {raw / 1e6:.0f} MB\n")

    # Full time axis, whole tile in one spatial block: the access pattern is "the entire time
    # series of a spatial region", so this is the layout docs/24 proposes.
    print("Layouts -- full time axis, whole-tile spatial block")
    print(f"  {'chunks (t,y,x)':22s} {'MB':>8s} {'ratio':>7s} {'write s':>8s} {'MB/tile @full':>14s}")
    stores = {}
    for chunks, level in [((T, NY, NX), 3), ((T, NY, NX), 5), ((T, 167, 167), 5),
                          ((T, 64, 64), 5)]:
        p, b, secs = build(da, a.root, chunks, level)
        stores[(chunks, level)] = p
        print(f"  {str(chunks):22s} {b / 1e6:8.1f} {raw / b:6.2f}x {secs:8.1f} "
              f"{b / 1e6 * a.full_dates / T:14.0f}")

    # Read scaling needs MANY chunks: a single-chunk store gives threads nothing to divide,
    # and would report a flat line that says nothing about blosc. Use the 64 px layout (36
    # chunks here) and repeat the read, since one tile decompresses too fast to time once.
    many = stores[((T, 64, 64), 5)]
    print("\nDecompression scaling -- does blosc release the GIL?"
          " (local disk, store opened once)")
    print(f"  {'threads':>8s} {'s':>8s} {'speedup':>8s}")
    base = None
    for n in (1, 2, 4, 8):
        secs = decompress_scaling(many, n)
        base = secs if base is None else base
        print(f"  {n:8d} {secs:8.3f} {base / secs:7.2f}x")

    # Alignment: the amplification only appears when a tile sits INSIDE a larger array, so
    # build a 2x2-tile mosaic and read one tile-sized, tile-aligned window from each layout.
    print("\nChunk alignment -- same 333x333 window out of a 666x666 mosaic")
    big = xr.concat([xr.concat([da, da], dim="y")] * 2, dim="x")
    big = big.assign_coords(y=np.arange(big.sizes["y"], dtype="float64"),
                            x=np.arange(big.sizes["x"], dtype="float64"))
    for label, ch in (("aligned (333 px chunks)", (T, NY, NX)),
                      ("misaligned (256 px chunks)", (T, 256, 256))):
        p_, b_, _ = build(big, a.root, ch, 5)
        t = time.perf_counter()
        xr.open_zarr(p_, consolidated=True)["kndvi"].isel(
            y=slice(0, NY), x=slice(0, NX)).values
        print(f"  {label:28s} {time.perf_counter() - t:6.2f} s   store {b_ / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
