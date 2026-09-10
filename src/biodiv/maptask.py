"""One tile, end to end: load, curves, ensemble, GeoTIFF -- wherever it runs.

`scripts/73_map_inference.py` used to hold this loop inline, which was fine while the only
place it ran was the pod. It now also runs on dask-gateway workers, and the two must not
drift: `scripts/74_check_map_consistency.py` gates map production by proving that the map
path reproduces the training path exactly, and that proof is worth nothing if the code the
gate checked is not the code the workers ran. So the loop lives here, and both callers
import it.

WHAT A GATEWAY WORKER DOES NOT HAVE, and how each gap is closed (all four measured in
`logs/gw_probe.log`, on a real worker):

* **the home directory** -- not mounted. The MapBiomas rasters are read from the scratch
  bucket instead -- which is where `mapbiomas.rasters_dir` points by default, so no
  configuration is needed and `RASTERS_DIR_ENV` is left for working against a copy -- the
  checkpoints are shipped as bytes, and the GeoTIFFs are written back to the scratch bucket
  rather than to `results/maps`.
* **the `biodiv` package** -- not installed. The caller uploads it as a zip.
* **torch** -- present, 2.12.0+cpu, and the whole worker image matches the pod version for
  version, which is what makes the scikit-learn 1.3.1 pickles inside the checkpoints safe
  to unpickle there (`docs/21` section 7 left that open; it is now closed).
* **the ODC index** -- reachable: the gateway injects `DB_*`, so `dc.load` works on the
  worker and the tile does not have to be loaded on the client and shipped.
"""

from __future__ import annotations

import io
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import mapbiomas as mb
from . import mapinfer as mi

UTM = "EPSG:32719"


# --------------------------------------------------------------------------------------
# geometry and output
# --------------------------------------------------------------------------------------

def to_utm(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    return tr.transform(np.asarray(lon, float), np.asarray(lat, float))


def to_lonlat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer
    tr = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    return tr.transform(np.asarray(x, float), np.asarray(y, float))


def grid_transform(template):
    from rasterio.transform import Affine
    x, y = template.x.values, template.y.values
    res = float(abs(x[1] - x[0]))
    return Affine(res, 0.0, float(x.min()) - res / 2, 0.0, -res, float(y.max()) + res / 2)


def native_mask(year: int, template, crs: str = UTM) -> tuple[np.ndarray, dict]:
    """Native-vegetation mask on the tile grid from the nearest MapBiomas annual map."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import from_bounds

    path, used, delta = mb.year_map(year)
    x, y = template.x.values, template.y.values
    res = float(abs(x[1] - x[0]))
    # bounds in lon/lat with a margin, so the reprojection has full coverage
    xs = np.array([x.min() - res, x.max() + res, x.min() - res, x.max() + res])
    ys = np.array([y.min() - res, y.min() - res, y.max() + res, y.max() + res])
    lons, lats = to_lonlat(xs, ys)
    pad = 0.01
    with rasterio.open(path) as src:
        win = from_bounds(lons.min() - pad, lats.min() - pad, lons.max() + pad,
                          lats.max() + pad, src.transform)
        arr = src.read(1, window=win)
        src_tr = src.window_transform(win)
        src_crs = src.crs
    dst = np.zeros((len(y), len(x)), dtype=arr.dtype)
    dst_tr = grid_transform(template)
    reproject(arr, dst, src_transform=src_tr, src_crs=src_crs, dst_transform=dst_tr,
              dst_crs=crs, resampling=Resampling.nearest)
    native = np.isin(dst, list(mb.NATIVE))
    return native, dict(map_year=int(used), map_delta=int(delta), map_path=str(path))


def write_tile_year(path: str | Path, template, layers: dict[str, np.ndarray], tags: dict,
                    crs: str = UTM) -> None:
    import rasterio
    names = list(layers)
    h, w = layers[names[0]].shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=len(names),
                       dtype="float32", crs=crs, transform=grid_transform(template),
                       nodata=np.nan, tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate", predictor=3, BIGTIFF="IF_SAFER") as dst:
        for i, n in enumerate(names, 1):
            dst.write(np.asarray(layers[n], np.float32), i)
            dst.set_band_description(i, n)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})


def _emit(dest: str, name: str, template, layers: dict[str, np.ndarray], tags: dict) -> str:
    """Write one tile-year, to a local directory or to an ``s3://`` prefix.

    S3 goes through a temporary local file and `boto3.upload_file` rather than GDAL's
    ``/vsis3`` writer: the file is ~1.5 MB and written once, so there is nothing to gain
    from streaming it, and a failed multipart upload leaves no half-written object behind
    that a ``--resume`` would mistake for a finished tile-year.
    """
    if not dest.startswith("s3://"):
        path = Path(dest) / name
        write_tile_year(path, template, layers, tags)
        return str(path)
    import tempfile
    import boto3
    bucket, _, prefix = dest[len("s3://"):].partition("/")
    key = f"{prefix.rstrip('/')}/{name}"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / name
        write_tile_year(tmp, template, layers, tags)
        boto3.client("s3").upload_file(str(tmp), bucket, key)
    return f"s3://{bucket}/{key}"


# --------------------------------------------------------------------------------------
# the tile loop
# --------------------------------------------------------------------------------------

@dataclass
class TileConfig:
    """Everything a tile needs that is not the tile itself. Picklable by construction."""

    years: tuple[int, ...]
    dest: str                                   # local directory or s3:// prefix
    tags: dict                                  # run metadata stamped into every GeoTIFF
    resolution: int = 30
    area_m2: float = 900.0
    stratum: str = "basal"
    mask: str = "mapbiomas"
    batch: int = 8192
    load_threads: int = 0
    load_client: bool = False                   # use the ambient distributed client instead
    torch_threads: int = 1
    mapbiomas_dir: str = ""
    device: str = "cpu"
    smearing_exact: bool = False                # evaluate the 128 draws instead of tabulating
    interp_block: int = 20000                   # pixel columns per `interp_common_grid` pass


#: Directory for the development tile cache. Unset -- which is what the Argo path and every
#: production run leave it -- means no cache at all, and `load_tile` behaves exactly as
#: before. See `_cache_read`.
TILE_CACHE_ENV = "BIODIV_TILE_CACHE"

#: Bumped whenever the cached layout or the meaning of the array changes, so a stale cache
#: is ignored rather than silently feeding the wrong numbers into a comparison.
_CACHE_VERSION = 1


def _cache_dir(root: str, tile: dict, cfg: TileConfig) -> Path:
    span = f"{cfg.years[0] - 2}_{cfg.years[-1]}"
    return Path(root) / f"v{_CACHE_VERSION}_{tile['tile_id']}_{cfg.resolution}m_{span}"


def _cache_read(root: str, tile: dict, cfg: TileConfig):
    """A previously loaded tile, or None.

    **Development scaffolding, not a production feature.** A real run visits each tile once,
    so this would never hit and would write ~0.7 GB per tile for nothing; it exists because
    iterating on the year loop otherwise pays the full Landsat load (measured 714 s for the
    1998-2026 span) on every attempt. The arrays are memory-mapped, so a hit costs no copy
    and the year loop pages in only the window it touches.

    The geometry is re-checked against the tile rather than trusted from the key, so a cache
    written for a different grid cannot be picked up by a same-named tile.
    """
    import json
    d = _cache_dir(root, tile, cfg)
    meta = d / "meta.json"
    if not meta.exists():
        return None
    try:
        m = json.loads(meta.read_text())
        if m.get("bbox") != [tile["xmin"], tile["ymin"], tile["xmax"], tile["ymax"]]:
            return None
        import xarray as xr
        values = np.load(d / "values.npy", mmap_mode="r")
        times = np.load(d / "times.npy")
        x, y = np.load(d / "x.npy"), np.load(d / "y.npy")
    except Exception:                                               # noqa: BLE001
        return None                                                 # a torn cache is a miss
    if values.shape != (len(times), len(y), len(x)):
        return None
    return xr.DataArray(values, coords={"time": times, "y": y, "x": x},
                        dims=("time", "y", "x"), name="kndvi")


def _cache_write(root: str, tile: dict, cfg: TileConfig, da) -> None:
    import json
    d = _cache_dir(root, tile, cfg)
    tmp = d.with_name(d.name + f".part{os.getpid()}")
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        np.save(tmp / "values.npy", np.asarray(da.values, np.float32))
        np.save(tmp / "times.npy", da.time.values)
        np.save(tmp / "x.npy", da.x.values)
        np.save(tmp / "y.npy", da.y.values)
        (tmp / "meta.json").write_text(json.dumps(
            {"bbox": [tile["xmin"], tile["ymin"], tile["xmax"], tile["ymax"]],
             "resolution": cfg.resolution, "version": _CACHE_VERSION}))
        tmp.replace(d)                       # atomic: a half-written cache is never read
    except Exception as e:                                          # noqa: BLE001
        print(f"tile cache: not written ({type(e).__name__}: {e})", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)


#: Root of the materialised kNDVI cube: one Zarr store per tile, named like the dev cache.
#: Unset -- which is what every run leaves it today -- means `load_tile` behaves exactly as
#: before. Unlike `BIODIV_TILE_CACHE` this is a *production* candidate, not scaffolding: it is
#: written once by a separate pass and read by every run after. See `docs/24`.
TILE_ZARR_ENV = "BIODIV_TILE_ZARR"

#: Dates per time chunk. The Fase 0 stores held the whole time axis as one chunk, which reads
#: and writes as one large object; splitting it is **strictly** better, and it is free because
#: blosc already compresses in blocks internally -- the compressed store came out identical to
#: the byte at every layout tried (335,8 MB on t18_600, `logs/bench_zarr_s3.log`). Measured on
#: local disk, write / read: whole 3,37 / 1,53 s, 188 dates 1,36 / 0,45, 94 dates 1,31 / 0,40.
#: From S3 the same tile reads in 3,1 s against 4,1 s for the single chunk. 128 sits between
#: the two best measured points and keeps a chunk near 28 MB compressed, well above the ~10 MB
#: floor under which a store on S3 turns into an object-listing problem (`docs/24` sec. 3).
_ZARR_TCHUNK = 128


def _zarr_path(root: str, tile: dict, cfg: TileConfig):
    """One tile's store: a local `Path`, or an ``s3://`` URI string if `root` is one.

    Both are handed to `zarr.open_group` as a string; the two types are kept apart only so
    the local path can still be stat-ed and renamed, neither of which S3 has.
    """
    span = f"{cfg.years[0] - 2}_{cfg.years[-1]}"
    name = f"v{_CACHE_VERSION}_{tile['tile_id']}_{cfg.resolution}m_{span}.zarr"
    if root.startswith("s3://"):
        return f"{root.rstrip('/')}/{name}"
    return Path(root) / name


def _zarr_read(root: str, tile: dict, cfg: TileConfig):
    """One tile's kNDVI from the materialised cube, or None.

    The arrays are read straight out of zarr rather than through `xarray.open_zarr`, and that
    is deliberate: xarray would CF-encode ``time`` on write and decode it on read, which can
    change datetime resolution on the round trip. That would break bit-identity with `dc.load`
    for a reason that has nothing to do with Zarr, and bit-identity is the whole gate
    (`docs/24` Fase 0). Storing the datetime64 as its int64 view sidesteps the question.

    The geometry is re-checked against the tile, so a store written for a different grid cannot
    be picked up by a same-named tile -- same guard as `_cache_read`.
    """
    p = _zarr_path(root, tile, cfg)
    if isinstance(p, Path) and not p.exists():
        return None
    try:
        import xarray as xr
        import zarr
        g = zarr.open_group(str(p), mode="r")
        if list(g.attrs.get("bbox", ())) != [tile["xmin"], tile["ymin"],
                                             tile["xmax"], tile["ymax"]]:
            return None
        values = g["values"][:]
        times = g["times"][:].view("datetime64[ns]")
        x, y = g["x"][:], g["y"][:]
    except Exception:                                               # noqa: BLE001
        return None                                                 # a torn store is a miss
    if values.shape != (len(times), len(y), len(x)):
        return None
    return xr.DataArray(values, coords={"time": times, "y": y, "x": x},
                        dims=("time", "y", "x"), name="kndvi")


def _zarr_exists(root: str, tile: dict, cfg: TileConfig) -> bool:
    """Whether a complete store for this tile is already there, without reading it.

    `_zarr_read` pulls the whole array down, which is the wrong question to ask 5.769 times
    when resuming a build pass: it would drag the entire cube across the network to decide it
    has nothing to do. This touches only the group metadata -- one small GET on S3 -- and
    applies the two guards that decide the answer anyway: the completion marker (`bbox`, which
    `zarr_write` puts last) and the geometry.

    For a whole run, listing the prefix once is cheaper still; this is the per-tile question.
    """
    p = _zarr_path(root, tile, cfg)
    if isinstance(p, Path) and not p.exists():
        return False
    try:
        import zarr
        g = zarr.open_group(str(p), mode="r")
    except Exception:                                               # noqa: BLE001
        return False
    return list(g.attrs.get("bbox", ())) == [tile["xmin"], tile["ymin"],
                                             tile["xmax"], tile["ymax"]]


def zarr_write(root: str, tile: dict, cfg: TileConfig, da, clevel: int = 5,
               tchunk: int = _ZARR_TCHUNK):
    """Materialise one tile's kNDVI as a Zarr store. Returns its `Path` or ``s3://`` URI.

    The tile is one spatial block and the time axis is cut into `tchunk` blocks, because the
    access pattern is "the entire time series of a spatial region" -- the year loop slices
    time, never space. Measured on t18_600: neither the zstd level nor the spatial chunking
    changes the size by more than ~2 % (`logs/bench_zarr.log`), so there is nothing to win by
    compressing harder and the layout is chosen for how it reads. ``tchunk=0`` writes the time
    axis as a single chunk, which is what the Fase 0 stores did; `_ZARR_TCHUNK` says why the
    default no longer does.

    One store per tile, so a tile can never straddle another tile's chunk and resumability is
    just "does this store exist" -- the same check `scripts/argo/tile_progress.py` already
    does for GeoTIFFs.
    """
    import zarr
    try:
        from zarr.codecs import BloscCodec, BloscShuffle
        comp = {"compressors": [BloscCodec(cname="zstd", clevel=clevel,
                                           shuffle=BloscShuffle.shuffle)]}
    except ImportError:                                             # zarr-python 2
        from numcodecs import Blosc
        comp = {"compressor": Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.SHUFFLE)}

    v = np.asarray(da.values, np.float32)
    t = np.asarray(da.time.values).astype("datetime64[ns]").view(np.int64)
    chunks = (tchunk if tchunk else v.shape[0], v.shape[1], v.shape[2])
    attrs = {"bbox": [tile["xmin"], tile["ymin"], tile["xmax"], tile["ymax"]],
             "resolution": cfg.resolution, "version": _CACHE_VERSION}

    def fill(target: str) -> None:
        g = zarr.open_group(target, mode="w")
        g.create_array("values", shape=v.shape, chunks=chunks, dtype="float32", **comp)
        g["values"][:] = v
        for name, arr in (("times", t), ("x", np.asarray(da.x.values)),
                          ("y", np.asarray(da.y.values))):
            g.create_array(name, shape=arr.shape, chunks=arr.shape, dtype=arr.dtype)
            g[name][:] = arr
        # Last, and in one shot. `_zarr_read` gates on `bbox`, so a store whose attributes
        # are missing reads as a miss -- which is what makes this the completion marker.
        g.attrs.put(attrs)

    p = _zarr_path(root, tile, cfg)
    if not isinstance(p, Path):
        # S3 has no rename, so the local trick of building under a `.part` name and moving it
        # into place is unavailable. The completion marker takes its place: the chunks are
        # written first and the attributes last, in a single atomic PUT of the group
        # metadata, so an interrupted write leaves a store that reads as a miss rather than
        # one that reads as short. `mode="w"` clears whatever such an attempt left behind.
        fill(p)
        return p

    tmp = p.with_name(p.name + f".part{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        fill(str(tmp))
        shutil.rmtree(p, ignore_errors=True)
        tmp.replace(p)                       # atomic: a half-written store is never read
    except Exception:                                               # noqa: BLE001
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return p


def load_tile(dc, tile: dict, cfg: TileConfig):
    """The whole archive span for one tile, computed.

    The load is 96 % fixed cost -- ~1,800 COG headers rather than bytes (`logs/load_scaling.log`)
    -- so it is done once per tile for ``years[0]-2 .. years[-1]`` and every year is cut from
    it in memory, and it is computed on a *thread* pool. Inside a dask worker the scheduler
    has to be named explicitly: the default would hand the graph back to the distributed
    scheduler that is already running this task.

    Threads are not free scaling, and the measurement says so (`logs/thread_scaling.log`,
    10 km tile, 1,812 dates): 1 thread 1,892 s, 4 threads 643 s (2.9x), 8 threads 562 s
    (3.4x). It flattens because GDAL's COG header parsing holds the GIL -- the same load on
    8 dask *processes* takes 304 s. So threads inside a tile are worth about 3x and no more,
    and the parallelism that actually pays is one process per tile.
    """
    # The materialised cube first: it is the production path when it exists, and it is the
    # cheapest of the three (measured ~1.3 s of decompression against ~700 s of COG headers).
    zroot = os.environ.get(TILE_ZARR_ENV)
    if zroot:
        hit = _zarr_read(zroot, tile, cfg)
        if hit is not None:
            print(f"tile zarr: hit for {tile['tile_id']}", flush=True)
            return hit
    cache = os.environ.get(TILE_CACHE_ENV)
    if cache:
        hit = _cache_read(cache, tile, cfg)
        if hit is not None:
            print(f"tile cache: hit for {tile['tile_id']}", flush=True)
            return hit
    bbox = (tile["xmin"], tile["ymin"], tile["xmax"], tile["ymax"])
    lazy = cfg.load_client or cfg.load_threads > 0
    da = mi.load_kndvi(dc, bbox, cfg.years[0] - 2, cfg.years[-1], resolution=cfg.resolution,
                       dask_chunks={"time": 1} if lazy else None)
    if da is None:
        return None
    if hasattr(da.data, "compute"):
        da = (da.compute() if cfg.load_client
              else da.compute(scheduler="threads", num_workers=cfg.load_threads))
    if cache:
        _cache_write(cache, tile, cfg, da)
    return da


def run_tile(dc, tile: dict, cfg: TileConfig, ens, resid, y_train,
             skip_years: frozenset[int] = frozenset()) -> list[dict]:
    """One tile, every requested year, written out. Returns the manifest rows."""
    todo = [y for y in cfg.years if y not in skip_years]
    if not todo:
        return []
    t0 = time.time()
    da = load_tile(dc, tile, cfg)
    if da is None:
        return [dict(tile_id=tile["tile_id"], year=y, status="no_data") for y in todo]
    t_load = time.time() - t0

    times = da.time.values
    ny, nx = da.sizes["y"], da.sizes["x"]
    obs = da.values.reshape(len(times), ny * nx)
    template = da.isel(time=0)
    lon_c, lat_c = to_lonlat(np.array([(tile["xmin"] + tile["xmax"]) / 2]),
                             np.array([(tile["ymin"] + tile["ymax"]) / 2]))
    terrain = mi.load_terrain(dc, template, float(lat_c[0]), resolution=cfg.resolution)
    topo = {k: v.reshape(-1) for k, v in terrain.items()}
    ctx = mi.context_frame(ens.context_columns, topo, cfg.area_m2, cfg.stratum)
    targets = ens.targets

    rows = []
    for y in todo:
        ty = time.time()
        # The mask comes first, and the window before the interpolation, because the
        # interpolation is the one expensive step and most of the tile is not native: on the
        # measured tiles 25-42 % of pixels survive `run`, and the rest used to be
        # interpolated and thrown away one line later. `year_window` still derives the grid
        # span from every pixel of the tile, so the plot-cube convention is untouched.
        if cfg.mask == "mapbiomas":
            native, minfo = native_mask(y, template)
        else:
            native, minfo = np.ones((ny, nx), bool), {}
        nat = native.reshape(-1)
        t_win, o_win, n_obs, grid, lo, hi = mi.year_window(times, obs, y)
        # `np.isfinite(curves).all(axis=1)` is exactly `n_obs >= MIN_OBS`: a curve comes back
        # NaN only where the pixel had no clear observation at all (then every grid point is
        # NaN, never some of them) or where `interp_common_grid` blanked it for having fewer
        # than MIN_OBS. So which pixels are worth running is known before interpolating.
        usable = n_obs >= mi.MIN_OBS
        run = usable & nat
        pred = np.full((ny * nx, len(targets)), np.nan, np.float32)
        if grid is not None and run.any():
            sel = np.flatnonzero(run)
            curves, _ = mi.interp_common_grid(t_win, np.ascontiguousarray(o_win[:, sel]),
                                              grid, block=cfg.interp_block)
            images = ens.model_inputs(curves)
            pred[run] = ens.predict(images, ctx.iloc[sel],
                                    resid_scaled=resid, y_train=y_train, batch=cfg.batch,
                                    exact=cfg.smearing_exact)
        layers = {t: pred[:, j].reshape(ny, nx) for j, t in enumerate(targets)}
        layers["n_obs"] = n_obs.reshape(ny, nx).astype(np.float32)
        layers["span_days"] = np.where(usable, hi - lo,
                                       np.nan).reshape(ny, nx).astype(np.float32)
        layers["native"] = native.astype(np.float32)
        tags = dict(cfg.tags, year=y, window=f"{y - 2}-01-01/{y}-12-31",
                    grid_first_day=lo, grid_last_day=hi, **minfo)
        where = _emit(cfg.dest, f"{tile['tile_id']}_{y}.tif", template, layers, tags)
        rows.append(dict(tile_id=tile["tile_id"], year=y, status="ok",
                         n_px=ny * nx, n_native=int(nat.sum()), n_pred=int(run.sum()),
                         n_dates_window=int(mi.window_mask(times, y).sum()),
                         grid_first=lo, grid_last=hi, seconds=round(time.time() - ty, 1),
                         load_seconds=round(t_load, 1), file=where))
    return rows


# --------------------------------------------------------------------------------------
# the worker side
# --------------------------------------------------------------------------------------

@dataclass
class Payload:
    """What every worker needs, broadcast once instead of ridden along with each tile.

    The checkpoints are 764 KB for all five seeds and the two arrays are a few thousand
    floats, so this is small; the reason it is scattered rather than closed over is that a
    closure is re-serialised into the graph for every one of the ~5,900 tasks.
    """

    ckpts: list[tuple[str, bytes]] = field(default_factory=list)
    resid: np.ndarray | None = None
    y_train: np.ndarray | None = None


_STATE: dict = {}


def worker_state(payload: Payload, cfg: TileConfig) -> dict:
    """Per-process singletons: the datacube connection and the five-seed ensemble.

    Built on the first tile a worker gets and kept for the rest, because a worker handles
    hundreds of tiles and rebuilding five torch models per tile is pure waste. Keyed on the
    checkpoint bytes, so a payload change cannot be served from a stale cache.
    """
    key = (hash(tuple(b for _, b in payload.ckpts)), cfg.device)
    if key in _STATE:
        return _STATE[key]

    import datacube
    import torch
    from datacube.utils.aws import configure_s3_access

    if cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)
    if cfg.mapbiomas_dir:
        os.environ[mb.RASTERS_DIR_ENV] = cfg.mapbiomas_dir
    # Idempotent, and not redundant with the client-side call: a worker that joined after
    # the client configured the cluster would otherwise read usgs-landsat unsigned and get
    # AccessDenied on every scene.
    configure_s3_access(aws_unsigned=False, requester_pays=True)

    ens = mi.FacetEnsemble([io.BytesIO(b) for _, b in payload.ckpts], device=cfg.device)
    st = dict(dc=datacube.Datacube(app="biodiv-map-tile"), ens=ens)
    _STATE.clear()                      # one ensemble per worker, never a growing cache
    _STATE[key] = st
    return st


def run_tile_remote(tile: dict, payload: Payload, cfg: TileConfig,
                    skip_years: tuple[int, ...] = ()) -> list[dict]:
    """Dask entry point: one whole tile on a gateway worker.

    Failures are returned, not raised. One tile that cannot be read -- a transient S3 error,
    a gap in the archive -- must not take down a run of thousands, and the manifest is the
    place where that is recorded and where ``--resume`` reads it back.
    """
    try:
        st = worker_state(payload, cfg)
        # worker_state is cached for the life of the worker, so its own
        # configure_s3_access ran once, on the first tile. GDAL holds the credential
        # values it was given and cannot renew them from the service-account token the
        # way boto3 does, so a worker outliving the role session starts failing every
        # read with RasterioIOError('The provided token has expired'). Refreshing per
        # tile is what keeps a long-lived worker usable; it only reaches STS when the
        # cached credentials are near expiry.
        from datacube.utils.aws import configure_s3_access
        configure_s3_access(aws_unsigned=False, requester_pays=True)
        return run_tile(st["dc"], tile, cfg, st["ens"], payload.resid, payload.y_train,
                        frozenset(skip_years))
    except Exception as e:                                          # noqa: BLE001
        return [dict(tile_id=tile["tile_id"], year=y, status="error",
                     error=f"{type(e).__name__}: {str(e)[:200]}")
                for y in cfg.years if y not in set(skip_years)]
