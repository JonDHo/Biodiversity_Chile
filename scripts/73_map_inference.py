#!/usr/bin/env python3
"""Multitemporal facet maps from the deployed model ensemble, tile by tile.

For every tile and every target year ``y`` in ``--years``: load the clear-sky kNDVI
observations of the tile for ``y-2 .. y`` from Data Cube Chile, build the 100-step raw
series per pixel exactly as for the training plots (`biodiv.mapinfer`), shape it into
whatever the checkpoint's architecture expects (`FacetEnsemble.model_inputs` -- the raw
curve for the deployed 1D-CNN, `C1D01_curve1d_kndvi`, or the serpentine image for a 2D-CNN
checkpoint), attach the centre-pixel topography (Copernicus GLO-30, same derivatives as
`scripts/03`) and the two constant covariates (plot area, recording protocol), run the five
all-data seed checkpoints, back-transform and average, and write one GeoTIFF per (tile,
year) with the seven facets plus quality layers.

The Landsat archive is read ONCE per tile for the whole ``years[0]-2 .. years[-1]`` span
and every year's window is cut from it in memory: 27 target years share 29 archive years,
so per-year loading would read the same scenes ~3x over.

Decisions that are not defaults but recorded choices (see docs/21_map_inference_spec.md):
``--area-m2`` and ``--stratum`` (the non-mappable covariates), ``--mask`` (native
vegetation from MapBiomas, nearest annual map), the tile size, and the smearing residuals.

Usage:
    # pilot: one 10 km tile around a plot cluster, all years, local dask
    python scripts/73_map_inference.py --bbox -72.50 -36.10 -72.35 -35.95 --years 2000-2026 \\
        --workers 4 --out results/maps --tag pilot
    # full run: whole tiles on dask-gateway workers, GeoTIFFs to the scratch bucket
    python scripts/73_map_inference.py --tiles-file results/figures/tiles_native_10km.csv \\
        --years 2000-2026 --gw-workers 32 --dest s3://BUCKET/PREFIX/outputs/maps/chile \\
        --mapbiomas-dir s3://BUCKET/PREFIX/MapBiomas --resume

Requires BIODIV_UNIFIED=1 and BIODIV_CURVES=_raw100 only for ``--oof-csv``/range clipping
(they load the unified target tables); the model itself needs neither.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import mapbiomas as mbm         # noqa: E402
from biodiv import mapinfer as mi           # noqa: E402
from biodiv import maptask as mt            # noqa: E402
from biodiv import targets as tg            # noqa: E402
from biodiv.maptask import to_utm, to_lonlat  # noqa: E402,F401

UTM = "EPSG:32719"
DEFAULT_CKPT = ("results/models_unified_topofix/"
                "C1D01_curve1d_kndvi_raw100_pg-all_unified_ctr_FINAL_alldata/final")
DEFAULT_OOF = ("results/models_unified_topofix/C1D01_curve1d_kndvi_raw100_pg-all_unified_ctr/"
               "kfold5_block20_unified/oof_predictions.csv")


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------

def extent_from_bbox(bbox_ll: list[float]) -> tuple[float, float, float, float]:
    w, s, e, n = bbox_ll
    xs, ys = to_utm(np.array([w, e, w, e]), np.array([s, s, n, n]))
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def extent_from_plots(derived: Path, margin_km: float) -> tuple[float, float, float, float]:
    p = pd.read_parquet(derived / "plots_unified.parquet")
    xs, ys = to_utm(p["lon"].to_numpy(), p["lat"].to_numpy())
    m = margin_km * 1000
    return float(xs.min() - m), float(ys.min() - m), float(xs.max() + m), float(ys.max() + m)


def parse_years(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"),
                   help="lon/lat extent (WGS84)")
    g.add_argument("--from-plots", action="store_true", dest="from_plots",
                   help="extent = envelope of plots_unified.parquet (+ --margin-km)")
    g.add_argument("--tiles-file", default=None, dest="tiles_file",
                   help="csv with tile_id,xmin,ymin,xmax,ymax (UTM 19S) to run")
    p.add_argument("--margin-km", type=float, default=20.0, dest="margin_km")
    p.add_argument("--tile-km", type=float, default=10.0, dest="tile_km")
    p.add_argument("--resolution", type=int, default=30)
    p.add_argument("--years", default="2000-2026")
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT, dest="ckpt_dir")
    p.add_argument("--area-m2", type=float, default=900.0, dest="area_m2",
                   help="plot area held constant for every pixel. Default 900 = one Landsat "
                        "pixel, decision D3 of docs/21: the map reads as 'expected diversity "
                        "in a 900 m2 unit'. That is above the pool's p90 (500 m2), so it is "
                        "mild, declared extrapolation -- pass 500 to sit inside the pool")
    p.add_argument("--stratum", default="basal", choices=["cover", "counts", "presence", "basal"],
                   help="recording-protocol indicator held constant (Living Trees = basal)")
    p.add_argument("--mask", default="mapbiomas", choices=["mapbiomas", "none"])
    p.add_argument("--oof-csv", default=DEFAULT_OOF, dest="oof_csv",
                   help="oof_predictions.csv of the same config under kfold5_block20_unified, "
                        "for Duan smearing (default: the block20 run of the deployed config). "
                        "Pass an empty string to disable; the plain inverse under-predicts the "
                        "upper tail of TD0 by about 0.25 R2, so disabling is for diagnostics only")
    p.add_argument("--no-clip", action="store_true", dest="no_clip",
                   help="do not clip predictions to the observed training range")
    p.add_argument("--smearing-exact", action="store_true", dest="smearing_exact",
                   help="evaluate all 128 smearing draws per pixel instead of reading them "
                        "off a precomputed table. The table agrees to ~7e-6 relative and is "
                        "a third of the cost of a tile-year; this is the escape hatch")
    p.add_argument("--workers", type=int, default=4, help="dask workers (0 = no dask)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--limit", type=int, default=0, help="run only the first N tiles")
    p.add_argument("--jobs", type=int, default=1,
                   help="tile-level parallelism: N worker processes, each taking whole tiles "
                        "end to end. The CNN is 95%% of a tile-year and scales badly on "
                        "threads (16 threads buy only 2.9x), so N single-threaded processes "
                        "beat one many-threaded one by ~5.6x per core.")
    p.add_argument("--manifest", default="manifest.csv",
                   help="manifest filename inside --out/--tag; --jobs gives each worker its "
                        "own and merges them, which avoids locking a shared append")
    p.add_argument("--torch-threads", type=int, default=0, dest="torch_threads",
                   help="0 = leave torch alone; workers under --jobs are given 1")
    p.add_argument("--load-threads", type=int, default=4, dest="load_threads",
                   help="threads for the Landsat load inside this process, via dask's "
                        "threaded scheduler. A tile load is ~96%% fixed cost (1,847 s + "
                        "765 s/Mpx): ~1,800 COG header opens, not bytes. Threads help but "
                        "flatten fast, because GDAL holds the GIL while parsing them -- "
                        "measured 1,892 s / 643 s / 562 s at 1 / 4 / 8 threads. 4 is the "
                        "knee; past it, add processes, not threads. Never 1: that is the "
                        "synchronous cost, ~100 h over a full run.")
    p.add_argument("--gw-workers", type=int, default=0, dest="gw_workers",
                   help="run whole tiles on a dask-gateway cluster of this many workers. "
                        "This is the only way past the pod's 36 cores and 64 GiB, and the "
                        "worker image is byte-for-byte the pod's, so the checkpoints "
                        "unpickle identically there (measured, logs/gw_probe.log)")
    p.add_argument("--worker-cores", type=int, default=2, dest="worker_cores")
    p.add_argument("--worker-memory", type=float, default=8.0, dest="worker_memory",
                   help="GB per gateway worker; a tile load peaks at ~3 GB")
    p.add_argument("--worker-threads", type=int, default=1, dest="worker_threads",
                   help="tiles a gateway worker runs at once. Keep at 1: concurrent tiles in "
                        "one process share a GIL, and the load is GIL-bound (8 threads buy "
                        "3.4x, 8 processes buy 6.2x)")
    p.add_argument("--gw-batch", type=int, default=250, dest="gw_batch",
                   help="tiles submitted to the gateway at a time. Submitting all of them at "
                        "once means losing the scheduler cancels every pending tile: that "
                        "happened twice, costing 3,434 and then 3,320 tiles. A wave bounds "
                        "the loss to itself, and --resume picks the rest up. 0 = all at once")
    p.add_argument("--dest", default="",
                   help="where the GeoTIFFs go: a local directory or an s3:// prefix. "
                        "Defaults to --out/--tag. Gateway workers cannot see the home "
                        "directory, so a gateway run needs an s3:// prefix")
    p.add_argument("--mapbiomas-dir", default="", dest="mapbiomas_dir",
                   help="directory or s3:// prefix holding the MapBiomas annual rasters. "
                        "Only needed to work against a copy: the default is the team "
                        "prefix, read windowed straight from S3, which pod and gateway "
                        "worker reach alike")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    p.add_argument("--out", default="results/maps")
    p.add_argument("--tag", default="run")
    args = p.parse_args()

    if args.torch_threads:
        import torch
        torch.set_num_threads(args.torch_threads)

    derived = Path(args.derived)
    years = parse_years(args.years)
    tile_m = args.tile_km * 1000
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    # ---- tiles --------------------------------------------------------------------
    if args.tiles_file:
        tiles = pd.read_csv(args.tiles_file)
    else:
        ext = (extent_from_bbox(args.bbox) if args.bbox
               else extent_from_plots(derived, args.margin_km))
        tiles = mi.tile_grid(ext, tile_m, args.resolution)
    if args.limit:
        tiles = tiles.head(args.limit)
    tiles.to_csv(out / "tiles.csv", index=False)
    print(f"{len(tiles)} tiles of {args.tile_km:g} km, years {years[0]}-{years[-1]} "
          f"({len(years)}), archive span {years[0] - 2}-{years[-1]}")

    # ---- model ----------------------------------------------------------------------
    ckpts = sorted(Path(args.ckpt_dir).glob("model_seed*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.ckpt_dir}")
    ens = mi.FacetEnsemble(ckpts, device=args.device)
    targets = ens.targets
    print(f"ensemble: {len(ckpts)} seeds, targets {targets}, "
          f"context {ens.context_columns}")

    resid = y_train = None
    if args.oof_csv:
        if not Path(args.oof_csv).exists():
            raise SystemExit(f"--oof-csv not found: {args.oof_csv}. The maps must be produced "
                             "with Duan smearing (the plain inverse loses ~0.25 R2 on TD0); "
                             "fetch the csv or pass --oof-csv '' deliberately.")
        resid = mi.oof_residuals_scaled(args.oof_csv, ens.members[0].scaler, targets)
        print(f"smearing: residuals from {args.oof_csv} "
              f"({np.isfinite(resid).sum(axis=0).tolist()} usable per target)")
    else:
        print("WARNING: smearing disabled -- diagnostic run only, not for publication")
    if not args.no_clip:
        try:
            y_train = mi.training_targets(derived, targets)
            print("range clipping: training targets loaded")
        except Exception as e:                      # noqa: BLE001
            print(f"range clipping unavailable ({type(e).__name__}: {e}); predictions unclipped")
    meta = dict(tag=args.tag, years=f"{years[0]}-{years[-1]}", tile_km=args.tile_km,
                resolution=args.resolution, area_m2=args.area_m2, stratum=args.stratum,
                mask=args.mask, ckpt_dir=str(args.ckpt_dir), n_seeds=len(ckpts),
                smearing="oof" if resid is not None else "none",
                # `smearing` stays the residual source, which scripts/argo/verify_run.py
                # asserts on; how it is evaluated is a separate tag.
                smearing_eval="exact" if args.smearing_exact else f"table{tg.SMEARING_GRID}",
                clip=y_train is not None, targets=targets,
                # Declared, not corrected (docs/21 section 6): the ensemble predicts a
                # conditional mean, so the map's upper tail is compressed relative to the
                # plot observations. Carried on every tile so the caveat cannot be separated
                # from the raster.
                caveat="predicted values are conditional means and under-disperse the upper "
                       "tail; use as a relative surface")
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    if args.dry_run:
        print(json.dumps(meta, indent=1))
        print(tiles.head(10).to_string(index=False))
        return

    dest = args.dest or str(out)
    if not dest.startswith("s3://"):
        Path(dest).mkdir(parents=True, exist_ok=True)
    meta["dest"] = dest
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    if args.gw_workers and not dest.startswith("s3://"):
        raise SystemExit("a --gw-workers run must write to an s3:// --dest: the gateway "
                         "workers have no view of the home directory")
    cfg = mt.TileConfig(years=tuple(years), dest=dest, tags=meta, resolution=args.resolution,
                        area_m2=args.area_m2, stratum=args.stratum, mask=args.mask,
                        batch=args.batch, load_threads=args.load_threads,
                        load_client=args.workers > 0, torch_threads=args.torch_threads,
                        mapbiomas_dir=args.mapbiomas_dir, device=args.device,
                        smearing_exact=args.smearing_exact)

    man_path = out / args.manifest
    done: set[tuple[str, int]] = set()
    if args.resume and man_path.exists():
        done, n_bad = resume_done(man_path)
        print(f"resume: {len(done)} (tile, year) already written"
              + (f"; {n_bad} earlier failures will be retried" if n_bad else ""))

    if args.gw_workers:
        fan_out_gateway(args, tiles, cfg, ckpts, resid, y_train, man_path, done)
        return
    if args.jobs > 1:
        fan_out(args, tiles, out)
        return

    # ---- datacube + dask ----------------------------------------------------------------
    import datacube
    from datacube.utils.aws import configure_s3_access

    client = cluster = None
    if args.workers > 0:
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster(n_workers=args.workers, processes=True, threads_per_worker=1)
        client = Client(cluster)
        # On JupyterHub `dashboard_link` resolves through the service proxy, so it is the
        # only form of the URL reachable from outside the pod. Without it a local run gives
        # no way to watch the workers.
        print(f"dask: local cluster, {args.workers} workers; "
              f"dashboard {dashboard_url(cluster)}", flush=True)
    elif args.load_threads > 0:
        print(f"load: dask threaded scheduler, {args.load_threads} threads", flush=True)
    # Boot order is not negotiable, same as `scripts/35`: cluster, then
    # configure_s3_access(client=...), then the Datacube. `usgs-landsat` is requester-pays, and
    # without this every worker read comes back RasterioIOError('AccessDenied: Access Denied').
    # Passing `client` is what propagates the setting to the workers; setting it only in the
    # driver process leaves the dask path broken. See `src/biodiv/cube.py`.
    configure_s3_access(aws_unsigned=False, requester_pays=True, client=client)
    if args.mapbiomas_dir:
        os.environ[mbm.RASTERS_DIR_ENV] = args.mapbiomas_dir
    dc = datacube.Datacube(app="biodiv-map-inference")

    try:
        for ti, t in enumerate(tiles.itertuples(index=False), 1):
            tile = t._asdict()
            skip = frozenset(y for y in years if (t.tile_id, y) in done)
            if len(skip) == len(years):
                continue
            # Refresh the S3 credentials before every tile. configure_s3_access resolves
            # them once and pins the values into GDAL's config; boto3 knows how to renew
            # them from the projected service-account token, GDAL does not, because it was
            # handed literals. In a pod that is a hard wall at the role's session length:
            # a run of 2026-09-08 died on all five chunks at the 63-minute mark with
            # RasterioIOError('The provided token has expired') -- Landsat and MapBiomas
            # alike, mid-tile. Re-resolving per tile costs nothing (it sets config, it does
            # not call STS unless the cached credentials are near expiry) and makes the pod
            # lifetime independent of the credential lifetime.
            configure_s3_access(aws_unsigned=False, requester_pays=True, client=client)
            t0 = time.time()
            rows = mt.run_tile(dc, tile, cfg, ens, resid, y_train, skip)
            _append(man_path, rows)
            ok = [r for r in rows if r.get("status") == "ok"]
            if not ok:
                print(f"[{ti}/{len(tiles)}] {t.tile_id}: "
                      f"{rows[0].get('status') if rows else 'nothing to do'}", flush=True)
                continue
            print(f"[{ti}/{len(tiles)}] {t.tile_id}: load {ok[0]['load_seconds']:.0f}s, "
                  f"{len(ok)} years in {time.time() - t0:.0f}s "
                  f"(pred px/yr median {int(np.median([r['n_pred'] for r in ok]))})",
                  flush=True)
    finally:
        if client is not None:
            client.close()
    print(f"\n-> {dest}/ (<tile>_<year>.tif), manifest {man_path}")


def fan_out_gateway(args, tiles: pd.DataFrame, cfg, ckpts, resid, y_train,
                    man_path: Path, done: set) -> None:
    """Run whole tiles on dask-gateway workers, one tile per task.

    WHY THE WHOLE TILE AND NOT JUST THE LOAD. The obvious use of a cluster here is to
    parallelise the Landsat read, and that is what an earlier version did. It is the wrong
    lever, and by a wide margin: with 10 target years a tile spends ~300 s loading and
    ~1,100 s in the CNN, so a cluster that only loads attacks 21 % of the work and leaves
    its workers idle for the other 79 % -- exactly the pattern the pilot already showed
    (`docs/21` section 6). Sending the tile itself makes every worker do all of it.

    WHAT HAD TO MOVE. A gateway worker sees no home directory (measured,
    `logs/gw_probe.log`), so: the `biodiv` package is uploaded as a zip, the five
    checkpoints ride along as bytes in a broadcast payload, MapBiomas is read from the
    scratch bucket, and the GeoTIFFs are written back to S3 by the worker. What it *does*
    have is the ODC index (the gateway injects `DB_*`) and a torch of the same version as
    the pod, which is what makes running the model there legitimate rather than merely
    possible.

    Failures come back as manifest rows with ``status="error"``, never as exceptions: one
    unreadable tile must not end a run of thousands, and ``--resume`` reads the manifest.
    """
    from dask.distributed import as_completed
    from dask_gateway import Gateway
    from datacube.utils.aws import configure_s3_access

    gw = Gateway()
    opts = gw.cluster_options()
    # Never print `opts`: its repr carries AWS STS credentials and the ODC database
    # password (docs/21 section 7).
    for name, val in (("worker_cores", args.worker_cores),
                      ("worker_memory", args.worker_memory),
                      ("worker_threads", args.worker_threads)):
        if val:
            try:
                opts[name] = val
            except (KeyError, TypeError):
                print(f"  gateway has no option {name}, leaving it at the default")
    # A tile running on a worker loads with the in-process thread pool, never by submitting
    # back to the scheduler. This is not a preference: `--workers` defaults to 4, which set
    # `load_client=True`, and the first full-scale attempt had every tile die with
    # `FutureCancelledError ... scheduler-connection-lost` -- each worker task was pushing a
    # nested graph to the same scheduler that was running it. The calibration survived it
    # only because 8 workers did not strain the scheduler, and paid for it in the load:
    # 1,102 s median there against 643 s for the threaded load on the pod. Forced here so a
    # command line cannot reintroduce it.
    cfg.load_client = False
    if not cfg.load_threads:
        cfg.load_threads = 4

    # A worker container reports the node's 16 CPUs no matter what `worker_cores` was
    # asked for (measured, logs/gw_probe.log), so torch left to its own devices would open
    # 16 threads against a 2-core quota and spend the tile in context switches. It is pinned
    # to the quota here, and only here: on the pod, --torch-threads 0 still means "leave it".
    if not cfg.torch_threads:
        cfg.torch_threads = max(1, args.worker_cores)
        print(f"torch threads pinned to {cfg.torch_threads} (= --worker-cores)", flush=True)

    # Reuse an existing cluster rather than adding one. This is not tidiness: raising a second
    # cluster on this hub is what cost this very run its scheduler and 3,434 in-flight tiles
    # -- five more 8-core pods on an allocation with room for about that many. Creating
    # unconditionally made that possible; connecting makes it impossible. (Pattern taken from
    # `~/Mangles/1_Mangles_S2_upscale_clean_CSIRO_Bpanama-JH.ipynb`, which had it right.)
    existing = [c for c in gw.list_clusters() if str(c.status) in ("2", "ClusterStatus.RUNNING")]
    if existing:
        cluster = gw.connect(existing[0].name)
        ours = False
        print(f"reusing running cluster {cluster.name}; not raising a second one", flush=True)
    else:
        cluster = gw.new_cluster(opts)
        ours = True
    try:
        cluster.scale(args.gw_workers)
        client = cluster.get_client()
        print(f"gateway {cluster.name}: {args.gw_workers} workers x {args.worker_cores} "
              f"cores / {args.worker_memory:g} GB, {args.worker_threads} tile(s) each; "
              f"dashboard {dashboard_url(cluster)}", flush=True)
        # Wait for most of the cluster, not for one worker: starting on the first arrival
        # means the early tiles fight over a fraction of the machine, and the request may
        # not be granted in full at all -- a shared hub, and 128 x 2 cores is a large ask.
        # Not all of it either, because one pod stuck pending would hold up the whole run.
        # (Tiles submitted before a worker joins are safe: `upload_file` registers a
        # scheduler-side plugin, so a late worker still gets the package.)
        # Polled by hand against the live worker list rather than with
        # `client.wait_for_workers`, which on a gateway cluster is satisfied by the count that
        # was *requested* and not by the workers actually up: asked for 102 against a cluster
        # scaled to 128 but holding 5, it returned in 1.9 s, and the run started on 5 workers.
        # (It does block when the request exceeds the cluster's own target -- 25 against a
        # cluster scaled to 5 raises WorkerStartTimeoutError -- so the trap is specifically a
        # scale request the hub has not filled.) Waiting matters because the hub grants pods
        # gradually and may not grant them all, so the run should start on the cluster it is
        # actually going to have, and say what that is.
        want = max(1, int(0.8 * args.gw_workers))
        deadline = time.time() + 1800
        got = 0
        while time.time() < deadline:
            got = len(client.scheduler_info()["workers"])
            if got >= want:
                break
            time.sleep(15)
        if got < want:
            print(f"  only {got} of {args.gw_workers} workers after "
                  f"{(1800) / 60:.0f} min; starting with those", flush=True)
        print(f"  {got} workers up ({got * args.worker_cores} cores granted of "
              f"{args.gw_workers * args.worker_cores} asked)", flush=True)
        configure_s3_access(aws_unsigned=False, requester_pays=True, client=client)

        zp = package_biodiv()
        client.upload_file(str(zp))
        print(f"uploaded {zp.name} ({zp.stat().st_size / 1e3:.0f} kB) to the workers",
              flush=True)

        payload = mt.Payload(ckpts=[(Path(c).name, Path(c).read_bytes()) for c in ckpts],
                             resid=resid, y_train=y_train)

        todo = [t._asdict() for t in tiles.itertuples(index=False)
                if any((t.tile_id, y) not in done for y in cfg.years)]
        batch = args.gw_batch or len(todo)
        print(f"{len(todo)} tiles to run, in waves of {batch}", flush=True)

        t0 = time.time()
        n_ok = n_err = i = 0
        for start in range(0, len(todo), batch):
            wave = todo[start:start + batch]
            # Re-broadcast per wave, not once for the whole run. Scattered once, the payload
            # is a single dependency under ~3,300 tasks: when the worker holding it goes, every
            # remaining tile fails instantly with `lost dependencies` -- which is exactly what
            # happened, 3,255 tiles cancelled in 28 minutes. It is 800 kB; a broadcast per 250
            # tiles costs nothing and confines that loss to one wave.
            pay = client.scatter(payload, broadcast=True)
            n_ok_before = n_ok
            futures = {client.submit(mt.run_tile_remote, tl, pay, cfg,
                                     tuple(y for y in cfg.years if (tl["tile_id"], y) in done),
                                     key=f"tile-{tl['tile_id']}", pure=False): tl["tile_id"]
                       for tl in wave}
            try:
                for fut in as_completed(list(futures)):
                    tile_id = futures[fut]
                    try:
                        rows = fut.result()
                    except Exception as e:                           # noqa: BLE001
                        rows = [dict(tile_id=tile_id, year=y, status="error",
                                     error=f"{type(e).__name__}: {str(e)[:200]}")
                                for y in cfg.years]
                    _append(man_path, rows)
                    bad = [r for r in rows if r.get("status") != "ok"]
                    n_err += bool(bad)
                    n_ok += not bad
                    i += 1
                    el = time.time() - t0
                    print(f"[{i}/{len(todo)}] {tile_id}: "
                          f"{'ok' if not bad else bad[0].get('error', bad[0]['status'])}"
                          f"  |  {n_ok} ok / {n_err} failed, {el / 60:.0f} min elapsed, "
                          f"eta {el / i * (len(todo) - i) / 3600:.1f} h", flush=True)
            except Exception as e:                                   # noqa: BLE001
                # The cluster went away mid-wave -- twice now, once with the hub rejecting the
                # JupyterHub API token. Everything already written is in the manifest, so stop
                # here and let --resume continue rather than grinding through thousands of
                # cancellations, which is what the unbatched version did.
                print(f"wave failed after {i} tiles ({type(e).__name__}: {str(e)[:160]}); "
                      f"stopping so --resume can continue", flush=True)
                raise

            # A wave where nothing at all succeeded is a broken cluster, not 250 unlucky tiles,
            # and it does not raise: the failures arrive as task *results*, one per future, so
            # the loop above consumes them happily. Left alone the run grinds through every
            # remaining wave in minutes, writes thousands of error rows and exits 0 -- and the
            # supervisor, seeing tiles still to do, starts it again. That thrash produced
            # 176,000 junk manifest rows before it was caught. Stop and let the supervisor
            # rebuild the cluster instead.
            if n_ok == n_ok_before and len(wave) > 1:
                raise SystemExit(
                    f"wave of {len(wave)} tiles produced no output at all "
                    f"(last error: {bad[0].get('error') if bad else 'unknown'}); "
                    f"stopping rather than burning through the remaining waves")
    finally:
        # Only tear down a cluster this run raised. A reused one may belong to another run --
        # the whole point of connecting instead of creating -- and shutting it down would do
        # to that run exactly what a stray second cluster did to this one.
        if ours:
            cluster.shutdown()
            print("gateway cluster shut down")
        else:
            print(f"leaving reused cluster {cluster.name} running: this run did not raise it")
    print(f"\n-> {cfg.dest}/ (<tile>_<year>.tif), manifest {man_path}")


def package_biodiv() -> Path:
    """Zip `src/biodiv` for `Client.upload_file`, which puts it on the workers' sys.path.

    Built fresh on every run rather than kept as an artefact: the whole point of shipping
    the package is that the workers run the same code as the gate, and a stale zip is
    exactly the way that stops being true.
    """
    import zipfile
    src = ROOT / "src" / "biodiv"
    zp = Path(tempfile.gettempdir()) / "biodiv_worker.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src.rglob("*.py")):
            z.write(f, str(Path("biodiv") / f.relative_to(src)))
        # `mapinfer.load_terrain` reads `terrain()` out of scripts/03 by path, to keep that
        # script the single source of the topographic derivatives. A worker has no repo to
        # read it from, and a by-path load cannot see inside a zip, so the same file rides
        # along as an importable module and `_terrain_fn` falls back to it. Copied from the
        # live script on every run: the two cannot drift within a run.
        z.write(ROOT / "scripts" / "03_extract_topography.py", "biodiv/_terrain_src.py")
    return zp


def dashboard_url(cluster) -> str:
    """Absolute, clickable dashboard URL.

    ``cluster.dashboard_link`` honours ``distributed.dashboard.link``, which EASI sets to
    ``{JUPYTERHUB_SERVICE_PREFIX}proxy/{port}/status`` -- a *relative* path, because the pod
    has no idea what its external hostname is (``JUPYTERHUB_PUBLIC_URL`` and
    ``JUPYTERHUB_HOST`` are both empty here). A relative path is useless to anyone reading
    the log outside the browser session, so take the scheme+host from the one place that
    does know it, ``gateway.public-address`` in /etc/dask/dask.yaml, and prepend it.

    Falls back to whatever dask gave us if that key is absent on some other deployment.
    """
    link = str(getattr(cluster, "dashboard_link", "") or "")
    if not link or link.startswith("http"):
        return link
    try:
        import dask.config
        from urllib.parse import urlsplit
        pub = dask.config.get("gateway.public-address", "") or ""
        u = urlsplit(pub)
        if u.scheme and u.netloc:
            return f"{u.scheme}://{u.netloc}{link}"
    except Exception:                                # noqa: BLE001
        pass
    return link


def fan_out(args, tiles: pd.DataFrame, out: Path) -> None:
    """Run whole tiles across ``args.jobs`` worker processes, then merge their manifests.

    Each worker is this same script re-invoked with ``--jobs 1``, its own slice of the tile
    grid and its own manifest file. Three reasons for separate processes rather than threads
    or a shared dask cluster:

    * the CNN is ~95 % of a tile-year and scales badly on threads (16 torch threads buy 2.9x
      on 16 cores, 18 % efficiency), so one single-threaded process per core is ~5.6x more
      core-efficient than one many-threaded process;
    * each worker loads with dask's *threaded* scheduler (``--load-threads``), not
      synchronously. The first design here used synchronous reads on the reasoning that N
      tile-processes already supply the S3 concurrency. That was wrong and measured so: a
      tile load is 96 % fixed cost (1,847 s + 765 s per Mpx at 30 m) -- ~1,800 COG header
      opens rather than bytes -- and one tile loads in 1,892 s synchronously against 643 s
      on 4 threads, a penalty that would have added ~100 h to a full run. Threads do not go
      much further than that (562 s at 8) because GDAL parses those headers holding the GIL;
      the same load on 8 *processes* takes 304 s, which is why the parallelism that pays is
      one process per tile and only a handful of threads inside it;
    * a per-worker manifest merged at the end is safe by construction, where concurrent
      appends to one csv would need locking.

    ``--resume`` is honoured: each worker skips the (tile, year) pairs already in the merged
    manifest, which is read in before the split.
    """
    import subprocess

    parts_dir = out / "_parts"
    parts_dir.mkdir(exist_ok=True)
    merged = out / args.manifest

    # Round-robin rather than contiguous blocks: tiles differ in native cover and therefore
    # in cost, and interleaving keeps the workers from finishing at wildly different times.
    assignments = [tiles.iloc[i::args.jobs] for i in range(args.jobs)]
    assignments = [a for a in assignments if len(a)]
    print(f"--jobs {args.jobs}: {len(tiles)} tiles -> "
          f"{[len(a) for a in assignments]} per worker", flush=True)

    procs, part_files = [], []
    for i, part in enumerate(assignments):
        tf = parts_dir / f"tiles_{i}.csv"
        part.to_csv(tf, index=False)
        man_i = f"_parts/manifest_{i}.csv"
        part_files.append(out / man_i)
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--tiles-file", str(tf), "--out", str(args.out), "--tag", args.tag,
               "--years", args.years, "--resolution", str(args.resolution),
               "--tile-km", str(args.tile_km), "--derived", str(args.derived),
               "--ckpt-dir", str(args.ckpt_dir), "--area-m2", str(args.area_m2),
               "--stratum", args.stratum, "--mask", args.mask, "--device", args.device,
               "--batch", str(args.batch), "--manifest", man_i,
               "--jobs", "1", "--workers", "0", "--torch-threads", "1",
               "--load-threads", str(args.load_threads or 1)]
        if args.dest:
            cmd += ["--dest", args.dest]
        if args.mapbiomas_dir:
            cmd += ["--mapbiomas-dir", args.mapbiomas_dir]
        if args.oof_csv:
            cmd += ["--oof-csv", args.oof_csv]
        if args.no_clip:
            cmd.append("--no-clip")
        if args.smearing_exact:
            cmd.append("--smearing-exact")
        if args.resume:
            cmd.append("--resume")
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1")
        log = (parts_dir / f"worker_{i}.log").open("w")
        procs.append((i, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env),
                      log))

    t0 = time.time()
    failed = []
    for i, p, log in procs:
        rc = p.wait()
        log.close()
        if rc != 0:
            failed.append((i, rc))
        print(f"  worker {i} exited {rc} after {time.time() - t0:.0f}s", flush=True)

    frames = [pd.read_csv(f) for f in part_files if f.exists()]
    if frames:
        man = pd.concat(frames, ignore_index=True).sort_values(["tile_id", "year"])
        man.to_csv(merged, index=False)
        print(f"merged {len(frames)} worker manifests -> {merged} ({len(man)} rows)")
    if failed:
        raise SystemExit(f"workers failed: {failed}; logs in {parts_dir}")


#: Every column any row can carry, in order. The manifest is appended to incrementally and
#: rows are not all the same shape -- an "ok" row has the counts and timings, a "no_data" or
#: "error" row has almost none -- so the columns are fixed here and every row is reindexed
#: onto them. Without that, a run whose first finished tile happened to fail would write a
#: four-column header and then silently misalign every later row under it.
MANIFEST_COLUMNS = ["tile_id", "year", "status", "n_px", "n_native", "n_pred",
                    "n_dates_window", "grid_first", "grid_last", "seconds", "load_seconds",
                    "file", "error"]


def resume_done(man_path: Path) -> tuple[set[tuple[str, int]], int]:
    """``(tile, year)`` pairs that produced a raster, and how many rows did not.

    Only ``status == "ok"`` counts. Taking every row would make a failure permanent, and the
    failure mode is not hypothetical: the first full-scale attempt wrote 594 ``error`` rows
    in a few minutes, and a resume that honoured them would have skipped those tiles for the
    rest of the run -- leaving holes in the mosaic that nothing downstream would flag.
    """
    m = pd.read_csv(man_path)
    ok = m[m["status"] == "ok"]
    return set(zip(ok["tile_id"], ok["year"].astype(int))), len(m) - len(ok)


def _append(path: Path, rows: list[dict]) -> None:
    """Append rows, widening a manifest that an older run left with a narrower header.

    Reindexing the new rows is not enough on its own: a manifest already on disk under a
    four-column header would get thirteen-column rows written beneath it, which is the same
    misalignment moved one step later -- and `--resume` reads exactly that file.
    """
    if not rows:
        return
    if path.exists():
        head = path.open().readline().strip().split(",")
        if head != MANIFEST_COLUMNS:
            old = pd.read_csv(path).reindex(columns=MANIFEST_COLUMNS)
            old.to_csv(path, index=False)
    df = pd.DataFrame(rows).reindex(columns=MANIFEST_COLUMNS)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


if __name__ == "__main__":
    main()
