"""The map-inference inputs, resolved the same way the Argo workflow resolves them.

`scripts/argo/process_argo.yaml` stages four things into every pod before `scripts/73` runs:
the five model checkpoints, the out-of-fold predictions the Duan retransformation needs, the
derived target tables, and the tile list. It does it with an initContainer that runs
`aws s3 sync` from one prefix, and the MapBiomas rasters come from a second prefix through
`BIODIV_MAPBIOMAS_DIR`. Everything downstream then reads ordinary local paths.

The map notebooks used to reach the same four things through paths that existed only on the
author's machines -- an external volume, a `/mnt/rapidita_4T/...` checkout -- so handing the
code to a collaborator was not enough to let them run it, even with permission to read the
data. This module closes that gap by giving the notebooks the pod's own resolution: the same
prefix, the same layout, the same files. A reader with credentials for the bucket runs the
notebook; a reader without them gets a message naming what to ask for.

WHAT LIVES WHERE. The layout is not invented here, it is the one `process_argo.yaml` and
`scripts/argo/upload_assets.sh` already agree on:

    <assets>/models/final/model_seed{0..4}.pt   the deployed ensemble
    <assets>/models/oof_predictions.csv         block-CV residuals, for Duan smearing
    <assets>/derived/                           plots_unified + the padded target tables
    <assets>/tiles/tiles_native_10km.csv        the 5,892 tiles carrying native vegetation
    <mapbiomas>/<year>_coverage_lclu_*.tif      26 annual rasters, read windowed

`BIODIV_ASSETS` and `BIODIV_MAPS_DEST` default to the team prefix the workflow uses, so a
notebook needs no configuration beyond credentials; point them elsewhere to work against a
copy. Both accept a local directory as well, which is what makes the notebook runnable on a
machine that already has the files -- and what the pod itself effectively does, since by the
time `scripts/73` starts, its assets are local.

WHY IT HANDS BACK A LOCAL PATH. The callers are `torch.load`, `pd.read_parquet` and
`rasterio.open`, and each wants something different from a remote URI: pandas needs `s3fs`,
rasterio wants `/vsis3/`, torch wants a file object. Fetching once into a cache and handing
everyone a real file keeps all three working with only `boto3`, and keeps the notebook path
the same shape as the pod path. The exception is `maps_dest()`, which stays a URI: it is a
destination, and `scripts/73 --dest` already writes to S3 by itself.

A cached file is reused only when its size matches the object's. Size is not a checksum, but
it catches the failure that happens in practice -- an interrupted download leaving a short
file -- for one HEAD request. `BIODIV_ASSETS_CACHE` moves the cache; deleting it is safe.

Usage:
    from biodiv import assets
    print(assets.describe())
    ens   = mi.FacetEnsemble(sorted(assets.ckpt_dir().glob("model_seed*.pt")))
    resid = mi.oof_residuals_scaled(assets.oof_csv(), ens.members[0].scaler, ens.targets)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

#: Staged inputs: an ``s3://`` prefix or a local directory with the layout above.
ASSETS_ENV = "BIODIV_ASSETS"

#: Where the GeoTIFFs are written. A URI, never downloaded.
MAPS_ENV = "BIODIV_MAPS_DEST"

#: Where fetched objects are kept. Defaults to ``~/.cache/biodiv-assets``.
CACHE_ENV = "BIODIV_ASSETS_CACHE"

_TEAM = "s3://easido-prod-dc-data-projects/easi-workflows-team/biodiv"

#: Defaults are the workflow's own values (`scripts/argo/process_argo.yaml`), so that a
#: notebook run by someone with read access needs no configuration at all.
DEFAULT_ASSETS = f"{_TEAM}/assets"
DEFAULT_MAPS = f"{_TEAM}/maps/chile_30m_2000_2026"

#: The tile list `scripts/75` produced: of 17,019 10 km tiles, the 5,892 that carry native
#: vegetation in at least one prediction year.
TILES_CSV = "tiles_native_10km.csv"


def assets_root() -> str:
    return os.environ.get(ASSETS_ENV) or DEFAULT_ASSETS


def maps_dest() -> str:
    """Destination prefix for the rasters. Returned as given -- not fetched."""
    return os.environ.get(MAPS_ENV) or DEFAULT_MAPS


def is_remote(where: str | None = None) -> bool:
    return (where or assets_root()).startswith("s3://")


def cache_dir() -> Path:
    d = os.environ.get(CACHE_ENV)
    return Path(d) if d else Path.home() / ".cache" / "biodiv-assets"


# --------------------------------------------------------------------------------------
# the four staged inputs
# --------------------------------------------------------------------------------------

def ckpt_dir() -> Path:
    """Directory holding ``model_seed*.pt`` for the deployed ensemble."""
    return _directory("models/final", pattern="model_seed")


def oof_csv() -> Path:
    """Out-of-fold predictions of the *same* configuration as the checkpoints.

    Same configuration is not a detail: the smearing factor is built from these residuals,
    and residuals of another model give the wrong retransformation on every pixel.
    """
    return _file("models/oof_predictions.csv")


def derived() -> Path:
    """Directory of derived tables (`--derived` for `scripts/73`)."""
    return _directory("derived")


def tiles_csv(name: str = TILES_CSV) -> Path:
    return _file(f"tiles/{name}")


# --------------------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------------------

def _split(uri: str) -> tuple[str, str]:
    bucket, _, prefix = uri[len("s3://"):].partition("/")
    return bucket, prefix.strip("/")


def _client():
    try:
        import boto3
    except ModuleNotFoundError:                                     # pragma: no cover
        raise ModuleNotFoundError(
            "boto3 is needed to read the assets from S3. Install it, or point "
            f"{ASSETS_ENV} at a local directory holding the same layout.")
    return boto3.client("s3")


def _unreadable(what: str, exc: Exception) -> FileNotFoundError:
    return FileNotFoundError(
        f"{what} is not readable ({type(exc).__name__}). Either the credentials of this "
        f"session have no access to it, or {ASSETS_ENV} points somewhere else. The prefix "
        f"in use is {assets_root()}; `aws sts get-caller-identity` says who you are.")


def _file(rel: str) -> Path:
    root = assets_root()
    if not is_remote(root):
        p = Path(root) / rel
        if not p.exists():
            raise FileNotFoundError(f"{p} not found under {ASSETS_ENV}={root}")
        return p

    bucket, prefix = _split(root)
    key = f"{prefix}/{rel}" if prefix else rel
    dest = cache_dir() / bucket / key
    s3 = _client()
    try:
        size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except Exception as e:                                          # noqa: BLE001
        raise _unreadable(f"s3://{bucket}/{key}", e)
    if dest.exists() and dest.stat().st_size == size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    s3.download_file(bucket, key, str(tmp))
    tmp.replace(dest)                              # atomic: a half file is never returned
    return dest


def _directory(rel: str, *, pattern: str = "") -> Path:
    root = assets_root()
    if not is_remote(root):
        d = Path(root) / rel
        if not d.is_dir():
            raise FileNotFoundError(f"{d} not found under {ASSETS_ENV}={root}")
        return d

    bucket, prefix = _split(root)
    key_prefix = f"{prefix}/{rel}" if prefix else rel
    dest = cache_dir() / bucket / key_prefix
    s3 = _client()
    n = 0
    try:
        pages = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket,
                                                             Prefix=key_prefix + "/")
        for page in pages:
            for obj in page.get("Contents", []):
                name = obj["Key"][len(key_prefix) + 1:]
                if not name or name.endswith("/") or (pattern and pattern not in name):
                    continue
                f = dest / name
                n += 1
                if f.exists() and f.stat().st_size == obj["Size"]:
                    continue
                f.parent.mkdir(parents=True, exist_ok=True)
                tmp = f.with_name(f.name + ".part")
                s3.download_file(bucket, obj["Key"], str(tmp))
                tmp.replace(f)
    except Exception as e:                                          # noqa: BLE001
        raise _unreadable(f"s3://{bucket}/{key_prefix}/", e)
    if not n:
        raise FileNotFoundError(
            f"nothing under s3://{bucket}/{key_prefix}/"
            + (f" matching {pattern!r}" if pattern else "")
            + ". The prefix exists but is empty -- run scripts/argo/upload_assets.sh.")
    return dest


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------

def describe() -> str:
    """One line for a notebook to print, so a reader knows where the inputs came from."""
    # `mapbiomas` is the authority on where the rasters are, but it imports rasterio at
    # module level and this line has to survive on a machine that has not installed the
    # geo stack yet -- which is exactly the machine a new collaborator runs first.
    try:
        from . import mapbiomas as mb
        mb_dir = mb.rasters_dir()
    except ModuleNotFoundError:
        mb_dir = os.environ.get("BIODIV_MAPBIOMAS_DIR") or "(unset: the team prefix)"
    lines = [f"assets   : {assets_root()}"
             + ("" if os.environ.get(ASSETS_ENV) else "   (default: the workflow prefix)"),
             f"maps     : {maps_dest()}",
             f"mapbiomas: {mb_dir}"]
    if is_remote():
        lines.append(f"cache    : {cache_dir()}")
    return "\n".join(lines)


def clear_cache() -> None:
    d = cache_dir()
    if d.exists():
        shutil.rmtree(d)
