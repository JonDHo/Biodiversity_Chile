#!/usr/bin/env python3
"""Copy a finished pod's run evidence to S3 next to the rasters it produced.

``scripts/73_map_inference.py`` writes ``manifest.csv`` and ``run.json`` under
``--out/--tag``, on the pod's own filesystem. Those two files are the only
per-(tile, year) record of *why* something happened -- status, n_pred,
seconds, the error string -- and the only record of which checkpoints and
settings produced the rasters. The GeoTIFFs go to S3; these do not, so
without this copy they die with the pod and a finished run can only be
audited from the tags baked into each raster.

Usage:
    python upload_run_evidence.py /work/out s3://bucket/prefix/work/runs/<workflow>/infer/chunk-0
"""
from __future__ import annotations

import sys
from pathlib import Path

import boto3


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    root, dest = Path(sys.argv[1]), sys.argv[2]
    if not dest.startswith("s3://"):
        raise SystemExit(f"destination must be an s3:// prefix, got {dest}")
    bucket, _, prefix = dest[len("s3://"):].partition("/")
    prefix = prefix.rstrip("/")

    s3 = boto3.client("s3")
    n = 0
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        key = f"{prefix}/{f.relative_to(root)}"
        s3.upload_file(str(f), bucket, key)
        print(f"evidence -> s3://{bucket}/{key}", flush=True)
        n += 1
    if not n:
        print(f"WARNING: nothing under {root} to upload", file=sys.stderr)


if __name__ == "__main__":
    main()
