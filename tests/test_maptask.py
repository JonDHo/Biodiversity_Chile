"""Gates for the pieces that let a tile run somewhere other than the pod.

Runs offline: no datacube, no gateway, no S3. What is checked here is the machinery that
was added so a dask-gateway worker -- which sees no home directory -- can do a whole tile:
the raster directory being redirectable, the checkpoints loading from memory, the GeoTIFF
writer, and the manifest staying aligned when rows of different shapes are appended.

The tile loop itself (`maptask.run_tile`) is not tested here: it needs the datacube. Its
gate is `scripts/74_check_map_consistency.py` plus the recorded fact that the refactor
reproduced `results/maps/pilot_cauquenes/t18_600_2020.tif` to 1.6e-5 on the facets and
bit-for-bit on the quality bands (docs/21 section 8).
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import mapbiomas as mb                 # noqa: E402
from biodiv import maptask as mt                   # noqa: E402


def _template(n=8, x0=180_000.0, y0=6_000_000.0, res=30.0):
    """A one-time slice with the coordinates `grid_transform` reads, and nothing else."""
    x = x0 + res * np.arange(n)
    y = y0 - res * np.arange(n)
    return xr.DataArray(np.zeros((n, n), np.float32), dims=("y", "x"),
                        coords={"y": y, "x": x})


# --------------------------------------------------------------------------------------
# where the rasters live
# --------------------------------------------------------------------------------------

def test_rasters_dir_follows_the_environment(tmp_path, monkeypatch):
    """The override is the whole reason a worker can build the mask at all."""
    for name in ("2001_coverage_lclu_20-1-1_aaa.tif", "1999_coverage_lclu_20-1-1_bbb.tif",
                 "not_a_year.tif", "2003_coverage.txt"):
        (tmp_path / name).touch()
    monkeypatch.setenv(mb.RASTERS_DIR_ENV, str(tmp_path))
    assert mb.rasters_dir() == str(tmp_path)
    assert sorted(mb.available_years()) == [1999, 2001]
    assert mb.year_map(2001)[0].endswith("2001_coverage_lclu_20-1-1_aaa.tif")


def test_rasters_dir_defaults_to_s3_when_the_repo_has_no_rasters(tmp_path, monkeypatch):
    """The repo ships `MapBiomas/legend.csv` and no rasters, so the directory existing is
    not evidence the maps are there. Falling back to the team prefix is what lets `run_tile`
    build a mask with nothing configured."""
    monkeypatch.delenv(mb.RASTERS_DIR_ENV, raising=False)
    monkeypatch.setattr(mb, "MB_DIR", tmp_path)
    (tmp_path / "legend.csv").touch()
    assert mb.rasters_dir() == mb.DEFAULT_RASTERS_DIR
    assert mb.DEFAULT_RASTERS_DIR.startswith("s3://")

    (tmp_path / "2010_coverage_x.tif").touch()      # a real local copy still wins
    assert mb.rasters_dir() == str(tmp_path)


def test_rasters_dir_scan_is_cached_per_directory(tmp_path, monkeypatch):
    """Two directories must not serve each other's listing: the cache is keyed, not global."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "2010_coverage_x.tif").touch()
    (tmp_path / "b" / "2011_coverage_x.tif").touch()
    monkeypatch.setenv(mb.RASTERS_DIR_ENV, str(tmp_path / "a"))
    assert sorted(mb.available_years()) == [2010]
    monkeypatch.setenv(mb.RASTERS_DIR_ENV, str(tmp_path / "b"))
    assert sorted(mb.available_years()) == [2011]


def test_no_rasters_names_the_directory_it_looked_in(tmp_path, monkeypatch):
    monkeypatch.setenv(mb.RASTERS_DIR_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        mb.year_map(2010)


# --------------------------------------------------------------------------------------
# writing a tile-year
# --------------------------------------------------------------------------------------

def test_emit_writes_bands_names_and_grid(tmp_path):
    import rasterio
    tpl = _template()
    layers = {"td_inext_q0": np.full((8, 8), 3.5, np.float32),
              "native": np.ones((8, 8), np.float32)}
    where = mt._emit(str(tmp_path), "t1_2.tif", tpl, layers, {"year": 2020, "caveat": "x"})
    assert where == str(tmp_path / "t1_2.tif")
    with rasterio.open(where) as src:
        assert src.count == 2
        assert src.descriptions == ("td_inext_q0", "native")
        assert src.tags()["year"] == "2020"
        np.testing.assert_allclose(src.read(1), 3.5)
        # north-up, half-pixel offset from the cell centres the template carries
        assert src.transform.a == 30.0 and src.transform.e == -30.0
        assert src.transform.c == pytest.approx(180_000.0 - 15.0)


def test_grid_transform_puts_the_first_cell_centre_where_the_template_has_it():
    tpl = _template()
    tr = mt.grid_transform(tpl)
    cx, cy = tr * (0.5, 0.5)
    assert cx == pytest.approx(float(tpl.x.values[0]))
    assert cy == pytest.approx(float(tpl.y.values[0]))


# --------------------------------------------------------------------------------------
# shipping the model
# --------------------------------------------------------------------------------------

CKPT = (ROOT / "results/models_unified_topofix/"
        "C1D01_curve1d_kndvi_raw100_pg-all_unified_ctr_FINAL_alldata/final")


@pytest.mark.skipif(not CKPT.is_dir(), reason="deployed checkpoints not present")
def test_ensemble_from_memory_matches_ensemble_from_disk():
    """Workers get the checkpoints as bytes; the weights must not change on the way."""
    from biodiv import mapinfer as mi
    paths = sorted(CKPT.glob("model_seed*.pt"))
    assert paths, "no seed checkpoints"
    disk = mi.FacetEnsemble(paths)
    mem = mi.FacetEnsemble([io.BytesIO(p.read_bytes()) for p in paths])
    assert mem.targets == disk.targets
    assert mem.context_columns == disk.context_columns
    for a, b in zip(disk.members, mem.members):
        for pa, pb in zip(a.model.parameters(), b.model.parameters()):
            np.testing.assert_array_equal(pa.detach().numpy(), pb.detach().numpy())


# --------------------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------------------

def _script():
    spec = importlib.util.spec_from_file_location(
        "s73", ROOT / "scripts" / "73_map_inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manifest_columns_survive_a_failure_first(tmp_path):
    """A run whose first finished tile failed used to write a 4-column header and then
    misalign every later row under it. Order of arrival is not ours to choose."""
    import pandas as pd
    s73 = _script()
    man = tmp_path / "manifest.csv"
    s73._append(man, [dict(tile_id="tA", year=2020, status="error", error="boom")])
    s73._append(man, [dict(tile_id="tB", year=2020, status="ok", n_px=9, n_pred=4,
                           file="s3://b/k.tif")])
    d = pd.read_csv(man)
    assert list(d.columns) == s73.MANIFEST_COLUMNS
    assert d.loc[d.tile_id == "tB", "file"].item() == "s3://b/k.tif"
    assert d.loc[d.tile_id == "tB", "n_px"].item() == 9
    assert d.loc[d.tile_id == "tA", "error"].item() == "boom"
    assert d.loc[d.tile_id == "tA", "file"].isna().all()


def test_manifest_widens_a_narrow_header_left_by_an_older_run(tmp_path):
    """The narrow header is not hypothetical: it is what the first gateway run wrote, and
    --resume reads the manifest, so it has to be repaired rather than appended under."""
    import pandas as pd
    s73 = _script()
    man = tmp_path / "manifest.csv"
    man.write_text("tile_id,year,status,error\ntA,2020,error,boom\n")
    s73._append(man, [dict(tile_id="tB", year=2020, status="ok", n_px=9, file="s3://b/k.tif")])
    d = pd.read_csv(man)
    assert list(d.columns) == s73.MANIFEST_COLUMNS
    assert len(d) == 2
    assert d.loc[d.tile_id == "tA", "error"].item() == "boom"
    assert d.loc[d.tile_id == "tB", "file"].item() == "s3://b/k.tif"


def test_gateway_run_refuses_a_local_destination(tmp_path, monkeypatch):
    """The workers cannot see the home directory, so writing there would fail per tile,
    hours in. It has to fail at argument parsing instead."""
    s73 = _script()
    tiles = tmp_path / "tiles.csv"
    tiles.write_text("tile_id,xmin,ymin,xmax,ymax\nt1_1,0,0,9990,9990\n")
    monkeypatch.setattr(sys, "argv", [
        "73", "--tiles-file", str(tiles), "--years", "2020-2020", "--gw-workers", "2",
        "--out", str(tmp_path / "out"), "--tag", "t", "--oof-csv", "",
        "--ckpt-dir", str(CKPT), "--dest", str(tmp_path / "local")])
    with pytest.raises(SystemExit, match="s3://"):
        s73.main()


# --------------------------------------------------------------------------------------
# shipping the code
# --------------------------------------------------------------------------------------

def test_worker_zip_carries_the_terrain_source_and_reproduces_it(tmp_path):
    """A worker imports `biodiv` from a zip and has no repo behind it.

    `mapinfer.load_terrain` reads `terrain()` out of `scripts/03` by path so that script
    stays the single source of the derivatives -- which is exactly what cannot work inside a
    zip, and what the first gateway run failed on: every tile came back
    `FileNotFoundError: .../scripts/03_extract_topography.py`. The script now rides along in
    the archive. This runs in a subprocess with only the zip on `sys.path`, because the
    point is what happens when the repo is *not* importable.
    """
    import subprocess
    s73 = _script()
    zp = s73.package_biodiv()
    probe = tmp_path / "probe.py"
    probe.write_text(f"""
import sys
sys.path.insert(0, {str(zp)!r})
import numpy as np
from biodiv import mapinfer as mi
assert "biodiv_worker.zip" in mi.__file__, mi.__file__
rng = np.random.default_rng(0)
elev = 500 + 50 * rng.standard_normal((40, 40))
got = mi._terrain_fn()(elev, 30.0, -36.0)
print(",".join(sorted(got)))
np.save({str(tmp_path / "got.npy")!r}, np.stack([got[k] for k in sorted(got)]))
""")
    r = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, cwd="/")
    assert r.returncode == 0, r.stderr[-2000:]
    keys = r.stdout.strip().splitlines()[-1].split(",")

    spec = importlib.util.spec_from_file_location(
        "repo03", ROOT / "scripts" / "03_extract_topography.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(0)
    elev = 500 + 50 * rng.standard_normal((40, 40))
    want = mod.terrain(elev, 30.0, -36.0)
    assert keys == sorted(want)
    got = np.load(tmp_path / "got.npy")
    for i, k in enumerate(keys):
        np.testing.assert_array_equal(got[i], np.asarray(want[k]), err_msg=k)


def test_resume_retries_failures_and_skips_only_successes(tmp_path):
    """A failed tile must come back on the next run, and a written one must not."""
    s73 = _script()
    man = tmp_path / "manifest.csv"
    s73._append(man, [
        dict(tile_id="tA", year=2020, status="ok", file="s3://b/tA_2020.tif"),
        dict(tile_id="tA", year=2021, status="error", error="scheduler-connection-lost"),
        dict(tile_id="tB", year=2020, status="no_data"),
    ])
    done, n_bad = s73.resume_done(man)
    assert done == {("tA", 2020)}
    assert n_bad == 2


# --------------------------------------------------------------------------------------
# the development tile cache
# --------------------------------------------------------------------------------------

def _fake_tile_da(T=11, ny=7, nx=5):
    import xarray as xr
    times = np.arange("2018-01-01", "2018-01-12", dtype="datetime64[D]").astype("datetime64[ns]")
    vals = np.random.default_rng(0).normal(0.3, 0.05, (T, ny, nx)).astype(np.float32)
    vals[3, 2, 2] = np.nan
    return xr.DataArray(vals, coords={"time": times[:T], "y": np.arange(ny) * 30.0,
                                      "x": np.arange(nx) * 30.0},
                        dims=("time", "y", "x"), name="kndvi")


def _cache_cfg(years=(2020, 2021), resolution=30):
    return mt.TileConfig(years=years, dest="", tags={}, resolution=resolution)


def test_tile_cache_round_trips_exactly_and_memory_maps(tmp_path):
    tile = dict(tile_id="tTEST", xmin=0.0, ymin=0.0, xmax=9990.0, ymax=9990.0)
    cfg = _cache_cfg()
    da = _fake_tile_da()
    assert mt._cache_read(str(tmp_path), tile, cfg) is None       # cold
    mt._cache_write(str(tmp_path), tile, cfg, da)
    got = mt._cache_read(str(tmp_path), tile, cfg)
    assert got is not None
    assert np.array_equal(da.values, got.values, equal_nan=True)
    assert np.array_equal(da.time.values, got.time.values)
    assert np.array_equal(da.x.values, got.x.values)
    assert np.array_equal(da.y.values, got.y.values)
    # memory-mapped, and not a dask array: `load_tile` must not try to compute it
    assert isinstance(got.data, np.memmap)
    assert not hasattr(got.data, "compute")


def test_tile_cache_misses_rather_than_serving_the_wrong_tile(tmp_path):
    """Every way the key could be stale has to miss, not return someone else's pixels.

    A cache that answered here would feed the wrong array into a bit-for-bit comparison and
    the difference would look like a code bug.
    """
    tile = dict(tile_id="tTEST", xmin=0.0, ymin=0.0, xmax=9990.0, ymax=9990.0)
    cfg = _cache_cfg()
    mt._cache_write(str(tmp_path), tile, cfg, _fake_tile_da())

    assert mt._cache_read(str(tmp_path), dict(tile, xmax=1.0), cfg) is None      # other bbox
    assert mt._cache_read(str(tmp_path), tile, _cache_cfg(years=(2005,))) is None  # other span
    assert mt._cache_read(str(tmp_path), tile, _cache_cfg(resolution=60)) is None  # other res
    # a torn cache is a miss, not an exception
    (tmp_path / "v1_tTEST_30m_2018_2021" / "times.npy").unlink()
    assert mt._cache_read(str(tmp_path), tile, cfg) is None


def test_load_tile_ignores_the_cache_when_the_env_var_is_unset(tmp_path, monkeypatch):
    """The production default. Argo sets nothing, so `load_tile` must never touch disk."""
    monkeypatch.delenv(mt.TILE_CACHE_ENV, raising=False)
    tile = dict(tile_id="tTEST", xmin=0.0, ymin=0.0, xmax=9990.0, ymax=9990.0)
    cfg = _cache_cfg()
    mt._cache_write(str(tmp_path), tile, cfg, _fake_tile_da())

    calls = []

    def fake_load_kndvi(dc, bbox, y0, y1, resolution=30, dask_chunks=None):
        calls.append(bbox)
        return None

    monkeypatch.setattr(mt.mi, "load_kndvi", fake_load_kndvi)
    assert mt.load_tile(None, tile, cfg) is None
    assert len(calls) == 1                    # went to the cube, not to the populated cache


def test_zarr_store_round_trips_bitwise(tmp_path):
    """The materialised cube must return exactly what `load_tile` would have returned.

    Bit-identity is the whole Fase 0 gate (`docs/24`), and the two ways it could quietly fail
    are float precision and datetime encoding -- so this checks values with
    ``equal_nan=True`` and the time axis for exact equality, including a same-day pair, which
    is where a CF round trip through xarray would show up.
    """
    import numpy as np
    import xarray as xr

    from biodiv import maptask as mt

    rng = np.random.default_rng(0)
    v = rng.random((7, 5, 4), dtype=np.float32)
    v[2, 1, 1] = np.nan                                    # NaN must survive as NaN
    times = np.array(["2003-01-01", "2003-01-01", "2003-06-02", "2004-01-01",
                      "2004-07-09", "2005-02-02", "2005-11-30"], dtype="datetime64[ns]")
    da = xr.DataArray(v, coords={"time": times, "y": np.arange(5.0), "x": np.arange(4.0)},
                      dims=("time", "y", "x"), name="kndvi")
    tile = {"tile_id": "t0_0", "xmin": 0.0, "ymin": 0.0, "xmax": 120.0, "ymax": 150.0}
    cfg = mt.TileConfig(years=[2005], resolution=30, dest="", tags={})

    mt.zarr_write(str(tmp_path), tile, cfg, da)
    got = mt._zarr_read(str(tmp_path), tile, cfg)

    assert got is not None
    assert np.array_equal(got.values, v, equal_nan=True)
    assert np.array_equal(got.time.values, times)
    assert np.array_equal(got.x.values, da.x.values)
    assert np.array_equal(got.y.values, da.y.values)


def test_zarr_store_rejects_a_different_grid(tmp_path):
    """A store written for one bbox must not be served to a same-named tile on another."""
    import numpy as np
    import xarray as xr

    from biodiv import maptask as mt

    da = xr.DataArray(np.zeros((2, 3, 3), np.float32),
                      coords={"time": np.array(["2003-01-01", "2004-01-01"], "datetime64[ns]"),
                              "y": np.arange(3.0), "x": np.arange(3.0)},
                      dims=("time", "y", "x"), name="kndvi")
    cfg = mt.TileConfig(years=[2005], resolution=30, dest="", tags={})
    tile = {"tile_id": "t0_0", "xmin": 0.0, "ymin": 0.0, "xmax": 90.0, "ymax": 90.0}
    mt.zarr_write(str(tmp_path), tile, cfg, da)

    moved = dict(tile, xmin=999.0, xmax=1089.0)
    assert mt._zarr_read(str(tmp_path), moved, cfg) is None
    assert mt._zarr_read(str(tmp_path), tile, cfg) is not None


def test_zarr_store_chunks_time_and_still_round_trips(tmp_path):
    """The time axis is cut into blocks, and cutting it changes nothing about the values.

    The default stopped being "one chunk for the whole span" once the cube had to be read
    from S3 (`docs/21` 8.16), and a layout change is exactly the kind of thing that can be
    right on the size report and wrong on the bytes -- so this pins both.
    """
    import numpy as np
    import xarray as xr
    import zarr

    from biodiv import maptask as mt

    rng = np.random.default_rng(1)
    v = rng.random((20, 4, 4), dtype=np.float32)
    v[3, 0, 0] = np.nan
    times = np.arange("2003-01-01", "2003-01-21", dtype="datetime64[D]").astype("datetime64[ns]")
    da = xr.DataArray(v, coords={"time": times, "y": np.arange(4.0), "x": np.arange(4.0)},
                      dims=("time", "y", "x"), name="kndvi")
    tile = {"tile_id": "t0_0", "xmin": 0.0, "ymin": 0.0, "xmax": 120.0, "ymax": 120.0}
    cfg = mt.TileConfig(years=[2005], resolution=30, dest="", tags={})

    p = mt.zarr_write(str(tmp_path), tile, cfg, da, tchunk=6)
    assert zarr.open_group(str(p), mode="r")["values"].chunks == (6, 4, 4)

    got = mt._zarr_read(str(tmp_path), tile, cfg)
    assert got is not None
    assert np.array_equal(got.values, v, equal_nan=True)
    assert np.array_equal(got.time.values, times)


def test_zarr_store_without_attributes_reads_as_a_miss(tmp_path):
    """A store whose attributes never landed must read as a miss, not as a short tile.

    On S3 there is no rename, so `zarr_write` cannot build under a `.part` name and move it
    into place the way the local path does. What stands in for it is the order of writes:
    chunks first, attributes last and in one atomic `put`. That only makes a torn store safe
    if a store *without* attributes is a miss, which is what this pins -- the local branch is
    used here because the property under test is `_zarr_read`'s, not the transport's.
    """
    import numpy as np
    import xarray as xr
    import zarr

    from biodiv import maptask as mt

    da = xr.DataArray(np.zeros((3, 2, 2), np.float32),
                      coords={"time": np.array(["2003-01-01", "2004-01-01", "2005-01-01"],
                                               "datetime64[ns]"),
                              "y": np.arange(2.0), "x": np.arange(2.0)},
                      dims=("time", "y", "x"), name="kndvi")
    cfg = mt.TileConfig(years=[2005], resolution=30, dest="", tags={})
    tile = {"tile_id": "t0_0", "xmin": 0.0, "ymin": 0.0, "xmax": 60.0, "ymax": 60.0}

    p = mt.zarr_write(str(tmp_path), tile, cfg, da)
    assert mt._zarr_read(str(tmp_path), tile, cfg) is not None

    zarr.open_group(str(p), mode="r+").attrs.put({})    # as if the run died before the put
    assert mt._zarr_read(str(tmp_path), tile, cfg) is None


def test_zarr_path_builds_an_s3_uri():
    """An ``s3://`` root yields a URI, not a `Path` that would collapse the double slash."""
    from pathlib import Path

    from biodiv import maptask as mt

    cfg = mt.TileConfig(years=[2005, 2015, 2024], resolution=30, dest="", tags={})
    tile = {"tile_id": "t18_600", "xmin": 0.0, "ymin": 0.0, "xmax": 60.0, "ymax": 60.0}

    uri = mt._zarr_path("s3://bucket/prefix/", tile, cfg)
    assert uri == f"s3://bucket/prefix/v{mt._CACHE_VERSION}_t18_600_30m_2003_2024.zarr"
    assert isinstance(mt._zarr_path("/tmp/cube", tile, cfg), Path)
