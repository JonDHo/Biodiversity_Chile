"""Gates for the map-inference path: it must reproduce the training preprocessing.

Runs offline: no datacube, no checkpoints on disk. The ensemble test builds a model,
fits the same Preprocessor/PowerTransformer the trainer uses, saves a checkpoint with the
same keys as `dl_runner.run_dl` and reloads it through `FacetEnsemble`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biodiv import mapinfer as mi                  # noqa: E402
from biodiv.curves import interp_grid              # noqa: E402
from biodiv.features import STRATA, TOPO_VARS, Preprocessor  # noqa: E402
from biodiv.transforms1d import make_transform     # noqa: E402


def _synthetic(T=140, N=37, seed=0, gap=0.35):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 1095, size=T))
    base = 0.3 + 0.2 * np.sin(2 * np.pi * t / 365.25)[:, None]
    obs = base + 0.05 * rng.standard_normal((T, N))
    obs[rng.uniform(size=(T, N)) < gap] = np.nan
    # a pixel with too few observations, one with none, one with a single run at the end
    obs[:, 0] = np.nan
    obs[:, 1] = np.nan
    obs[-3:, 1] = 0.4
    obs[:100, 2] = np.nan
    return t, obs.astype(np.float32)


def test_interp_common_grid_matches_interp_grid_pixel_by_pixel():
    t, obs = _synthetic()
    grid = np.linspace(t.min(), t.max(), mi.NGS)
    curves, n_obs = mi.interp_common_grid(t, obs, grid, roll=5)
    for j in range(obs.shape[1]):
        v = obs[:, j].astype(float)
        ok = np.isfinite(v)
        if ok.sum() < mi.MIN_OBS:
            assert np.isnan(curves[j]).all(), j
            continue
        ref = interp_grid(t[ok], v[ok], mi.NGS, 5, t_min=grid[0], t_max=grid[-1])
        np.testing.assert_allclose(curves[j], ref, rtol=0, atol=2e-6, err_msg=f"pixel {j}")
    assert n_obs[0] == 0 and n_obs[1] == 3


def test_interp_common_grid_handles_unsorted_times():
    t, obs = _synthetic(seed=3)
    perm = np.random.default_rng(1).permutation(len(t))
    grid = np.linspace(t.min(), t.max(), 50)
    a, _ = mi.interp_common_grid(t, obs, grid)
    b, _ = mi.interp_common_grid(t[perm], obs[perm], grid)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_year_curves_window_and_grid_span():
    times = np.arange("2003-01-05", "2007-12-30", 16, dtype="datetime64[D]")
    T = len(times)
    rng = np.random.default_rng(0)
    obs = (0.4 + 0.1 * rng.standard_normal((T, 5))).astype(np.float32)
    curves, n_obs, lo, hi = mi.year_curves(times, obs, 2005)
    m = mi.window_mask(times, 2005)
    assert m.sum() == n_obs[0]
    assert times[m].min() >= np.datetime64("2003-01-01")
    assert times[m].max() <= np.datetime64("2005-12-31")
    assert np.isclose(lo, mi.days_since_epoch(times[m])[0])
    assert np.isclose(hi, mi.days_since_epoch(times[m])[-1])
    assert curves.shape == (5, mi.NGS) and np.isfinite(curves).all()
    # a year without any acquisition
    c2, n2, lo2, hi2 = mi.year_curves(times, obs, 2030)
    assert np.isnan(c2).all() and n2.sum() == 0 and np.isnan(lo2)


def test_serpentine_perm_is_the_training_transform():
    rng = np.random.default_rng(0)
    curve = rng.uniform(size=mi.NGS)
    perm = mi.serpentine_perm(mi.NGS)
    assert perm.shape == (10, 10)
    ref = make_transform("serpentine", normalize="none").transform(curve)
    np.testing.assert_allclose(curve[perm], ref)
    imgs = mi.images_from_curves(curve[None, :], perm)
    assert imgs.shape == (1, 1, 10, 10) and imgs.dtype == np.float32
    np.testing.assert_allclose(imgs[0, 0], ref, rtol=1e-6)


def test_context_frame_columns_and_flags():
    n = 4
    topo = {v: np.linspace(0, 1, n) for v in TOPO_VARS}
    topo["heat_load"] = np.array([0.5, np.nan, 0.7, 0.9])
    cols = [f"topo_{v}" for v in TOPO_VARS] + ["topo_topo_flat", "area_log10"] + \
           [f"area_stratum_{s}" for s in STRATA[1:]]
    df = mi.context_frame(cols, topo, 500.0, "basal")
    assert list(df.columns) == cols
    assert df["topo_topo_flat"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert np.allclose(df["area_log10"], np.log10(500))
    assert df["area_stratum_basal"].tolist() == [1.0] * n
    assert df["area_stratum_counts"].tolist() == [0.0] * n
    with pytest.raises(ValueError):
        mi.context_frame(cols + ["not_a_column"], topo, 500.0, "basal")
    with pytest.raises(ValueError):
        mi.context_frame(cols, topo, 500.0, "trees")


def test_ensemble_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from sklearn.preprocessing import PowerTransformer
    from biodiv.models_conv import build_model

    rng = np.random.default_rng(0)
    n, n_t = 60, 3
    targets = ["a", "b", "c"]
    cols = [f"topo_{v}" for v in TOPO_VARS] + ["topo_topo_flat", "area_log10"] + \
           [f"area_stratum_{s}" for s in STRATA[1:]]
    X = pd.DataFrame(rng.standard_normal((n, len(cols))), columns=cols)
    X.loc[X.sample(5, random_state=0).index, "topo_heat_load"] = np.nan   # -> isna column
    pre = Preprocessor(standardise=True).fit(X, X.index)
    scaler = PowerTransformer(method="yeo-johnson", standardize=True).fit(
        np.abs(rng.standard_normal((n, n_t))) + 1)
    ck_paths = []
    for seed in range(2):
        torch.manual_seed(seed)
        model = build_model("C2D", c_in=1, n_out=n_t, n_ctx=len(pre.feature_names),
                            width="B", pad_mode="zeros", fusion="late")
        p = tmp_path / f"model_seed{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "n_params": 0, "input_shape": (1, 10, 10),
                    "rows": [f"row{i}" for i in range(10)], "targets": targets,
                    "ctx_preprocessor": pre, "target_scaler": scaler, "fit_ids": [],
                    "train_ids": [], "test_ids": []}, p)
        ck_paths.append(p)

    ens = mi.FacetEnsemble(ck_paths)
    assert ens.targets == targets and ens.context_columns == cols
    curves = rng.uniform(0.1, 0.7, size=(n, mi.NGS)).astype(np.float32)
    curves[3] = np.nan                                  # a pixel without a curve
    images = mi.images_from_curves(curves, mi.serpentine_perm())
    pred = ens.predict(images, X)
    assert pred.shape == (n, n_t)
    assert np.isnan(pred[3]).all() and np.isfinite(np.delete(pred, 3, axis=0)).all()
    # original units: back through the scaler, so strictly positive here
    assert (np.delete(pred, 3, axis=0) > 0).all()
    # seed mean of per-member inverses, not inverse of the mean
    scaled = ens.predict_scaled(images, X)
    assert scaled.shape == (2, n, n_t)
    # clipping to the training range, in transformed and in original space
    y_train = np.abs(rng.standard_normal((n, n_t))) + 1
    y_train[::7, 1] = np.nan
    zlo, zhi = ens.scaled_bounds(y_train)
    assert zlo.shape == (n_t,) and (zlo < zhi).all()
    clipped = ens.predict(images, X, y_train=y_train)
    keep = np.delete(np.arange(n), 3)
    assert (clipped[keep] >= np.nanmin(y_train, axis=0) - 1e-6).all()
    assert (clipped[keep] <= np.nanmax(y_train, axis=0) + 1e-6).all()


def test_tile_grid_snaps_and_covers():
    g = mi.tile_grid((100_000.0, 5_000_000.0, 125_000.0, 5_012_000.0), 10_000, 30)
    assert set(g.columns) == {"tile_id", "xmin", "ymin", "xmax", "ymax"}
    assert g.xmin.min() <= 100_000 and g.xmax.max() >= 125_000
    assert g.ymin.min() <= 5_000_000 and g.ymax.max() >= 5_012_000
    step = 333 * 30                                   # 10 km snapped to whole pixels
    assert ((g.xmin % step) == 0).all() and ((g.ymin % step) == 0).all()
    assert ((g.xmax - g.xmin) == step).all()
    assert len(g) == 3 * 2                              # x: 10..12, y: 500..501
    g2 = mi.tile_grid((0, 0, 1, 1), 10_001, 30)
    assert float(g2.xmax.iloc[0] - g2.xmin.iloc[0]) == step
    with pytest.raises(ValueError):
        mi.tile_grid((0, 0, 1, 1), 1, 30)


# --------------------------------------------------------------------------------------
# what makes it safe to mask before interpolating (`maptask.run_tile`)
# --------------------------------------------------------------------------------------

def test_finite_curve_is_exactly_the_min_obs_threshold():
    """`isfinite(curves).all(axis=1) == (n_obs >= MIN_OBS)`, the identity `run_tile` rests on.

    It lets the native mask be applied before the interpolation instead of after it. If a
    curve could ever be *partially* finite, or finite below MIN_OBS, the reordering would
    silently change which pixels are predicted -- so this is checked at every count on
    either side of the threshold, not just on a random sample.
    """
    rng = np.random.default_rng(11)
    T, G = 60, mi.NGS
    t = np.sort(rng.uniform(0, 1095, T))
    grid = np.linspace(t[0], t[-1], G)
    counts = list(range(0, 2 * mi.MIN_OBS)) + [T]
    obs = np.full((T, len(counts) * 8), np.nan, np.float32)
    for i, c in enumerate(counts):
        for r in range(8):
            rows = rng.choice(T, size=c, replace=False)
            obs[rows, i * 8 + r] = rng.normal(0.3, 0.05, c)

    curves, n_obs = mi.interp_common_grid(t, obs, grid)
    finite = np.isfinite(curves)
    assert np.array_equal(finite.all(axis=1), n_obs >= mi.MIN_OBS)
    # never partially finite: a curve is all-NaN or has no NaN at all
    assert not (finite.any(axis=1) & ~finite.all(axis=1)).any()


def test_interp_common_grid_column_subset_matches_the_full_result():
    """Interpolating a subset of pixels gives those pixels' rows of the full result."""
    t, obs = _synthetic(T=120, N=400, seed=5)
    grid = np.linspace(t.min(), t.max(), mi.NGS)
    full, n_full = mi.interp_common_grid(t, obs, grid)
    sel = np.random.default_rng(2).choice(obs.shape[1], 137, replace=False)
    sub, n_sub = mi.interp_common_grid(t, np.ascontiguousarray(obs[:, sel]), grid)
    assert np.array_equal(full[sel], sub, equal_nan=True)
    assert np.array_equal(n_full[sel], n_sub)


@pytest.mark.parametrize("block", [1, 7, 64, 399, 400, 4096])
def test_interp_common_grid_blocking_is_bit_identical(block):
    """Blocking bounds the transient; it must not move a single bit of the result.

    Includes block sizes that do not divide N, and one larger than N.
    """
    t, obs = _synthetic(T=120, N=400, seed=6)
    grid = np.linspace(t.min(), t.max(), mi.NGS)
    a, na = mi.interp_common_grid(t, obs, grid)
    b, nb = mi.interp_common_grid(t, obs, grid, block=block)
    assert np.array_equal(a, b, equal_nan=True)
    assert np.array_equal(na, nb)


def test_year_window_grid_span_comes_from_every_pixel():
    """The grid spans the window's first/last date clear *anywhere in the tile*.

    This is the plot-cube convention, and it is why `run_tile` may mask before it
    interpolates but may not mask before it computes the span: here only pixel 3 is ever
    clear on the first and last dates, so a span computed from the other pixels would be
    narrower and every curve in the tile would land on a different grid.
    """
    times = np.arange("2003-01-05", "2006-01-01", 16, dtype="datetime64[D]")
    T = len(times)
    obs = np.full((T, 6), np.nan, np.float32)
    m = mi.window_mask(times, 2005)
    idx = np.flatnonzero(m)
    obs[idx, 3] = 0.4                                  # pixel 3 clear on every window date
    obs[idx[5:-5], :3] = 0.35                          # the rest clear only in the middle

    t, o, n_obs, grid, lo, hi = mi.year_window(times, obs, 2005)
    days = mi.days_since_epoch(times[m])
    assert np.isclose(lo, days[0]) and np.isclose(hi, days[-1])
    assert grid is not None and len(grid) == mi.NGS
    # and `year_curves` agrees with the split it was refactored into
    curves, n2, lo2, hi2 = mi.year_curves(times, obs, 2005)
    assert np.array_equal(n_obs, n2) and lo == lo2 and hi == hi2


def test_year_window_reports_no_grid_for_an_empty_window():
    times = np.arange("2003-01-05", "2006-01-01", 16, dtype="datetime64[D]")
    obs = np.full((len(times), 4), 0.4, np.float32)
    t, o, n_obs, grid, lo, hi = mi.year_window(times, obs, 2030)
    assert grid is None and t is None and o is None
    assert n_obs.sum() == 0 and np.isnan(lo) and np.isnan(hi)
    # nothing is usable, so `run_tile` predicts nothing -- the identity still holds
    assert not (n_obs >= mi.MIN_OBS).any()
