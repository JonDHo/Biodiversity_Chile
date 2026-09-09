# Biodiversity_Chile

Multitemporal maps of several plant-diversity facets for native vegetation in Chile, predicted
from the three-year Landsat phenological trajectory of each pixel and its topography.

The premise is that a single-date vegetation index is the wrong predictor. What a pixel does
across three growing seasons — when it greens up, how much, how consistently — carries more
about the vegetation standing there than how green it happened to be on one clear day. The
model reads that whole trajectory, and predicts several facets at once from one shared trunk,
because species richness, phylogenetic depth and compositional uniqueness are different
questions that a single map cannot answer.

**Status.** Modelling is complete on the unified plot pool and the deployment model is fitted;
map production over the native-vegetation extent is in progress. A manuscript describing the
method and results is in preparation — this section will carry the reference once it is
available.

---

## What is predicted, and how well

Seven facets over 3,102 vegetation plots (1,082 from Parcelas-CL, 2,020 from Living Trees
Chile), spanning 30.2°–55.0°S:

| Facet | What it is |
|---|---|
| `TD₀`, `TD₁`, `TD₂` | Taxonomic Hill numbers, coverage-standardised (iNEXT) |
| `PD₀`, `PD₁`, `PD₂` | Phylogenetic Hill numbers, coverage-standardised |
| `LCBD` | Local contribution to beta diversity, quantitative Sørensen |

Under spatial block cross-validation (20 km blocks), out-of-fold R² separates them into three
groups that no model family crosses:

- **Richness is predictable.** TD₀ reaches 0.78 and PD₀ 0.60.
- **Compositional uniqueness is moderately predictable.** LCBD 0.42–0.45.
- **Abundance-weighted facets are not.** `q = 1` and `q = 2` sit at or near zero under every
  model, vegetation index and curve format tested. This is a limit of the signal, not of the
  architecture.

**The binding constraint is time, not model capacity.** Under leave-location-and-time-out
validation — holding out whole spatial blocks *and* whole census periods — richness collapses
in every neural family, while LCBD retains 0.34–0.42. The maps are therefore published as
spatial interpolation within the sampled domain and period, and carry no accuracy claim for
census years the plot network never observed. See [`docs/20`](docs/20_pg_facets_unified.md) §7.

## The deployed model

A one-dimensional convolutional network (`Pheno1D`, 14,535 parameters) over the 100-step raw
kNDVI series of the plot pixel, with topography and plot context fused late, refit on all
3,102 plots with no held-out fold and ensembled over five seeds.

It began as the *control* for a two-dimensional network that folds the series into an image
with a serpentine layout. The second dimension brought no measurable gain — the three networks
separate by less than their own seed dispersion — so the smaller and simpler model is the one
deployed. Masked-autoencoder pretraining was also tested and is reported as exploratory: its
pretraining pool overlaps the test folds on the input side, and it showed no measurable
benefit at this sample size.

Map settings that are recorded choices rather than defaults — plot area held at 900 m² (one
Landsat pixel), `basal` recording protocol, native-vegetation mask from the nearest annual
MapBiomas map, 10 km tiles at 30 m for 2000–2026 — are documented with their reasoning in
[`docs/21`](docs/21_map_inference_spec.md).

---

## Running it

### Analysis, from a clone

`data/derived/` is versioned deliberately. Those tables are the output of scripts 01–05, which
need the Data Observatory datacube and cannot be regenerated on an arbitrary machine, so a
fresh clone runs the analysis notebooks and the modelling scripts with no further setup:

```bash
pip install -r requirements.txt
python scripts/53_unified_block20_folds.py     # spatial-block folds on the unified pool
python scripts/11_run_conv.py --substrate curve1d --index kndvi --scheme kfold5_block20_unified
```

The source databases themselves are **not** versioned here; they are public and retrieved by
DOI (see *Data* below).

### Map inference

Inference needs four staged inputs — checkpoints, the out-of-fold predictions the
retransformation depends on, the derived target tables and the tile list — plus the MapBiomas
rasters and read access to Data Cube Chile.

`biodiv.assets` resolves those the same way the production workflow does, so the same notebook
runs from a local tree or from S3:

```python
from biodiv import assets
print(assets.describe())          # says where each input is coming from
ckpts = sorted(assets.ckpt_dir().glob("model_seed*.pt"))
```

```bash
# defaults point at the workflow's own prefix; credentials are all a reader needs
export BIODIV_ASSETS=s3://<bucket>/<prefix>/assets
export BIODIV_MAPBIOMAS_DIR=s3://<bucket>/<prefix>/MapBiomas   # optional, same default

python scripts/73_map_inference.py --tiles-file <tiles.csv> --years 2000-2026 \
    --area-m2 900 --dest s3://<bucket>/<prefix>/maps --resume
```

Point both variables at local directories instead and nothing else changes. `scripts/74`
is the gate that must pass first: it checks that the map path reproduces the training path
exactly — same context block, same model input, same output — and no map is produced if it
fails.

**Landsat itself is not staged.** It is read from the Data Cube Chile ODC index, which only
answers from inside the EASI cluster. There is no public STAC fallback in this code path.

### At scale

[`scripts/argo/process_argo.yaml`](scripts/argo/process_argo.yaml) fans the same script out
over tiles on Kubernetes: it lists what is already written to S3, splits the remainder into
per-pod chunks, and runs them in parallel. Tiles are deduplicated against S3 rather than a
separate tracker, so a rerun never repeats finished work.

---

## Data

| Source | Contents | Access |
|---|---|---|
| **Parcelas-CL** | 1,485 georeferenced vegetation plots, 675 woody species, 1976–2026, 30.25°–54.82°S | [10.5281/zenodo.20602096](https://doi.org/10.5281/zenodo.20602096) · preprint [10.21203/rs.3.rs-9986019/v1](https://doi.org/10.21203/rs.3.rs-9986019/v1) |
| **Living Trees Chile** | 2,021 forest inventory sites, 59,408 stem records | Used by collaboration; citation pending publication |
| **MapBiomas Chile** | Annual land cover, collection 2, 1999–2024 | [mapbiomas.org](https://chile.mapbiomas.org/) |
| **Landsat** | Collection 2 surface reflectance, via Data Cube Chile | Requester-pays; ODC index inside EASI |

---

## Documentation

Written in Spanish, and written to be read: each document records the decisions that were not
obvious and what was measured to settle them, not just what was done.

| Document | Contents |
|---|---|
| [`01_state_of_the_art.md`](docs/01_state_of_the_art.md) | Critical review across seven axes, closing with a table of five gaps |
| [`02_innovation_and_impact.md`](docs/02_innovation_and_impact.md) | Graded claim of contribution, quantified risks, and an explicit section on what the project will *not* demonstrate |
| [`03_cnn_architecture.md`](docs/03_cnn_architecture.md) | `PhenoNet-S` design for n ≈ 1,000 plots; signal-to-image transforms; experimental matrix |
| [`06_phase_and_2d_transform.md`](docs/06_phase_and_2d_transform.md) | Phase anchoring of the seasonal curve and the global rotation |
| [`08_modelling.md`](docs/08_modelling.md) | The benchmark on the Parcelas-CL-only pool: retransformation bias, circular DOY, centre pixel vs 5×5, paired tests |
| [`14_cnn_search_results.md`](docs/14_cnn_search_results.md) | 195 runs of architecture search; the raw three-year series as the only real finding; measured GPU noise floor |
| [`16_stemp_protocol.md`](docs/16_stemp_protocol.md) | Spatio-temporal modelling protocol and the cross-validation schemes |
| [`19_unified_facets_methodology.md`](docs/19_unified_facets_methodology.md) | Building the unified pool and its coverage-standardised facets |
| [`20_pg_facets_unified.md`](docs/20_pg_facets_unified.md) | **Main results:** curve format, vegetation index, model families, spatio-temporal transfer, and the topography-correction addendum |
| [`21_map_inference_spec.md`](docs/21_map_inference_spec.md) | **Map specification:** every mapping decision with its justification, the consistency gate, the cost budget and the run record |
| [`07_run_record.md`](docs/07_run_record.md) | Reproducible acquisition record with [`run_manifest.json`](docs/run_manifest.json) (`sha256` inventory) |

The remaining documents (`05`, `09`–`13`, `15`–`18`, `22`–`23`) cover acquisition, predictor
screening, phylogeny and rarefaction, sampling design and bibliographic search.

## Related repositories

| Repository | Role |
|---|---|
| [PhenoSensing](https://github.com/JavierLopatin/PhenoSensing) | Phenological curve reconstruction (`PhenoShape`) and 18 LSP metrics. Provides the predictor. |
| [Trait_2DCNN](https://github.com/JavierLopatin/Trait_2DCNN) | Signal-to-image transforms, masked multi-target loss, MAE pretraining. Provides the modelling framework. |

---

## Funding

ANID FONDECYT Iniciación 11241088 · FSEQ210022 · Fundación Data Observatory.

## License

To be defined. Until a licence is added, no permission is granted beyond viewing the code;
if you want to use or build on it, please open an issue.
