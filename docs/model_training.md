# Model training (LightGBM)

Phase 3 trains composition-based regressors on `data/processed/training_exports/`.
Weights stay local under `data/models/` and are not committed.

## Quick start

```bash
pip install -e ".[dev]"
python scripts/train_models.py --all-ready
python scripts/train_models.py --all-niche
python scripts/predict_models.py --target density --formula SiO2 --formula Au
```

Config file: [`configs/models.yaml`](../configs/models.yaml).

## Ready vs Niche

| Tier | Role | Targets |
|------|------|---------|
| **Ready** | Primary supervised heads; conflict rows excluded where configured | `energy_above_hull`, `formation_energy_per_atom`, `band_gap`, `density` |
| **Niche** | Smaller / method-specific sets; train after Ready baselines look sane | elastic B/G, AGL thermal κ, Debye T, heat capacity Cᵥ, battery capacity |

Hold-out ID lists under `training_exports/holdouts/` are for evaluation slices (see that folder’s README). Default train/val/test splits still include those materials unless you filter explicitly.

## Features

From each target parquet (and formula at predict time):

1. **Element fractions** — `frac_H` … `frac_Og` (118)
2. **Composition stats** — mean/max Z, mean mass, `n_elements`
3. **Element descriptors** (derived from `formula_pretty`) — mean electronegativity, atomic radius, row/group/block, metal fraction, Z std, ionization energy

Structure-only inputs (CIF graphs, etc.) are out of scope for this Phase 3 pipeline.

## Labels and transforms

- Each export table has a `label` column (already gated).
- Some targets use a `log1p` label transform in model space (`configs/models.yaml` → `targets.*.transform`).
- Reported metrics (MAE / RMSE / R²) are always in **original** physical units after inverse transform.

## Splits

- Column: `split` ∈ {`train`, `val`, `test`}
- Global recipe: 80 / 10 / 10, seed `42`, fingerprint-group leak safety  
  (see `DATASET_CARD.md`)

## Training behavior

- Estimator: LightGBM regressor
- Early stopping on the validation split (`training.early_stopping_rounds`)
- Global hyperparameters plus optional per-target overrides in `models.yaml`
- CLI helpers:
  - `--max-train-rows` — smoke / debug subsample
  - `--n-estimators` — override the tree cap
  - `--no-predictions` — skip `predictions_*.parquet`

## Artifacts

```text
data/models/
  registry.json
  LEADERBOARD.md
  <target>/
    <target>_<UTC>/
      model.joblib
      feature_cols.json
      metrics.json
      run_record.json
      predictions_{train,val,test}.parquet   # optional
```

`LEADERBOARD.md` summarizes the latest registered run per target (test MAE / RMSE / R²).

## Reading metrics

| Metric | Meaning |
|--------|---------|
| MAE | Mean absolute error (original units) |
| RMSE | Root mean squared error |
| R² | Coefficient of determination on that split |

Representative local composition-only test R² (re-train to refresh):

| Target | Tier | Test R² (approx.) |
|--------|------|------------------:|
| formation_energy_per_atom | Ready | ~0.91 |
| debye_temperature | Niche | ~0.89 |
| bulk_modulus_vrh | Niche | ~0.81 |
| density | Ready | ~0.81 |
| band_gap | Ready | ~0.73 |
| battery_capacity_grav | Niche | ~0.72 |
| thermal_conductivity_agl | Niche | ~0.67 |
| shear_modulus_vrh | Niche | ~0.66 |
| heat_capacity_cv | Niche | ~0.55 |
| energy_above_hull | Ready | ~0.40 |

Hull and gap are especially limited without structure features; pretrained MACE / ALIGNN-style models are the planned next quality jump.

## Package layout (`src/ml/`)

| Module | Role |
|--------|------|
| `targets` | Ready / Niche specs from `configs/models.yaml` |
| `data` | Load export parquets → train/val/test arrays (train-median impute) |
| `composition_features` | Formula → fractions / stats / elemental descriptors |
| `transforms` | Optional `log1p` label map + inverse |
| `metrics` | MAE / RMSE / R² helpers |
| `train` | Fit LightGBM, write run artifacts, update registry |
| `predict` | Load runs; predict from formulas or matrices |
| `registry` | `registry.json` + `LEADERBOARD.md` |
| `io_util` | Shared JSON path helpers |

## Python API

```python
from ml.data import load_xy_splits
from ml.train import train_target
from ml.predict import predict_formulas

data = load_xy_splits("band_gap")
result = train_target("band_gap")
frame = predict_formulas("band_gap", ["SiO2", "GaN"])
```

## Related docs

- [`data/processed/training_exports/DATASET_CARD.md`](../data/processed/training_exports/DATASET_CARD.md) — export freeze and per-target counts
- [`docs/label_policy.md`](label_policy.md) — Ready / Niche / Proxy tiers
- [`docs/exclusion_rules.md`](exclusion_rules.md) — conflict and gate rules
