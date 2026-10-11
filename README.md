# Materials Predictor

Open-source materials **data collection**, a frozen **training export** package,
**LightGBM** composition models, and an optional **frozen-first MACE** structure
stack for formation energy / stability scoring.

This repository ships:

1. Folder skeleton for raw / processed / structure data
2. Code to download, normalize, and catalog materials from public sources
3. Quality-gated training tables under `data/processed/training_exports/`
4. A LightGBM training + prediction pipeline (`src/ml/`, `scripts/train_models.py`)
5. Optional MACE structure models (`src/ml/mace/`, `pip install -e ".[mace]"`)

Raw downloads, the live DuckDB catalog, crystal structure files, and trained model
weights are **not** committed (see `.gitignore`).

## Training export (`data/processed/training_exports/`)

Export version **4.1.0** — parquet targets, leak-safe splits, specialty hold-outs, and `DATASET_CARD.md`.

| Tier | Targets |
|------|---------|
| Ready | `energy_above_hull`, `formation_energy_per_atom`, `band_gap`, `density` |
| Niche | `bulk_modulus_vrh`, `shear_modulus_vrh`, `thermal_conductivity_agl`, `debye_temperature`, `heat_capacity_cv`, `battery_capacity_grav` |

Ready heads exclude cross-source label conflicts. See `DATASET_CARD.md`, `docs/label_policy.md`, and `docs/exclusion_rules.md`.

## Train models (LightGBM)

Composition-only tabular models on the training export. Config: `configs/models.yaml`. Details: [`docs/model_training.md`](docs/model_training.md).

```bash
# Ready targets (default scope)
python scripts/train_models.py --all-ready

# Niche targets
python scripts/train_models.py --all-niche

# Single target
python scripts/train_models.py --target band_gap

# Predict with the latest local run
python scripts/predict_models.py --target band_gap --formula SiO2 --formula GaN
```

Artifacts are written under `data/models/<target>/<run_id>/` (`model.joblib`, metrics, optional predictions) plus `data/models/registry.json` and `LEADERBOARD.md`. **Do not commit these files** — they are gitignored.

## Structure models (MACE, optional)

Frozen-first strategy: pretrained MACE-MPA-0 for E_form / hull ranking; LightGBM
remains the composition bar. Promote on **Spearman** (MAE reported). Fine-tune is
last resort. Band gap / moduli → frozen ALIGNN (deferred wiring).

- Strategy: [`docs/mace_strategy.md`](docs/mace_strategy.md)
- Contracts: [`docs/mace_plan.md`](docs/mace_plan.md)
- Runbook: [`docs/mace_runbook.md`](docs/mace_runbook.md)
- Config: [`configs/mace.yaml`](configs/mace.yaml)

```bash
pip install -e ".[mace]"   # optional; prefer Python 3.12/3.13 (avoid 3.14 on macOS)

python scripts/smoke_mace.py --device cpu
python scripts/diagnose_mace_harness.py --limit 200 --device cpu
python scripts/run_mace_pipeline.py --preset smoke --dry-run

# Candidate scoring: relax → E_form → E_hull
python scripts/cache_mace_eform.py --device cpu --limit-per-split 128
python scripts/score_mace_structures.py --structure path/to/POSCAR
python scripts/predict_mace.py --material-id MAT-1   # default: mace_pretrained
python scripts/compare_mace_lightgbm.py --family mace_pretrained
```

Cite MACE-MPA-0: Batatia et al., arXiv:2401.00096.

## Layout

```text
configs/
  database.yaml           # ingestion + DuckDB paths
  models.yaml             # LightGBM features, splits, targets, hyperparameters
  mace.yaml               # MACE contracts (foundation model, units, gates)
data/
  raw/                    # source downloads (local only)
  processed/
    training_exports/     # published dataset (committed)
    materials.duckdb      # local catalog (gitignored)
  structures/             # CIFs / structure files (local only)
  models/                 # trained runs (local only; gitignored)
docs/
  label_policy.md
  exclusion_rules.md
  model_training.md
  mace_plan.md            # Phase 4 MACE contracts
  mace_strategy.md        # Frozen-first strategy (binding)
  mace_training.md        # MACE install / howto
  mace_runbook.md         # Production runbook (smoke vs full)
scripts/                  # download / build / export / train / predict / mace CLIs
src/
  databases/ ingestion/ normalization/ database/ data_packs/
  ml/                     # LightGBM + ml/mace (optional structure stack)
tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add MP_API_KEY if pulling Materials Project
```

## Data collection (local)

```bash
python scripts/download_data.py --source materials_project   # or jarvis / oqmd / aflow / cod / all
python scripts/build_database.py --source all
python scripts/run_entity_resolution.py
python scripts/refresh_conflict_flags.py
python scripts/refresh_train_flags.py
python scripts/export_training_package.py
```

Source-specific pipelines (`run_*_pipeline.py`), structure downloads, and repair/enrich scripts live under `scripts/`.

## Tests

```bash
pytest
```

## Constraints

- Do not commit `.env` or API keys
- Do not commit raw dumps, DuckDB files, structure trees, or trained weights
- Preserve source provenance; never silently average conflicting labels
- Proxy / derived properties are not DFT ground truth

## License

MIT — see `LICENSE`.
