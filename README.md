# Materials Predictor

Open-source materials **data collection** and a frozen **training export** package.

This repository ships:

1. Folder skeleton for raw / processed / structure data
2. Code to download, normalize, and catalog materials from public sources
3. The quality-gated training tables under `data/processed/training_exports/`

Raw downloads, the live DuckDB catalog, crystal structure files, and trained models are **not** committed (see `.gitignore`).

## Training export (`data/processed/training_exports/`)

Version **4.1.0** — parquet targets, leak-safe splits, specialty hold-outs, and `DATASET_CARD.md`.

| Tier | Targets |
|------|---------|
| Ready | `energy_above_hull`, `formation_energy_per_atom`, `band_gap`, `density` |
| Niche | elastic B/G, AGL thermal / Debye / Cᵥ, battery capacity |

Ready heads exclude cross-source label conflicts. See `DATASET_CARD.md`, `docs/label_policy.md`, and `docs/exclusion_rules.md`.

## Layout

```text
configs/database.yaml     # ingestion + DuckDB paths
data/
  raw/                    # source downloads (local only)
  processed/
    training_exports/     # published dataset
    materials.duckdb      # local catalog (gitignored)
  structures/             # CIFs / structure files (local only)
  models/                 # reserved; empty in this release
docs/                     # label + exclusion policy
scripts/                  # download / build / repair / export CLIs
src/
  databases/              # source clients
  ingestion/              # raw download writers
  normalization/          # source → unified schema
  database/               # DuckDB catalog, flags, entity resolution
  data_packs/             # derived enrichment tables
  ml/composition_features.py  # composition features for exports
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

## Constraints

- Do not commit `.env` or API keys
- Do not commit raw dumps, DuckDB files, or structure trees
- Preserve source provenance; never silently average conflicting labels
- Proxy / derived properties are not DFT ground truth

## License

MIT — see `LICENSE`.
