# MACE training / runtime (Phase 4+)

Contracts (binding): [`docs/mace_plan.md`](mace_plan.md).  
Config: [`configs/mace.yaml`](../configs/mace.yaml).  
**Production runbook (Phase 4.9):** [`docs/mace_runbook.md`](mace_runbook.md).  
**Strategy (frozen-first):** [`docs/mace_strategy.md`](mace_strategy.md).

## Phase 4.1 — optional install + smoke

The LightGBM stack stays the default. MACE is an **optional** extra.

### Known-good stack (floors)

| Package | Floor | Notes |
|---------|------:|-------|
| Python | 3.11–3.13 | Project allows `>=3.11`; **avoid 3.14 for MACE** (see below) |
| `torch` | `>=2.2` | Install a CUDA wheel from [pytorch.org](https://pytorch.org) first if you need GPU; then install the extra |
| `ase` | `>=3.22` | Structure / calculator host |
| `mace-torch` | `>=0.3.10` | Needed for `medium-mpa-0` loader id |

Avoid relying on bare `mace_mp()` defaults — always pass `model=` (pinned: `medium-mpa-0`).
Default dtype is **`float32`**. On Python 3.14, `float64` is coerced to `float32`.

### “Python quit unexpectedly” (macOS)

This is a **native segfault** in torch/MACE on some wheels (seen on **Python 3.14**),
not an app bug in our scripts. Triggers:

1. Missing thread caps (`OMP_NUM_THREADS` etc.) before the first torch import
2. `float64` MACE forwards on 3.14

Mitigations already in code: `ml.mace.runtime.configure_mace_runtime()` runs
before smokes and sets single-threaded BLAS/torch. Still prefer a stabler
interpreter when available:

```bash
# Recommended on this machine (Homebrew has 3.13)
python3.13 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,mace]"
```

### Install

```bash
python3.13 -m venv .venv   # or python3.12; avoid 3.14 for MACE
source .venv/bin/activate
pip install -U pip
# Optional GPU: install the matching torch wheel, then:
pip install -e ".[dev,mace]"
```

### Smoke (one energy)

Uses ASE diamond Si and the pinned foundation checkpoint. Weights are cached under
`data/models/mace/checkpoints/` via `requests` (gitignored). First run needs
network; later runs are offline.

```bash
python scripts/smoke_mace.py --device cpu
python scripts/smoke_mace.py --device cpu --json
```

Programmatic:

```python
from ml.mace.smoke import run_smoke

result = run_smoke(device="cpu")
print(result.energy_eV, result.energy_per_atom_eV)
```

If download fails (e.g. macOS Homebrew SSL), place
`mace-mpa-0-medium.model` in `data/models/mace/checkpoints/` manually from the
[MACE-MPA-0 release](https://github.com/ACEsuit/mace-mp/releases/tag/mace_mpa_0).

### Devices

`ml.mace.device.resolve_device` honors `configs/mace.yaml` → `device.preference`
(`cuda` → `mps` → `cpu`) unless you pass an explicit device. Explicit `cuda` /
`mps` when unavailable raises a clear `RuntimeError`.

### Tests

Default pytest does **not** download foundation weights:

```bash
pytest tests/test_mace
```

Full smoke (network + checkpoint cache):

```bash
MACE_SMOKE=1 pytest tests/test_mace/test_smoke.py
```

### Phase 4.2 — structure I/O + audit

Load repo `structure_path` files into ASE `Atoms` (`ml.mace.structures`):

* JARVIS JSON (`lattice_mat` / `coords` / `elements`)
* pymatgen Structure JSON
* CIF / VASP

Element gate: MACE-MPA-0 coverage (Z=1–83, 89–94). Failures return reason codes
(`MISSING_PATH`, `FILE_NOT_FOUND`, `PARSE_ERROR`, `ELEMENT_OUT_OF_COVERAGE`, …)
and are never counted as successful loads.

```bash
# Smoke a slice
python scripts/audit_mace_structures.py --limit 1000 --workers 4

# Full formation-energy export
python scripts/audit_mace_structures.py --target formation_energy_per_atom --workers 8
```

Outputs (local / gitignored under processed data):

```text
data/processed/mace_exports/structure_audit/<target>/
  audit.parquet
  summary.json
  SUMMARY.md
```

Reference E_f audit (export 4.1.0): **227,026 / 260,162** load OK (87.26%).
All failures are `MISSING_PATH` (OQMD structure hole). Every present file loaded.

### Phase 4.3 — structure training pack

Join the frozen training export with the structure audit into a slim MACE pack
(labels + splits + `structure_path` + `n_atoms`; no composition `frac_*` columns).

```bash
python scripts/export_mace_pack.py --target formation_energy_per_atom
# optional: mark huge cells for training exclusion (still kept in the table)
python scripts/export_mace_pack.py --max-atoms 200
```

```text
data/processed/mace_exports/
  DATASET_CARD.md
  export_manifest.json
  db_freeze.json
  targets/formation_energy_per_atom.parquet
  structure_audit/...
```

Load in code: `from ml.mace.pack import load_mace_pack_frame`.
Filter with `mace_train_ok` (respects `--max-atoms`). **Do not resplit.**

### Phase 4.4 — frozen pretrained eval (default quality path)

Runs **frozen** foundation MACE on the structure pack. Registers
`model_family: mace_pretrained`. Formation energies use **`elemental_bulk_v1`**
(not isolated-atom E0s).

```bash
# Harness check (legacy e0 vs bulk μ)
python scripts/diagnose_mace_harness.py --limit 200 --device cpu

# Frozen eval (CPU-first)
python scripts/eval_mace_pretrained.py --device cpu --limit-per-split 200
python scripts/eval_mace_pretrained.py --device cpu --splits val test
```

### Phase 4.5 — fine-tune (last resort only)

**Not part of the default plan.** Use only after `diagnose_mace_harness.py` is
clean and frozen MACE still fails the LightGBM promote gate. See
[`mace_strategy.md`](mace_strategy.md).

```bash
# Explicit last-resort smoke (warns under frozen_first)
python scripts/train_mace.py --device cpu \
  --max-train-rows 64 --epochs 3 --batch-size 1 \
  --eval-limit-per-split 32
```

### Phase 4.6 — predict + dual-family compare

Predict formation energy for local structures or pack `material_id`s using the
latest registered MACE run (`mace_pretrained` by default):

```bash
python scripts/predict_mace.py \
  --structure data/structures/computational/JVASP-10.json
python scripts/predict_mace.py --material-id MAT-243797   # default: mace_pretrained
python scripts/predict_mace.py --structure path.json --json
```

Side-by-side LightGBM vs MACE on the **structure-gated pack ∩ test predictions**
intersection (promote gate = Spearman; writes under
`data/processed/mace_exports/comparisons/`):

```bash
python scripts/compare_mace_lightgbm.py --target formation_energy_per_atom
python scripts/compare_mace_lightgbm.py --family mace_pretrained --limit 500
```

`LEADERBOARD.md` (regenerated on each `register_run`) includes a Dual-family
table when both LightGBM and a MACE family exist for the same target.

### Phase 4.7 — density + E_hull protocols

**Density** is geometry (not an energy head). Default protocol `geometry_v1`
uses pymatgen `Structure.density` (g/cm³). Optional `geometry_relax_v1` runs a
short MACE FIRE relax first. Ready val/test come from the density MACE pack;
experimental COD cells are scored via `holdout_cod_geometry`.

```bash
# Pack reuses the formation-energy structure audit where IDs overlap
python scripts/export_mace_pack.py --target density
python scripts/eval_mace_density.py --limit-per-split 200 --cod-limit 100
# Optional relax protocol (needs [mace])
python scripts/eval_mace_density.py --protocol geometry_relax_v1 \
  --limit-per-split 20 --no-cod-holdout --device cpu
```

**Energy above hull** uses named protocol `mace_pd_eform_v1` — **not** a direct
structure→label regressor. MACE predicts `elemental_e0_v1` formation energies
on a reference pool (structure-gated train split); a pymatgen `PhaseDiagram`
(with elemental refs at 0) yields E_hull for val/test.

```bash
python scripts/export_mace_pack.py --target energy_above_hull
# Smoke (CPU)
python scripts/eval_mace_hull.py --device cpu \
  --limit-reference 128 --limit-per-split 64
```

### Phase 4.8 — gap / Niche embedding heads

MACE is not a native band-gap model. Phase 4.8 freezes the foundation, extracts
per-atom descriptors (`get_descriptors`), pools them (`mean‖std` =
`mace_desc_meanstd_v1`), and trains a **LightGBM readout**. Register as
`mace_embedding_head`. **Promote only if** test MAE beats composition LightGBM
on the structure-gated pack ∩ prediction intersection.

```bash
python scripts/export_mace_pack.py --target band_gap
python scripts/train_mace_embedding_head.py --target band_gap \
  --limit-per-split 200 --device cpu

# Niche exemplar
python scripts/export_mace_pack.py --target bulk_modulus_vrh
python scripts/train_mace_embedding_head.py --target bulk_modulus_vrh \
  --limit-per-split 150 --device cpu

# Compare only
python scripts/compare_mace_lightgbm.py --target band_gap \
  --family mace_embedding_head
```

Embedding caches live under
`data/processed/mace_exports/features/<target>/<model_id>/`.

### Phase 4.9 — scale, robustness, runbook

Operator-facing production guide: [`docs/mace_runbook.md`](mace_runbook.md).

```bash
python scripts/mace_status.py
python scripts/run_mace_pipeline.py --preset smoke --dry-run
python scripts/run_mace_pipeline.py --preset full --dry-run
# --device auto follows configs/mace.yaml device.preference
```

Presets: `configs/mace.yaml` → `runs.smoke` / `runs.full`. Embedding extract
resumes by appending missing `material_id`s (omit `--overwrite`). Checkpoint
downloads retry (`foundation.checkpoint_download_retries`).

### Package layout

Code lives under **`ml.mace`** (`src/ml/mace/`), not top-level `mace`, so it does
not collide with the `mace-torch` import name.
