# MACE operator runbook (Phase 4.9)

Contracts: [`docs/mace_plan.md`](mace_plan.md).  
**Strategy:** [`docs/mace_strategy.md`](mace_strategy.md) (**frozen-first**).  
Howto by phase: [`docs/mace_training.md`](mace_training.md).  
Config: [`configs/mace.yaml`](../configs/mace.yaml).

This runbook is the **production** entrypoint: preflight → harness check →
frozen eval → promote gates. Fine-tune is **not** in the default DAG.

---

## Preflight

1. **Python 3.11–3.13** (avoid 3.14 for torch/MACE on macOS — native segfaults).
2. Fresh venv + optional extra:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,mace]"
# GPU: install the matching torch CUDA wheel from pytorch.org first, then the extra.
```

3. Frozen training export present: `data/processed/training_exports/` (4.1.0).
4. Disk: expect multi‑GB under `data/models/mace/` (checkpoints, FT workdirs,
   embedding caches) and `data/processed/mace_exports/`.
5. Status probe:

```bash
python scripts/mace_status.py
python scripts/smoke_mace.py --device cpu
```

---

## Smoke vs full

Named presets live in `configs/mace.yaml` → `runs.smoke` / `runs.full`.
Plan without executing:

```bash
python scripts/run_mace_pipeline.py --preset smoke --dry-run
python scripts/run_mace_pipeline.py --preset full --dry-run
```

| Knob | `smoke` | `full` |
|------|---------|--------|
| device | `cpu` | `auto` (cuda→mps→cpu) |
| audit limit | 1000 | all |
| eval / density / hull limits | 64–200 | all |
| FT train rows / epochs | 64 / 3 | all / 30 |
| embedding limit/split | 200 | all |
| pack targets | E_f, density, hull, band_gap | + bulk_modulus_vrh |

CLI flags on individual scripts always override presets.

---

## Pipeline DAG (frozen-first)

```text
diagnose_mace_harness  (elemental_bulk_v1 vs legacy e0)
        │
audit (formation_energy_per_atom)
        │
        ▼
export packs ──► (reuse_audit_from …)
        │
        ├─► eval_mace_pretrained   ← frozen MACE E_form (default)
        ├─► eval_mace_density      ← geometry_v1 (no train)
        ├─► cache_mace_eform       ← resumable E_form pool (hull + score)
        ├─► eval_mace_hull         ← frozen E_form → PhaseDiagram
        └─► compare_mace_lightgbm --family mace_pretrained
                    │
                    ▼
           promote if intersection Spearman > LightGBM (MAE reported)

Candidate scoring (outside DAG)::

    score_mace_structures  ← relax → E_form → E_hull

train_mace / embedding heads / ALIGNN wiring: only after frozen promote fails
```

Partial runs:

```bash
python scripts/run_mace_pipeline.py --preset smoke --stages pack density --dry-run
```

---

## Artifact map

| Path | Contents |
|------|----------|
| `data/processed/mace_exports/targets/*.parquet` | Structure-gated packs |
| `data/processed/mace_exports/targets/*.manifest.json` | Per-target pack manifests |
| `data/processed/mace_exports/targets_index.json` | Index of available packs |
| `data/processed/mace_exports/structure_audit/<target>/` | Load quarantine |
| `data/processed/mace_exports/features/<target>/<model>/` | Embedding + E_form caches (resumable) |
| `…/features/…/eform_elemental_bulk_v1.parquet` | Frozen E_form cache for hull/score |
| `data/models/mace/checkpoints/` | Foundation `.model` cache |
| `data/models/mace/<target>/<run_id>/` | Metrics, preds, checkpoints/heads |
| `data/models/registry.json` + `LEADERBOARD.md` | Shared with LightGBM |

**Do not commit** weights, structures, DuckDB, or local model runs.

---

## Device policy

- `--device cpu` — required for smoke / CI-like checks.
- `--device auto` — uses `configs/mace.yaml` → `device.preference` (`cuda`→`mps`→`cpu`).
- Explicit `cuda` / `mps` raises if unavailable.

Fine-tune / full embedding extract / full hull eval: **GPU strongly recommended**.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| “Python quit unexpectedly” | 3.14 + torch/MACE segfault | Use 3.12/3.13; ensure `configure_mace_runtime()` (scripts already do) |
| Checkpoint download fails | SSL / GitHub flake | Retries (`foundation.checkpoint_download_retries`); or place `mace-mpa-0-medium.model` in `data/models/mace/checkpoints/` |
| `mace_run_train` exit ≠ 0 | OOM, bad xyz, element filter | Read `train.log`; try smaller `--batch-size` / `--max-train-rows`; ensure `foundation_model_elements` path in train orchestration |
| Audit mostly `MISSING_PATH` | OQMD structure hole | Expected; packs exclude those rows |
| Embedding extract “resume” | Partial cache | Re-run without `--overwrite` to append missing `material_id`s |
| E_form MAE ~3 eV/atom | Legacy `elemental_e0_v1` harness | Use `elemental_bulk_v1`; run `diagnose_mace_harness.py` |
| Frozen MAE ~0.2–0.3 vs LGBM ~0.16 | Absolute residual; ranking may still win | Check Spearman on intersection compare; FT only if Spearman also loses |
| Embedding head not promoted | Spearman (or MAE) ≤ LightGBM | Prefer frozen ALIGNN for gap/moduli (deferred); keep LightGBM default |

---

## Post-run verification

1. `python scripts/mace_status.py` — packs, caches, latest metrics, promotion flags.
2. Open `data/models/LEADERBOARD.md` — Dual-family **Spearman** (promote) + MAE.
3. Per run: `run_record.json`, `metrics.json`, `predictions_test.parquet`.
4. Density: also check `holdout_cod_geometry` metrics when COD eval ran.
5. Embedding heads: ablation only; `promoted_vs_lightgbm` must be **true** before treating as default (ALIGNN preferred for gap/moduli when wired).

---

## Citation

Runs using MACE-MPA-0 should cite Batatia et al., arXiv:2401.00096.
