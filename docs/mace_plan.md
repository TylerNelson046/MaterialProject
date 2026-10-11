# MACE implementation plan (Phase 4+)

**Status:** Phase **4.0** contracts frozen; Phases **4.1–4.9** done (2026-10-11).  
**Strategy (binding):** **Frozen-first** — see [`docs/mace_strategy.md`](mace_strategy.md).
Fine-tune is last resort after harness-verified promote failure.  
**Authority:** Binding decisions for structure / MACE work. Implementation phases
must follow this file; changing a pinned item requires an explicit revision here.

**Related:** [`docs/mace_strategy.md`](mace_strategy.md) (frozen-first),
[`docs/model_training.md`](model_training.md) (Phase 3 LightGBM),
[`docs/mace_training.md`](mace_training.md) (operator howto),
[`docs/mace_runbook.md`](mace_runbook.md),
[`docs/label_policy.md`](label_policy.md), [`docs/exclusion_rules.md`](exclusion_rules.md),
[`configs/mace.yaml`](../configs/mace.yaml).

---

## Why MACE (and what it is not)

MACE is an equivariant message-passing **interatomic potential**: structure →
**energy / forces / stress**. Under **frozen-first**, default use is the pinned
**MACE-MPA-0 checkpoint with no fine-tuning** for formation energy and as the
energy input to the hull protocol.

It is **not** a native band-gap or Niche multi-property model. Those heads stay
on LightGBM and/or **frozen ALIGNN** pretrained predictors (download-and-predict),
not MACE fine-tunes. Promote only by beating LightGBM on the structure-gated
intersection.

---

## Phase map

| Phase | Focus | Status |
|-------|--------|--------|
| **4.0** | Scope & contracts (this document) | **Done** |
| **4.1** | Optional `[mace]` deps + smoke energy | **Done** |
| **4.2** | Structure I/O (`structure_path` → ASE) + audit | **Done** |
| **4.3** | Structure training pack (`mace_exports`) | **Done** |
| **4.4** | Zero-shot pretrained eval + registry | **Done** |
| **4.5** | Energy-only fine-tune on export labels | **Done** |
| **4.6** | Predict CLI + dual-family leaderboard | **Done** |
| **4.7** | Density / E_hull protocols | **Done** |
| **4.8** | Gap & Niche via embeddings/readouts | **Done** |
| **4.9** | Scale, robustness, runbook | **Done** |

---

## 4.0 Pinned contracts

### C1 — Software stack

| Item | Pin |
|------|-----|
| Python | `>=3.11` (same as project) |
| Core libs | `mace-torch`, `torch`, `ase` |
| Structure bridge | existing `pymatgen` (JSON/CIF → ASE) |
| Install | **optional** extra: `pip install -e ".[mace]"` — must not break LightGBM-only installs |
| Device | `cuda` preferred for train/eval; `cpu` required for smoke tests; `mps` allowed if usable but not required |

Pinned dependency **version floors** (Phase 4.1): see `pyproject.toml` optional
extra `[mace]` (`torch>=2.2`, `ase>=3.22`, `mace-torch>=0.3.10`) and
[`docs/mace_training.md`](mace_training.md).

### C2 — Foundation checkpoint

| Item | Pin |
|------|-----|
| Family | **MACE-MPA-0** |
| Loader id | `medium-mpa-0` via `mace.calculators.mace_mp(model="medium-mpa-0")` |
| Why | Materials-oriented (MPTrj + sAlex), MIT license (matches this repo), recommended default in `mace-torch>=0.3.10` |
| Elements | 89 (foundation coverage); materials outside coverage are **load failures**, not silent drops in metrics |
| Dispersion | `dispersion=false` by default (enable only as an explicit ablation) |

**Allowed ablations** (must set `foundation_model` explicitly in run config; never
rely on library default which can change across `mace-torch` versions):

| Ablation id | Use |
|-------------|-----|
| `medium-mpa-0` | **Default** zero-shot + fine-tune init |
| `medium` | Classic MACE-MP-0a medium (repro / paper compare) |
| `medium-omat-0` / MATPES / MH | **Out of default path** (ASL or different theory); only with a documented experiment note |

Do **not** call bare `mace_mp()` without `model=...` in project code.

### C3 — Primary target and label units

| Item | Pin |
|------|-----|
| Primary supervised target | `formation_energy_per_atom` (Ready tier) |
| Label source | Frozen export `data/processed/training_exports/targets/formation_energy_per_atom.parquet` |
| Label column | `label` |
| Physical unit | **eV/atom** (same as [`docs/exclusion_rules.md`](exclusion_rules.md) ranges) |
| Freeze | Export **4.1.0** / DuckDB freeze SHA in `DATASET_CARD.md` — pin experiments to that SHA |

**Energy semantics:**

1. Raw MACE / ASE `get_potential_energy()` → **total energy in eV** for the cell.
2. Per-atom potential energy → `E_tot / n_atoms` (eV/atom). This is **not**
   formation energy until reference chemical potentials are applied.
3. **Formation energy for metrics vs export `label`:**
   - **Default (frozen):** convert MACE `E_tot` with protocol
     **`elemental_bulk_v1`** (standard-state elemental μ from frozen MACE on
     unary/ASE elemental structures). Report MAE/RMSE/R² vs export `label`.
   - **Legacy (do not promote on):** `elemental_e0_v1` isolated-atom checkpoint
     E0s — known ~3 eV/atom systematic bias vs DFT labels (harness bug).
   - **Fine-tune (last resort only):** energy-only FT after frozen promote fails
     with a verified harness; same reported metric space (export `label` eV/atom).
4. Elemental μ tables must be stored in the run directory
   (`reference_energies.json`) and named in `configs/mace.yaml` under
   `energy.reference_protocol` (default `elemental_bulk_v1`).

**Out of scope for default path:** force/stress labels; scheduled GPU fine-tunes.

### C4 — Splits and leakage

| Item | Pin |
|------|-----|
| Split column | `split` ∈ {`train`, `val`, `test`} from the training export |
| Recipe | 80 / 10 / 10, seed `42`, fingerprint-group leak safety |
| Resplit | **Forbidden** — do not draw new random splits for MACE |
| Compare to LightGBM | Prefer metrics on the **structure-present intersection** of the same split; also report full LightGBM test for context |

### C5 — Structure gate and I/O

| Item | Pin |
|------|-----|
| Hard gate | `EX-STRUCT`: `structure_path IS NOT NULL` ([`docs/exclusion_rules.md`](exclusion_rules.md)) |
| Ready conflicts | Keep export conflict exclusions already applied in target parquets |
| Path root | Paths are repo-relative (e.g. `data/structures/computational/...`) |
| Loader | pymatgen → ASE `Atoms`; support project JSON (lattice + coords + elements) and CIF |
| Valid structure | Finite cell, ≥1 site, finite positions, species in foundation element set |
| Failures | Quarantine with reason codes in an audit table; **never** count failed loads as successful predictions |
| Huge cells | Config `max_atoms` may filter **training** rows; evaluation sets must document any filter |

### C6 — Package, config, and artifact layout

| Item | Pin |
|------|-----|
| Config | `configs/mace.yaml` |
| Code package | `src/ml/mace/` (import `ml.mace`) — **not** top-level `mace` (collides with `mace-torch`) |
| CLIs | `scripts/smoke_mace.py` (4.1); `scripts/audit_mace_structures.py` (4.2); `scripts/export_mace_pack.py` (4.3); `scripts/eval_mace_pretrained.py` (4.4); `scripts/train_mace.py` (4.5); `scripts/predict_mace.py` + `scripts/compare_mace_lightgbm.py` (4.6); `scripts/eval_mace_density.py` + `scripts/eval_mace_hull.py` (4.7); `scripts/extract_mace_embeddings.py` + `scripts/train_mace_embedding_head.py` (4.8); `scripts/mace_status.py` + `scripts/run_mace_pipeline.py` (4.9) |
| Structure pack | `data/processed/mace_exports/` (local by default; gitignored with processed data until a freeze card is explicitly published) |
| Model artifacts | `data/models/mace/<target>/<run_id>/` (gitignored like other weights) |
| Registry | Shared `data/models/registry.json` with required field `model_family` |

**`model_family` values:**

| Value | Meaning |
|-------|---------|
| `lightgbm` | Phase 3 composition models |
| `mace_pretrained` | Zero-shot foundation (4.4) |
| `mace_finetuned` | Fine-tuned on our export (4.5+) |
| `mace_embedding_head` | Frozen/finetuned backbone + property head (4.8) |

**Per-run directory (minimum):**

```text
data/models/mace/<target>/<run_id>/
  config_snapshot.yaml
  metrics.json
  run_record.json
  reference_energies.json    # if used
  checkpoint/                # fine-tune only
  predictions_{val,test}.parquet   # optional
```

Leaderboard (`LEADERBOARD.md`) lists one row per `(target, model_family)` and a
**Dual-family** section (LightGBM vs best MACE) when both exist.

### C7 — Target roadmap (what “full MACE” means here)

| Target | Contract |
|--------|----------|
| `formation_energy_per_atom` | **Primary** — **frozen** MACE-MPA-0 + `elemental_bulk_v1`; promote if **Spearman** beats LightGBM on structure-gated test (MAE reported). Score CLI: relax → E_form → hull. Fine-tune only after harness-verified failure |
| `energy_above_hull` | Protocol **`mace_pd_eform_v1`** from frozen MACE E_form (`elemental_bulk_v1`); not a naive structure→label regressor |
| `density` | **`geometry_v1`** (no training); optional `geometry_relax_v1`; Ready + `holdout_cod_geometry` |
| `band_gap` + moduli / dielectric | Prefer **frozen ALIGNN** download-and-predict (no training); LightGBM remains composition bar. MACE embedding heads optional ablation only |
| Niche (battery, thermal, …) | **LightGBM** composition defaults |
| Proxy labels | Never |

### C8 — Quality bars and non-goals

**Acceptance bars**

1. 4.1: smoke total energy on one local structure (CPU).
2. 4.2: audit reports load success / failure reasons for E_f pack.
3. 4.4: registered `mace_pretrained` metrics on val/test.
4. 4.5: fine-tune path exists but is **not** the quality plan (last resort).
5. 4.6: predict CLI + dual-family leaderboard.
6. 4.7: density `geometry_v1` (+ COD holdout) and E_hull `mace_pd_eform_v1` registered.
7. 4.8: optional embedding ablations; default gap/moduli → frozen ALIGNN (**deferred**).
8. 4.9: operator runbook + presets + status/pipeline CLIs.
9. **Frozen-first correction:** `elemental_bulk_v1` harness; promote gates target
   `mace_pretrained` first; no GPU FT until frozen promote fails after diagnosis.
10. Ranking-first promote (Spearman); score/cache CLIs; phase status = frozen-first complete.

**Non-goals until explicitly reopened**

- Default-path MACE fine-tuning / GPU rental
- Force/stress training without labels
- Claiming experimental gap accuracy from PBE labels
- Committing weights, raw structures, or DuckDB
- Using ASL-licensed foundation models as the default path

### C9 — Documentation and citation

- Operator docs live in `docs/mace_training.md` starting Phase 4.1 (howto);
  this file remains the **contract**.
- README links here for structure-model plans.
- Runs that use MACE-MPA-0 should cite the MACE foundation model paper
  (Batatia et al., arXiv:2401.00096) in docs; code comments need not repeat it.

---

## Revision log

| Date | Change |
|------|--------|
| 2026-10-11 | Phase 4.0: initial frozen contracts (C1–C9). |
| 2026-10-11 | Phase 4.1: `[mace]` extra, `ml.mace` package, device helper, smoke CLI/tests. C6 package path revised to `src/ml/mace/` to avoid `mace-torch` import collision. Checkpoints cached under `data/models/mace/checkpoints/` via `requests`. Default dtype set to `float32` for smoke/eval stability. |
| 2026-10-11 | Phase 4.2: `ml.mace.structures` + `audit` + `scripts/audit_mace_structures.py`; reason-coded quarantine; MACE-MPA-0 element set. MP JSON→ASE strips null site props (AseAtomsAdaptor `~None` crash). E_f audit: 227,026/260,162 OK; all failures `MISSING_PATH` (OQMD hole). |
| 2026-10-11 | Phase 4.3: `ml.mace.pack` + `scripts/export_mace_pack.py`; pack version 4.3.0 under `data/processed/mace_exports/`. |
| 2026-10-11 | Phase 4.4: zero-shot eval (`elemental_e0_v1`), registry `model_family=mace_pretrained`, `scripts/eval_mace_pretrained.py`. Leaderboard rows are per `(target, model_family)`. Quick run (200/split): test E_form Spearman≈0.52, MAE≈3.3 eV/atom (absolute scale weak; correlation gates stack). |
| 2026-10-11 | Phase 4.5: energy-only fine-tune via `mace_run_train` + pack→extxyz targets (`E_tot = label·n + ΣE0`); `scripts/train_mace.py`; registry `mace_finetuned`. Smoke (32 train / 2 epochs): test E_form MAE≈0.82 vs pretrained≈3.3 (still above LightGBM≈0.16; needs larger/GPU FT). |
| 2026-10-11 | Phase 4.6: `scripts/predict_mace.py` + `scripts/compare_mace_lightgbm.py`; dual-family `LEADERBOARD.md` section (ΔMAE = MACE − LightGBM). |
| 2026-10-11 | Phase 4.7: density protocols `geometry_v1` / `geometry_relax_v1` (`scripts/eval_mace_density.py`, COD holdout); hull protocol `mace_pd_eform_v1` (`scripts/eval_mace_hull.py`); packs may `--reuse-audit-from formation_energy_per_atom`. |
| 2026-10-11 | Phase 4.8: `mace_desc_meanstd_v1` embeddings + LightGBM readout (`scripts/extract_mace_embeddings.py`, `scripts/train_mace_embedding_head.py`); registry `mace_embedding_head`; promotion via structure-gated compare. |
| 2026-10-11 | Phase 4.9: `docs/mace_runbook.md`; `runs.smoke`/`runs.full` in `configs/mace.yaml`; `scripts/mace_status.py` + `scripts/run_mace_pipeline.py`; checkpoint download retries; embedding incremental resume; `--device auto`; FT dtype coerce. |
| 2026-10-11 | **Frozen-first correction:** default `elemental_bulk_v1` (fixes ~3 eV/atom isolated-E0 harness bug); `docs/mace_strategy.md`; FT out of pipeline defaults; promote prefers `mace_pretrained`; ALIGNN frozen for gap/moduli (planned); diagnose script. |
| 2026-10-11 | Ranking-first promote (Spearman); `score_mace_structures` + `cache_mace_eform`; hull prefers E_form cache. |
| 2026-10-11 | Phase polish: README/runbook/strategy aligned; leaderboard Spearman; predict default `mace_pretrained`; ALIGNN deferred. |
