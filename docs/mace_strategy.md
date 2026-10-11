# MACE strategy — frozen-first (binding correction)

**Status:** Phase 4 frozen-first complete (2026-10-11). ALIGNN gap/moduli wiring deferred.  
**Supersedes:** any implication that fine-tuning is the default path to quality.

## Goal

High-quality structure-aware predictions **without training by default**.
Pretrained / frozen models first; training is a **last resort** after the
evaluation harness is verified and a frozen model still fails the promote gate.

## Division of labor

| Role | Model | Training? |
|------|--------|-----------|
| Formation energy + stability/hull inputs | **Frozen MACE-MPA-0** | No (default) |
| Composition screening + Niche heads | **LightGBM** | Already done (Phase 3) |
| Band gap, moduli, dielectric (when needed) | **Frozen ALIGNN** pretrained heads | Download-and-predict; no training |
| Fine-tune MACE | Only if frozen MACE fails promote **after** harness OK | Last resort |

LightGBM remains the **bar to beat**. Dual-family registry + promote gates stay.

## Harness (critical)

Export `formation_energy_per_atom` labels are DFT E_form vs **elemental standard
states**. Comparing them to MACE energies converted with checkpoint
**isolated-atom** E0s (`elemental_e0_v1`) produced a ~3 eV/atom systematic bias
(smoke MAE ≈ 3.3). That was a **broken harness**, not a verdict on MACE-MPA-0.

**Default protocol:** `elemental_bulk_v1` — frozen MACE μ on unary pack /
ASE elemental structures.

```bash
python scripts/diagnose_mace_harness.py --limit 200 --device cpu
# Expect: elemental_bulk_v1 MAE ≪ elemental_e0_v1; Spearman high
```

Promote bar for E_f (ranking-first): beat LightGBM **Spearman** on the
**structure-gated test intersection** using `mace_pretrained` +
`elemental_bulk_v1`. MAE is still reported; absolute MAE is secondary because
the Ready loop is score → relax → E_form → hull ranking.

```bash
python scripts/compare_mace_lightgbm.py --family mace_pretrained
# promote.metric: spearman (configs/mace.yaml)
```

## Candidate scoring path

```bash
# Resumable frozen E_form cache (hull reference pool)
python scripts/cache_mace_eform.py --device cpu --limit-per-split 64

# Score candidates: FIRE relax → E_form → E_hull
python scripts/score_mace_structures.py --structure path/to/POSCAR
python scripts/score_mace_structures.py --material-id MAT-1 --no-relax --no-hull
```

## Default plan (CPU-first, 8GB Mac)

1. Keep MACE-MPA-0 **frozen**.
2. Diagnose / keep `elemental_bulk_v1` as the only promote path for E_form.
3. Run frozen eval + hull + density geometry; promote on Spearman (MAE reported).
4. Build E_form cache; use score CLI for candidate ranking.
5. Wire frozen ALIGNN for gap/moduli when those heads are needed (no training) — **deferred**.
6. **Do not** schedule GPU rental or fine-tune until (2)–(3) fail fairly.

Fine-tune CLI (`train_mace.py`) remains available but warns under
`energy.frozen_first: true` and is **out of** `run_mace_pipeline` defaults.

## Local evidence (not committed)

Recorded on CPU with `elemental_bulk_v1` (2026-10-11; refresh after larger evals):

| Check | Result |
|-------|--------|
| Harness diagnose (n=200) | bulk MAE ≈ 0.267, Spearman ≈ 0.957; legacy e0 MAE ≈ 3.33 |
| Frozen eval (200/split) | test E_form MAE ≈ 0.267, Spearman ≈ 0.957 |
| Promote (n=200 overlap) | **Spearman winner: MACE** (0.957 vs LGBM 0.956); MAE still LGBM (0.15 vs 0.27) |
| Score CLI | `score_mace_structures` → E_form + optional E_hull OK |

Artifacts under `data/models/` and `data/processed/mace_exports/` are gitignored.

## Constraints

- Local, CPU-first (8GB Mac).
- No GPU burst planned until frozen promote fails after harness verification.
- Do not claim experimental gap accuracy from PBE labels.
- ASL foundation models stay out of the default path.
