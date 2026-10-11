# Materials Predictor — Training Dataset Card

**Export version:** 4.1.0  
**Generated (UTC):** 2026-10-11T16:07:41Z  
**Exclusion rules:** 0.3.1  

## Catalog freeze

- Live source DB: `data/processed/materials.duckdb`
- Export freeze copy: `data/processed/materials.duckdb.pretrain-export-20261011`
- SHA-256: `039313a7153e598e037ee7fc938ff2f5c82d4c719af1155a62ca8ab00b88682d`
- Size: 2,268,868,608 bytes
- Frozen at: 2026-10-11T16:05:52Z
- Materials in `search_index`: **460,679**
- Deprecated excluded from supervised gates: **8,834**

## Quality tiers (see `docs/label_policy.md`)

- **Ready:** E_hull, E_f, band gap, density
- **Niche:** elastic B/G, AGL thermal, battery electrodes
- **Proxy / derived:** never as DFT ground truth (element-mix κ, nuclear scores, …)

## Known biases

- Band gaps are predominantly GGA/PBE (underestimate experiment).
- Cross-source hull / E_f / gap / density disagreements are flagged (`conflict_*`); Ready targets **exclude** conflict rows (`EXCLUDE_LABEL_CONFLICTS_V1=True`).
- Density mixes DFT cells and COD experimental geometries.
- Thermal κ on SI is AGL-only; proxy κ is in `thermal_conductivity_proxy`.
- Structure models must use `structure_path IS NOT NULL` (OQMD hole scoped).

## Composition features

- `118` elemental fraction columns (`frac_H` … `frac_Og`, Z=1–118).
- Stats: `comp_mean_atomic_number`, `comp_mean_atomic_mass`, `comp_max_atomic_number`, `comp_n_elements`.
- Derived from `formula_pretty` via pymatgen Composition (no matminer).

## Split recipe

- Seed: `42`
- Fractions: train 80% / val 10% / test 10%
- Leak safety: materials sharing a structure fingerprint (composition + SG + nsites) share one `group_id` and one split.
- Stratification bins: `nelements` (1/2/3/4+) × primary `source` × chemsys family.

### Global split counts (all materials)

| Split | N |
|-------|--:|
| test | 48,994 |
| train | 366,882 |
| val | 44,803 |

## Per-target exports

| Target | Tier | Rows | Excl. conflict | Train | Val | Test |
|--------|------|-----:|---------------:|------:|----:|-----:|
| `energy_above_hull` | Ready | 253,247 | 40,171 | 202,143 | 24,738 | 26,366 |
| `formation_energy_per_atom` | Ready | 260,162 | 33,256 | 207,637 | 25,419 | 27,106 |
| `band_gap` | Ready | 315,866 | 13,097 | 251,744 | 30,734 | 33,388 |
| `density` | Ready | 425,900 | 25,781 | 339,235 | 41,439 | 45,226 |
| `bulk_modulus_vrh` | Niche | 40,690 | 0 | 32,180 | 3,983 | 4,527 |
| `shear_modulus_vrh` | Niche | 40,690 | 0 | 32,180 | 3,983 | 4,527 |
| `thermal_conductivity_agl` | Niche | 5,571 | 0 | 4,400 | 537 | 634 |
| `debye_temperature` | Niche | 5,571 | 0 | 4,400 | 537 | 634 |
| `heat_capacity_cv` | Niche | 5,571 | 0 | 4,400 | 537 | 634 |
| `battery_capacity_grav` | Niche | 17,331 | 0 | 13,884 | 1,666 | 1,781 |

## Specialty hold-outs

| Hold-out | N |
|----------|--:|
| `holdout_cod_geometry` | 45,672 |
| `holdout_agl_thermal` | 5,571 |
| `holdout_battery_electrode` | 17,331 |
| `holdout_elastic_mechanical` | 40,690 |

Details: `holdouts/README.md`.

## Files

```text
training_exports/
  DATASET_CARD.md
  export_manifest.json
  db_freeze.json
  splits.parquet
  holdouts/
  targets/*.parquet
```

## Training contract

1. Filter with the target’s `gate` / export table (already applied).
2. Ready thermo/gap/density exports already exclude `conflict_* == true` (targets: band_gap, density, energy_above_hull, formation_energy_per_atom).
3. Do not train on Proxy columns or `thermal_conductivity_proxy` as DFT κ.
4. Pin experiments to the freeze SHA-256 above.
5. Use `frac_*` + composition-stat columns as tabular features when needed.
   LightGBM training also adds elemental descriptors from `formula_pretty`
   (see `docs/model_training.md`).

