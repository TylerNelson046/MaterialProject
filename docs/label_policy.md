# Training Label Policy

**Authority:** This document is the training-label contract for Materials Predictor.
**Catalog:** `data/processed/materials.duckdb` (`search_index` + `properties`).
**Snapshot reference:** 2026-10-05 (~460,679 materials in `search_index`).
**Related plan:** `DATASET_PERFECTION_PLAN.md` Step 0.1.

Nothing marked **Proxy** may be trained as ground truth. Prefer `properties.origin` /
`properties.method` (and, after Phase 1, `search_index` method columns / train flags)
over unlabeled `search_index` values when a column is mixed.

---

## Tiers

| Tier | Meaning | Allowed ML use |
|------|---------|----------------|
| **Ready** | Broad DFT / database labels suitable as primary supervised targets | Regression / classification targets after hard filters |
| **Niche** | Real DFT or curated experimental labels, smaller / specialized coverage | Supervised targets only on filtered subsets |
| **Derived** | Algebraic or threshold transforms of other labels | Features, derived heads, or consistency checks — **not** independent ground truth |
| **Proxy** | Element-table mixes, screening scores, heuristic packs | Ranking / screening only — **never** as DFT ground truth |
| **Flag / gate** | Provenance, exclusion, or availability markers | Filters, masks, split keys — not regression targets |

---

## Hard rules (all exports)

Authoritative detail + SQL/Python encoding: [`docs/exclusion_rules.md`](exclusion_rules.md)
and `src/database/exclusion_rules.py` (Step 0.3).

1. Exclude rows with `deprecated = true` (~8.8k; also treat linked-source OR as deprecated).
2. Quarantine / null physical outliers via `src/normalization/physical_ranges.py` (see Phase 1.3).
3. Never average property values across sources — priority coalesce only (`src/database/property_policy.py`).
4. Do not treat AFLOW (or COD) hull energies as comparable to MP / JARVIS / OQMD hulls.
5. Train thermal κ only from SI `thermal_conductivity` (AGL-backed after Step 1.2). Never treat `thermal_conductivity_proxy` as DFT ground truth.
6. Structure models must scope to `structure_path IS NOT NULL` (Step 2.1 decision: remaining OQMD hole is API-limited; do not block non-structure heads).

**Source coalesce priority (null-fill only):**  
MP → JARVIS → OQMD → AFLOW → COD → Alexandria → derived.

**Hull-comparable sources only:** MP, JARVIS, OQMD.

---

## Property → tier → ML use

Fill counts are non-null `search_index` rows at the reference snapshot (N ≈ 460,679).

### Ready (primary training targets)

| Property | SI fill (approx.) | Typical methods | Notes / allowed use |
|----------|-------------------|-----------------|---------------------|
| `energy_above_hull` | 371k | `mp-gga-pbe`, `jarvis-optb88vdw`, `oqmd-dft`, `alexandria-pbe-hull` | Core stability target; hull-comparable sources only |
| `formation_energy_per_atom` | 307k | `mp-gga-pbe`, `jarvis-optb88vdw`, `oqmd-dft`, `aflow-dft` | Core thermo target; Alexandria primary currently has **no** E_f |
| `band_gap` | 329k | MP / JARVIS / OQMD / AFLOW DFT | PBE-biased; treat as DFT-PBE label, not experimental gap |
| `density` | 461k | Multi-source DFT + COD cell-mass | Geometry / packing target |

### Niche (train only on method-filtered subsets)

| Property | SI fill (approx.) | Typical methods | Notes / allowed use |
|----------|-------------------|-----------------|---------------------|
| `bulk_modulus_vrh` | 43k | `jarvis-optb88vdw`, `mp-gga-pbe`, `aflow-ael` | Elastic B target |
| `shear_modulus_vrh` | 41k | same | Elastic G target |
| `homogeneous_poisson` | 35k | elastic DFT / AEL | Niche mechanical |
| `universal_anisotropy` | 13k | elastic DFT | Niche mechanical |
| `debye_temperature` | 5.6k | `aflow-agl` | AGL thermal only |
| `heat_capacity_cv` | 5.6k | `aflow-agl` | AGL thermal only |
| `thermal_conductivity` | 5.6k | `aflow-agl` | Niche AGL κ (SI column is AGL-only after Step 1.2) |
| `average_voltage`, `capacity_grav`, `capacity_vol`, `energy_grav`, `energy_vol`, `working_ion` | 17k | `mp-insertion-electrode` | Battery / electrode hosts only |
| `mbj_bandgap` | 22k | `jarvis-mbj` | Better gap estimate; small JARVIS slice |
| `total_magnetization` (+ related mag fields) | ~203k / 154k | MP / JARVIS DFT | Niche magnetism |
| `dielectric_*`, `refractive_index` | 7–10k | MP / JARVIS | Sparse optical/dielectric |
| `slme`, `tc_supercon`, `exfoliation_energy`, effective masses | 0.8–18k | JARVIS specialty | Specialty heads only |
| `weighted_surface_energy`, `weighted_work_function` | ~133 | MP surfaces | Extremely sparse — optional niche |

### Derived (not independent ground truth)

| Property | Method tag | Notes / allowed use |
|----------|------------|---------------------|
| `youngs_modulus_vrh` | `elastic-vrh-derived` | From B/G — feature or consistency check |
| `specific_stiffness`, `specific_bulk_modulus` | `cheap-kpi-v1` | B or E / density |
| `thermal_to_density_ratio` | `cheap-kpi-v1` | Inherits SI κ (mostly proxy) — **do not** treat as DFT thermal KPI |
| `melting_to_density_ratio` | `cheap-kpi-v1` | From melting-point mix |
| `is_near_hull` | `cheap-kpi-v1` | Threshold on E_hull (≤ 0.05 eV/atom) |
| `is_semiconductor_window` | `cheap-kpi-v1` | Threshold on band gap |
| `elastic_anisotropy_abs` | `cheap-kpi-v1` | Abs of universal anisotropy |
| `neutron_mean_free_path`, `macroscopic_absorption_per_mass` | `cheap-kpi-v1` | From nuclear proxies |
| `energy_per_atom`, `uncorrected_energy_per_atom`, `equilibrium_reaction_energy_per_atom` | source DFT | Related thermo fields — not primary Ready targets unless explicitly scoped |
| `is_stable`, `is_metal`, `is_gap_direct`, `is_magnetic` | source / derived flags | Classification features or weak labels, not continuous DFT targets |

### Proxy (screening only — never DFT ground truth)

| Property | Method tag | Notes |
|----------|------------|-------|
| `thermal_conductivity_proxy` | `element-table-kappa` | Element-mix screening κ (~461k); never DFT GT |
| `melting_point_element_mix` | `element-table-nist` | Not experimental T_m |
| `neutron_absorption_cross_section` | `element-table-nist` | Element-table σ_a mix |
| `macroscopic_absorption`, `has_strong_neutron_absorber`, `radiation_hardness_score` | `nuclear-screening-v1` | Screening scores |
| `corrosion_resistance_score`, `oxidation_tendency` | `corrosion-proxy-v1` | Heuristic corrosion pack |
| `displacement_energy_mix`, `radiation_damage_susceptibility` | `radiation-damage-proxy-v1` | Not measured dpa / damage |

### Flags / gates

| Column | Role |
|--------|------|
| `deprecated` | Hard exclude when true |
| `theoretical` | Provenance (COD experimental vs computed) |
| `structure_path` | Gate for structure / graph models |
| `has_phonon` | Availability gate (not a label) |
| `source`, `source_id`, `n_sources` | Provenance / coalescence context |
| `space_group_*`, `nsites`, `volume`, `elements`, `nelements`, `chemsys` | Features / split strata — geometry may be COD experimental |
| Future: `train_ok_*` | Planned Phase 1.4 gates |
| `exclude_deprecated`, `train_ok_thermo`, `train_ok_gap`, `train_ok_elastic`, `train_ok_thermal_agl`, `train_ok_battery` | Train gates on SI (Step 1.4) |
| `label_conflict`, `conflict_band_gap`, `conflict_formation_energy`, `conflict_energy_above_hull`, `conflict_density` | Cross-source disagreement beyond tolerance (Step 3.3) |
| `thermal_conductivity_method`, `thermal_conductivity_source` | κ provenance on SI main column (AGL) |
| `thermal_conductivity_proxy` | Proxy κ column (Step 1.2) |

---

## Critical mixed column: thermal conductivity

| Column | Method | Count (approx.) | Tier |
|--------|--------|-----------------|------|
| `thermal_conductivity` | `aflow-agl` | 5,578 | **Niche** (train-safe) |
| `thermal_conductivity_proxy` | `element-table-kappa` | 460,659 | **Proxy** |

After Step 1.2 the main SI column is AGL-only — no method filter required for κ heads
beyond `thermal_conductivity IS NOT NULL` (method/source still stamped as `aflow-agl` / `aflow`).

Debye and C_v remain AGL-only and are safe Niche targets as-is.
`thermal_to_density_ratio` uses `COALESCE(AGL, proxy)` for screening fill — **not** a DFT thermal feature.

---

## Recommended training heads (pre-train)

| Head | Tier | Minimum row gate (conceptual) |
|------|------|-------------------------------|
| E_hull | Ready | value present, not deprecated, physical QA, hull-comparable provenance |
| E_f | Ready | value present, not deprecated, physical QA |
| Band gap | Ready | value present, not deprecated, physical QA |
| Density | Ready | value present, not deprecated, physical QA |
| Elastic B / G | Niche | DFT/AEL method present |
| Battery pack | Niche | MP electrode fields present |
| AGL Debye / C_v / κ | Niche | `aflow-agl` only |

Concrete boolean columns (`train_ok_thermo`, etc.) are on `search_index`
(Step 1.4; refresh with `scripts/refresh_train_flags.py`).

---

## Known biases and out-of-scope labels

**Biases to document in every export / dataset card:**

- Band gaps are predominantly GGA/PBE (underestimate experimental gaps).
- Hull energies are source-specific; cross-source disagreement is expected (Step 3.3).
- Density includes experimental COD geometries mixed with DFT cells.
- SI thermal κ is dominated by proxy until dual-column / method tags exist.

**Out of scope as ground truth (do not block training):**  
experimental melting points, measured radiation damage / dpa, ion-migration barriers,
full-catalog dielectrics / surfaces / T_c, 100% elastic or thermal DFT coverage.

---

## How to use this policy in code

1. Pick a head from Ready or Niche only.
2. Apply hard rules (deprecated, physical ranges, method filters).
3. Prefer `properties.method` / future SI method columns over unlabeled SI floats for mixed fields.
4. Export features + label + `material_id` + source/method + structure path + train flags (Phase 4).

Policy changes require an update to this file **and** a progress-log note in
`DATASET_PERFECTION_PLAN.md`.
