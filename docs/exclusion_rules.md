# Exclusion and Comparability Rules

**Authority:** Hard filters and cross-source rules for any training export or
train-mask build. Applies **before** label-tier selection in
[`docs/label_policy.md`](label_policy.md).

**Machine encoding:** `src/database/exclusion_rules.py`  
**Related code:** `src/database/property_policy.py`, `src/normalization/physical_ranges.py`  
**Plan step:** `DATASET_PERFECTION_PLAN.md` Step 0.3  
**Snapshot reference:** `pretrain-20261005` (N ≈ 460,679)

---

## 1. Row exclusions

| Rule ID | Condition | Action | Rationale |
|---------|-----------|--------|-----------|
| `EX-DEP` | `search_index.deprecated IS TRUE` | **Exclude** entire material from all supervised training heads | MP (or linked) deprecation; ER sets SI true if any linked source is deprecated (~8,834 rows at freeze) |
| `EX-DEP-NULL` | `deprecated IS NULL` or `FALSE` | **Keep** (subject to other gates) | NULL means “not marked deprecated,” not unknown-bad |
| `EX-STRUCT` | `structure_path IS NULL` | **Exclude** from structure / graph / geometry models only | ~56k OQMD-primary hole at freeze; non-structure heads may still train |
| `EX-PROXY-TARGET` | Label tier is **Proxy** (see label policy) | **Never** use as regression/classification ground truth | Element-mix κ, nuclear/corrosion/radiation scores, etc. |
| `EX-MIXED-KAPPA` | Using `thermal_conductivity_proxy` (or pre-split mix) as DFT κ | **Exclude** from thermal-κ heads | After Step 1.2, train on `thermal_conductivity` (AGL-only) only |

SQL base gate for material-level exports (all heads):

```sql
WHERE (deprecated IS NOT TRUE)   -- keeps FALSE and NULL; drops TRUE
```

Structure-model gate (additional):

```sql
AND structure_path IS NOT NULL
```

---

## 2. Physical outlier quarantine

| Rule ID | Condition | Action | Rationale |
|---------|-----------|--------|-----------|
| `EX-OOR` | Value outside `PROPERTY_RANGE_GATES` / `PROPERTY_ABS_GATES` in `physical_ranges.py` | **Null / drop** the value (do **not** clip) | Absurd density, moduli, gaps, thermal, etc. poison training; enforced by `scripts/sanitize_physical_outliers.py` |
| `EX-OOR-ROW` | After quarantine, required label for the head is NULL | Row fails that head’s train mask | Per-head, not global delete |

Critical gates (inclusive bounds):

| Property | Lo | Hi | Unit |
|----------|---:|---:|------|
| `density` | 0.01 | 30 | g/cm³ |
| `formation_energy_per_atom` | −10 | 5 | eV/atom |
| `energy_above_hull` | −1e−4 | 5 | eV/atom |
| `band_gap` / `mbj_bandgap` | 0 | 20 | eV |
| `bulk_modulus_vrh` / `shear_modulus_vrh` | 0 | 1000 | GPa |
| `homogeneous_poisson` | −1 | 0.5 | — |
| `volume` | 1e−3 | 1e5 | Å³ |

AGL thermal κ uses a separate ingest gate (~0.001–5000 W/m·K) in the AFLOW thermal downloader; treat OOR the same way (null, don’t clip).

Phase 1.3 owns the idempotent sanitize pass that enforces these on the live DB.
Exports must still re-check ranges when reading a snapshot.

---

## 3. Hull energy comparability

| Rule ID | Rule | Detail |
|---------|------|--------|
| `CMP-HULL` | Only **MP, JARVIS, OQMD** hull energies may enter `search_index.energy_above_hull` coalesce | Encoded as `HULL_COMPARABLE_SOURCES` in `property_policy.py` |
| `CMP-HULL-AFLOW` | **Never** coalesce or train-compare AFLOW “hull-like” fields as E_hull | AFLOW is not on the same hull reference; normalizers mark hull unavailable |
| `CMP-HULL-COD` | **Never** use COD for E_hull | Experimental geometry source; no DFT hull |
| `CMP-HULL-ALEX` | Alexandria hull (`alexandria-pbe-hull`) may appear on SI for Alexandria-primary rows | Do **not** silently merge Alexandria hull into MP/JARVIS/OQMD materials as if identical; treat as same *name*, different catalog bias — prefer primary-source value, no averaging |

When exporting an E_hull training table, record the contributing `source` / `method` and restrict to hull-comparable provenance where conflict is possible.

---

## 4. Cross-source value resolution (no averaging)

| Rule ID | Rule | Detail |
|---------|------|--------|
| `CMP-NO-AVG` | **Never** average numeric properties across sources into `search_index` | One winner only |
| `CMP-COALESCE` | Null-fill only, by priority | Primary source if non-null; else lowest `SOURCE_PRIORITY` among linked sources with a value |
| `CMP-PRIORITY` | Priority order | materials_project → jarvis → oqmd → aflow → cod → alexandria → derived |
| `CMP-KEEP-ROWS` | Every source keeps its own `properties` row forever | Detail / conflict views use per-source values |
| `CMP-CONFLICT` | Tracked properties may disagree | `CONFLICT_TRACKED_PROPERTIES`; SI flags from Step 3.3 (`label_conflict`, per-property) — do not collapse via mean |
| `EX-CONFLICT-V1` | Ready head has `conflict_<property> IS TRUE` | **Exclude** from that Ready export when `EXCLUDE_LABEL_CONFLICTS_V1` is on | Targets: E_hull, E_f, band_gap, density (`CONFLICT_EXCLUDE_TARGETS`); Niche heads keep conflict rows for optional down-weighting |

---

## 5. Per-head minimum gates (conceptual)

These are the exclusion/comparability overlays for Ready/Niche heads.
Concrete SI booleans (Step 1.4): `exclude_deprecated`, `train_ok_thermo`,
`train_ok_gap`, `train_ok_elastic`, `train_ok_thermal_agl`, `train_ok_battery`.

| Head | Must pass | SI flag |
|------|-----------|---------|
| Any supervised head | `EX-DEP` | `exclude_deprecated = false` |
| E_hull + E_f | Labels present + in-range | `train_ok_thermo` |
| Band gap | Label present + in-range | `train_ok_gap` |
| Elastic B/G | Labels present + in-range + DFT/AEL method | `train_ok_elastic` |
| AGL thermal (Debye / C_v / κ) | AGL provenance | `train_ok_thermal_agl` |
| Battery pack | MP electrode fields present | `train_ok_battery` |
| Structure models | `EX-STRUCT` | `structure_path IS NOT NULL` |

---

## 6. What these rules do *not* require

- Filling every physics property from open data
- Dropping materials that lack optional Niche labels
- Resolving (reconciling) all cross-source conflicts before first train — Ready
  exports **exclude** conflict rows under `EX-CONFLICT-V1`; Niche may still include them
- Waiting for 100% OQMD `structure_path` before non-structure heads

---

## 7. Implementation checklist for exporters

1. Pin DB to a frozen snapshot (e.g. `materials.duckdb.pretrain-export-*`) when reproducibility matters.
2. Apply `sql_base_exclusion()` from `exclusion_rules.py`.
3. Apply head-specific label presence + physical range + method filters.
4. For Ready thermo/gap/density: drop rows where the per-property `conflict_*` flag is true (`EX-CONFLICT-V1`).
5. Never invent labels by averaging linked sources.
6. Write `material_id`, source/method, exclusion version, and composition-feature metadata into the export.
