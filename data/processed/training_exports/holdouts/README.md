# Specialty hold-outs

These sets are **evaluation slices**. Default stratified splits still include these materials in train/val/test; for honest niche reporting, evaluate metrics on the hold-out material_ids (or restrict test metrics to `holdout_* == true`).

- **holdout_cod_geometry** (45,672): COD-primary materials with structure_path. Use for experimental geometry / density evaluation; not a DFT label hold-out.
- **holdout_agl_thermal** (5,571): AFLOW AGL thermal rows (train_ok_thermal_agl). Prefer as the only honest thermal κ / Debye / C_v evaluation set.
- **holdout_battery_electrode** (17,331): MP insertion-electrode hosts (train_ok_battery).
- **holdout_elastic_mechanical** (40,690): DFT/AEL elastic subset (train_ok_elastic) for B/G evaluation.
