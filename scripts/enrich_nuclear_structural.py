"""Enrich DuckDB with nuclear structural screening props (items 1, 2, 4).

1. Radiation-hardness score + has_strong_neutron_absorber
2. Macroscopic absorption Σ_a (1/cm)
4. Young’s modulus VRH from existing B/G (and G+ν fallback)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.constants import PROPERTY_UNITS, DataSource, PropertyName  # noqa: E402
from config.settings import clear_settings_cache, get_settings  # noqa: E402
from data_packs.radiation_screening import (  # noqa: E402
    has_strong_neutron_absorber,
    macroscopic_absorption_cm,
    parse_composition,
    radiation_hardness_score,
    youngs_modulus_vrh,
)
from database.duckdb import MaterialsDatabase, ensure_search_index_columns  # noqa: E402
from database.write_lock import duckdb_write_lock  # noqa: E402
from normalization.physical_ranges import sanitize_elastic  # noqa: E402

SOURCE = DataSource.DERIVED.value
METHOD_RAD = "nuclear-screening-v1"
METHOD_E = "elastic-vrh-derived"
ORIGIN = "derived"

DERIVED_NAMES = (
    PropertyName.MACROSCOPIC_ABSORPTION.value,
    PropertyName.HAS_STRONG_NEUTRON_ABSORBER.value,
    PropertyName.RADIATION_HARDNESS_SCORE.value,
    PropertyName.YOUNGS_MODULUS_VRH.value,
)


def _double(
    *,
    material_id: str,
    source_id: str,
    name: PropertyName,
    value: float,
    method: str,
) -> dict[str, Any]:
    return {
        "source": SOURCE,
        "source_id": source_id,
        "material_id": material_id,
        "name": name.value,
        "value_double": float(value),
        "value_bool": None,
        "value_text": None,
        "unit": PROPERTY_UNITS[name],
        "origin": ORIGIN,
        "method": method,
        "uncertainty": None,
    }


def _bool(
    *,
    material_id: str,
    source_id: str,
    name: PropertyName,
    value: bool,
    method: str,
) -> dict[str, Any]:
    return {
        "source": SOURCE,
        "source_id": source_id,
        "material_id": material_id,
        "name": name.value,
        "value_double": None,
        "value_bool": bool(value),
        "value_text": None,
        "unit": PROPERTY_UNITS[name],
        "origin": ORIGIN,
        "method": method,
        "uncertainty": None,
    }


def main() -> int:
    clear_settings_cache()
    settings = get_settings()

    with duckdb_write_lock(settings, label="nuclear-structural-1-2-4"):
        db = MaterialsDatabase(settings.duckdb_path)
        with db.connect() as conn:
            ensure_search_index_columns(conn)
            rows = conn.execute(
                """
                SELECT
                    si.material_id,
                    s.composition,
                    si.density,
                    si.neutron_absorption_cross_section,
                    si.melting_point_element_mix,
                    si.bulk_modulus_vrh,
                    si.shear_modulus_vrh,
                    si.homogeneous_poisson
                FROM search_index AS si
                JOIN source_records AS s
                  ON s.source = si.source AND s.source_id = si.source_id
                """
            ).fetchall()

            prop_rows: list[dict[str, Any]] = []
            si_rows: list[dict[str, Any]] = []
            n_sigma = n_hard = n_abs = n_e = 0

            for (
                material_id,
                composition_raw,
                density,
                sigma,
                t_m,
                bulk,
                shear,
                poisson,
            ) in rows:
                composition = parse_composition(composition_raw)
                derived_id = f"nuclear-struct-{material_id}"
                update: dict[str, Any] = {
                    "material_id": material_id,
                    "macroscopic_absorption": None,
                    "has_strong_neutron_absorber": None,
                    "radiation_hardness_score": None,
                    "youngs_modulus_vrh": None,
                }

                absorber = None
                if composition is not None:
                    absorber = has_strong_neutron_absorber(composition)
                    update["has_strong_neutron_absorber"] = absorber
                    prop_rows.append(
                        _bool(
                            material_id=material_id,
                            source_id=derived_id,
                            name=PropertyName.HAS_STRONG_NEUTRON_ABSORBER,
                            value=bool(absorber),
                            method=METHOD_RAD,
                        )
                    )
                    n_abs += 1

                    sigma_m = macroscopic_absorption_cm(
                        sigma_barns=float(sigma) if sigma is not None else None,
                        density_g_cm3=float(density) if density is not None else None,
                        composition=composition,
                    )
                    if sigma_m is not None:
                        update["macroscopic_absorption"] = sigma_m
                        prop_rows.append(
                            _double(
                                material_id=material_id,
                                source_id=derived_id,
                                name=PropertyName.MACROSCOPIC_ABSORPTION,
                                value=sigma_m,
                                method=METHOD_RAD,
                            )
                        )
                        n_sigma += 1

                score = radiation_hardness_score(
                    sigma_barns=float(sigma) if sigma is not None else None,
                    melting_point_k=float(t_m) if t_m is not None else None,
                    bulk_modulus_gpa=float(bulk) if bulk is not None else None,
                    density_g_cm3=float(density) if density is not None else None,
                    has_absorber=absorber,
                )
                if score is not None:
                    update["radiation_hardness_score"] = score
                    prop_rows.append(
                        _double(
                            material_id=material_id,
                            source_id=derived_id,
                            name=PropertyName.RADIATION_HARDNESS_SCORE,
                            value=score,
                            method=METHOD_RAD,
                        )
                    )
                    n_hard += 1

                e_mod = sanitize_elastic(
                    youngs_modulus_vrh(
                        float(bulk) if bulk is not None else None,
                        float(shear) if shear is not None else None,
                        float(poisson) if poisson is not None else None,
                    )
                )
                if e_mod is not None:
                    update["youngs_modulus_vrh"] = e_mod
                    prop_rows.append(
                        _double(
                            material_id=material_id,
                            source_id=derived_id,
                            name=PropertyName.YOUNGS_MODULUS_VRH,
                            value=e_mod,
                            method=METHOD_E,
                        )
                    )
                    n_e += 1

                if any(
                    update[k] is not None
                    for k in (
                        "macroscopic_absorption",
                        "has_strong_neutron_absorber",
                        "radiation_hardness_score",
                        "youngs_modulus_vrh",
                    )
                ):
                    si_rows.append(update)

            conn.execute(
                f"""
                DELETE FROM properties
                WHERE source = ?
                  AND name IN ({",".join("?" for _ in DERIVED_NAMES)})
                """,
                [SOURCE, *DERIVED_NAMES],
            )

            if prop_rows:
                frame = pd.DataFrame(prop_rows)
                conn.register("ns_props", frame)
                conn.execute("INSERT INTO properties SELECT * FROM ns_props")
                conn.unregister("ns_props")

            if si_rows:
                frame = pd.DataFrame(si_rows)
                conn.register("ns_si", frame)
                conn.execute(
                    """
                    UPDATE search_index AS si
                    SET
                        macroscopic_absorption = n.macroscopic_absorption,
                        has_strong_neutron_absorber = n.has_strong_neutron_absorber,
                        radiation_hardness_score = n.radiation_hardness_score,
                        youngs_modulus_vrh = COALESCE(n.youngs_modulus_vrh, si.youngs_modulus_vrh)
                    FROM ns_si AS n
                    WHERE si.material_id = n.material_id
                    """
                )
                conn.unregister("ns_si")

            print(
                "Nuclear structural enrich done: "
                f"Σ_a={n_sigma:,}, absorber_flag={n_abs:,}, "
                f"hardness={n_hard:,}, E_vrh={n_e:,}"
            )
            # Spot checks
            sample = conn.execute(
                """
                SELECT formula_pretty, neutron_absorption_cross_section,
                       macroscopic_absorption, radiation_hardness_score,
                       has_strong_neutron_absorber, youngs_modulus_vrh
                FROM search_index
                WHERE youngs_modulus_vrh IS NOT NULL
                ORDER BY radiation_hardness_score DESC NULLS LAST
                LIMIT 5
                """
            ).fetchall()
            print("Top hardness with E:", sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
