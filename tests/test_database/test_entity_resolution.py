"""Tests for entity-resolution fingerprints and linking."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from config.constants import DataSource, PropertyName
from database.duckdb import MaterialsDatabase
from database.entity_resolution import resolve_entities
from database.fingerprints import (
    reduced_composition_key,
    resolve_space_group_number,
    structure_fingerprint,
)
from normalization.schema import (
    CompositionInfo,
    MaterialRecord,
    SourceRecord,
    StructureRef,
    SymmetryInfo,
    make_material_id,
    make_source_property,
)


def test_reduced_composition_key_normalizes_supercell() -> None:
    key_a = reduced_composition_key(composition={"Ba": 2.0, "F": 8.0, "Mg": 2.0})
    key_b = reduced_composition_key(composition={"Ba": 1.0, "F": 4.0, "Mg": 1.0})
    assert key_a == key_b
    assert key_a == "Ba1F4Mg1"


def test_space_group_from_symbol() -> None:
    assert resolve_space_group_number(space_group_symbol="Fm-3m") == 225
    assert resolve_space_group_number(space_group_number=63) == 63


def test_fingerprint_matches_across_symbol_and_number() -> None:
    fp_mp = structure_fingerprint(
        formula_pretty="Si",
        space_group_symbol="Fd-3m",
        space_group_number=227,
        nsites=2,
    )
    fp_aflow = structure_fingerprint(
        formula_pretty="Si2",
        composition={"Si": 2.0},
        space_group_number=227,
        nsites=2,
    )
    assert fp_mp is not None
    assert fp_mp == fp_aflow


def test_fingerprint_rejects_different_nsites() -> None:
    a = structure_fingerprint(formula_pretty="FeOF", space_group_number=6, nsites=12)
    b = structure_fingerprint(formula_pretty="FeOF", space_group_number=6, nsites=18)
    assert a != b


def _material(
    index: int,
    *,
    source: DataSource,
    source_id: str,
    formula: str,
    composition: dict[str, float],
    space_group_number: int | None,
    space_group_symbol: str | None,
    band_gap: float,
    density: float | None = None,
    volume: float | None = None,
    nsites: int | None = None,
) -> MaterialRecord:
    elements = tuple(sorted(composition))
    props = []
    prop = make_source_property(
        PropertyName.BAND_GAP, band_gap, source, source_id, method="test"
    )
    if prop is not None:
        props.append(prop)
    if density is not None:
        dprop = make_source_property(
            PropertyName.DENSITY, density, source, source_id, method="test"
        )
        if dprop is not None:
            props.append(dprop)
    sites = nsites if nsites is not None else int(sum(composition.values()))
    record = SourceRecord(
        source=source,
        source_id=source_id,
        composition=CompositionInfo(
            formula_pretty=formula,
            reduced_formula=formula,
            chemsys="-".join(elements),
            elements=elements,
            nelements=len(elements),
            composition=composition,
        ),
        properties=tuple(props),
        symmetry=SymmetryInfo(
            space_group_symbol=space_group_symbol,
            space_group_number=space_group_number,
        ),
        structure=StructureRef(nsites=sites, density=density, volume=volume),
        material_id=make_material_id(index),
    )
    return MaterialRecord(material_id=make_material_id(index), sources=(record,))


def test_resolve_entities_merges_cross_source(tmp_path: Path) -> None:
    db_path = tmp_path / "test.duckdb"
    db = MaterialsDatabase(db_path)
    db.reset()
    db.insert_materials(
        [
            _material(
                1,
                source=DataSource.MATERIALS_PROJECT,
                source_id="mp-si",
                formula="Si",
                composition={"Si": 2.0},
                space_group_number=227,
                space_group_symbol="Fd-3m",
                band_gap=0.6,
            ),
            _material(
                2,
                source=DataSource.AFLOW,
                source_id="aflow-si",
                formula="Si2",
                composition={"Si": 2.0},
                space_group_number=227,
                space_group_symbol=None,
                band_gap=0.55,
            ),
            _material(
                3,
                source=DataSource.JARVIS,
                source_id="jv-other",
                formula="Al",
                composition={"Al": 1.0},
                space_group_number=225,
                space_group_symbol="Fm-3m",
                band_gap=0.0,
            ),
        ]
    )
    assert db.count_materials() == 3

    settings = SimpleNamespace(
        duckdb_path=db_path,
        database_config={
            "entity_resolution": {
                "enabled": True,
                "strategy": "composition_and_structure_fingerprint",
            }
        },
        ensure_directories=lambda: db_path.parent.mkdir(parents=True, exist_ok=True),
    )
    stats = resolve_entities(
        settings=settings,  # type: ignore[arg-type]
        dry_run=False,
        require_cross_source=True,
    )
    assert stats.merge_groups == 1
    assert stats.materials_after == 2
    assert db.count_materials() == 2
    assert db.count_by_source(DataSource.MATERIALS_PROJECT.value) == 1
    assert db.count_by_source(DataSource.AFLOW.value) == 1

    with db.connect() as conn:
        si_sources = conn.execute(
            """
            SELECT source FROM source_records
            WHERE material_id = (
                SELECT material_id FROM source_records WHERE source_id = 'mp-si'
            )
            ORDER BY source
            """
        ).fetchall()
        assert {r[0] for r in si_sources} == {"aflow", "materials_project"}
        n_sources = conn.execute(
            """
            SELECT n_sources FROM search_index
            WHERE material_id = (
                SELECT material_id FROM source_records WHERE source_id = 'mp-si'
            )
            """
        ).fetchone()[0]
        assert n_sources == 2


def test_density_compatible_helper() -> None:
    from database.fingerprints import density_or_vpa_compatible

    assert density_or_vpa_compatible(density_a=2.33, density_b=2.35)
    assert not density_or_vpa_compatible(density_a=2.33, density_b=4.0)
    assert density_or_vpa_compatible(
        volume_a=40.0, nsites_a=2, volume_b=41.0, nsites_b=2
    )
    assert not density_or_vpa_compatible(
        volume_a=40.0, nsites_a=2, volume_b=80.0, nsites_b=2
    )


def test_resolve_entities_rejects_density_mismatch(tmp_path: Path) -> None:
    db_path = tmp_path / "test.duckdb"
    db = MaterialsDatabase(db_path)
    db.reset()
    db.insert_materials(
        [
            _material(
                1,
                source=DataSource.MATERIALS_PROJECT,
                source_id="mp-fe",
                formula="Fe",
                composition={"Fe": 2.0},
                space_group_number=229,
                space_group_symbol="Im-3m",
                band_gap=0.0,
                density=7.87,
                volume=23.0,
                nsites=2,
            ),
            _material(
                2,
                source=DataSource.AFLOW,
                source_id="aflow-fe-dense",
                formula="Fe2",
                composition={"Fe": 2.0},
                space_group_number=229,
                space_group_symbol=None,
                band_gap=0.0,
                density=12.5,
                volume=15.0,
                nsites=2,
            ),
            _material(
                3,
                source=DataSource.JARVIS,
                source_id="jv-fe",
                formula="Fe2",
                composition={"Fe": 2.0},
                space_group_number=229,
                space_group_symbol=None,
                band_gap=0.0,
                density=7.90,
                volume=23.1,
                nsites=2,
            ),
        ]
    )

    settings = SimpleNamespace(
        duckdb_path=db_path,
        database_config={
            "entity_resolution": {
                "enabled": True,
                "strategy": "composition_and_structure_fingerprint",
                "density_rel_tol": 0.05,
                "density_abs_tol": 0.15,
            }
        },
        ensure_directories=lambda: db_path.parent.mkdir(parents=True, exist_ok=True),
    )
    stats = resolve_entities(
        settings=settings,  # type: ignore[arg-type]
        dry_run=False,
        require_cross_source=True,
    )
    # MP + JARVIS merge; AFLOW density outlier stays separate → 2 materials
    assert stats.merge_groups == 1
    assert stats.materials_after == 2
    with db.connect() as conn:
        linked = {
            r[0]
            for r in conn.execute(
                """
                SELECT source FROM source_records
                WHERE material_id = (
                    SELECT material_id FROM source_records WHERE source_id = 'mp-fe'
                )
                """
            ).fetchall()
        }
        assert linked == {"jarvis", "materials_project"}
        aflow_alone = conn.execute(
            """
            SELECT COUNT(*) FROM source_records
            WHERE material_id = (
                SELECT material_id FROM source_records WHERE source_id = 'aflow-fe-dense'
            )
            """
        ).fetchone()[0]
        assert aflow_alone == 1
