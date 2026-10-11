"""Tests for the unified material schema."""

from __future__ import annotations

import pytest

from config.constants import PROPERTY_UNITS, DataSource, PropertyName
from normalization.schema import (
    CompositionInfo,
    MaterialRecord,
    SourceProperty,
    SourceRecord,
    make_material_id,
    make_source_property,
)


def _si_composition() -> CompositionInfo:
    return CompositionInfo(
        formula_pretty="Si",
        reduced_formula="Si",
        chemsys="Si",
        elements=("Si",),
        nelements=1,
        composition={"Si": 1.0},
    )


def test_make_material_id() -> None:
    assert make_material_id(1) == "MAT-000001"
    assert make_material_id(149) == "MAT-000149"
    with pytest.raises(ValueError):
        make_material_id(0)


def test_source_property_enforces_canonical_units() -> None:
    prop = SourceProperty.from_canonical(
        PropertyName.BAND_GAP,
        1.1,
        DataSource.MATERIALS_PROJECT,
        "mp-149",
    )
    assert prop.unit == PROPERTY_UNITS[PropertyName.BAND_GAP]

    with pytest.raises(ValueError, match="Unit mismatch"):
        SourceProperty(
            name=PropertyName.BAND_GAP.value,
            value=1.1,
            unit="wrong",
            source=DataSource.MATERIALS_PROJECT,
            source_id="mp-149",
        )


def test_conflicting_properties_are_preserved() -> None:
    """Same material, two sources, different band gaps — both kept."""
    mp_gap = SourceProperty.from_canonical(
        PropertyName.BAND_GAP, 1.1, DataSource.MATERIALS_PROJECT, "mp-149"
    )
    jarvis_gap = SourceProperty.from_canonical(
        PropertyName.BAND_GAP, 1.2, DataSource.JARVIS, "JVASP-1002"
    )

    material = MaterialRecord(
        material_id=make_material_id(1),
        sources=(
            SourceRecord(
                source=DataSource.MATERIALS_PROJECT,
                source_id="mp-149",
                composition=_si_composition(),
                properties=(mp_gap,),
                material_id=make_material_id(1),
            ),
            SourceRecord(
                source=DataSource.JARVIS,
                source_id="JVASP-1002",
                composition=_si_composition(),
                properties=(jarvis_gap,),
                material_id=make_material_id(1),
            ),
        ),
    )

    gaps = material.properties_for(PropertyName.BAND_GAP)
    assert len(gaps) == 2
    assert {g.source for g in gaps} == {
        DataSource.MATERIALS_PROJECT,
        DataSource.JARVIS,
    }
    assert {g.value for g in gaps} == {1.1, 1.2}


def test_round_trip_dict_serialization() -> None:
    band_gap = make_source_property(
        PropertyName.BAND_GAP,
        1.1,
        DataSource.MATERIALS_PROJECT,
        "mp-149",
    )
    assert band_gap is not None

    original = MaterialRecord(
        material_id=make_material_id(1),
        sources=(
            SourceRecord(
                source=DataSource.MATERIALS_PROJECT,
                source_id="mp-149",
                composition=_si_composition(),
                properties=(band_gap,),
                original_metadata={"raw_id": "mp-149"},
                material_id=make_material_id(1),
            ),
        ),
    )

    restored = MaterialRecord.from_dict(original.to_dict())
    assert restored.material_id == original.material_id
    assert restored.formula_pretty == "Si"
    assert restored.primary_source.source_id == "mp-149"
    assert restored.primary_source.get_property(PropertyName.BAND_GAP).value == 1.1
    assert restored.primary_source.original_metadata["raw_id"] == "mp-149"


def test_make_source_property_skips_none() -> None:
    assert (
        make_source_property(
            PropertyName.BAND_GAP,
            None,
            DataSource.MATERIALS_PROJECT,
            "mp-149",
        )
        is None
    )


def test_material_record_requires_valid_id_and_sources() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MaterialRecord(material_id=make_material_id(1), sources=())

    with pytest.raises(ValueError, match="MAT-"):
        MaterialRecord(
            material_id="BAD-1",
            sources=(
                SourceRecord(
                    source=DataSource.MATERIALS_PROJECT,
                    source_id="mp-149",
                    composition=_si_composition(),
                ),
            ),
        )
