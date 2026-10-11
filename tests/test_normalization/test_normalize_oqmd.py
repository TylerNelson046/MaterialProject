"""Tests for OQMD normalization."""

from __future__ import annotations

from config.constants import DataSource, PropertyName
from normalization.normalize_oqmd import normalize_oqmd_document

SAMPLE = {
    "source_id": "oqmd-1216058",
    "name": "Lu",
    "entry_id": 1216058,
    "composition": "Lu1",
    "composition_generic": "A",
    "spacegroup": "R-3m",
    "volume": 86.3513,
    "ntypes": 1,
    "natoms": 3,
    "band_gap": 0.0,
    "delta_e": 0.0125,
    "stability": 0.0125,
    "prototype": "C19_alpha_Sm",
    "unit_cell": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
}


def test_normalize_oqmd_document() -> None:
    material = normalize_oqmd_document(SAMPLE, index=7)
    assert material.material_id == "MAT-000007"
    source = material.primary_source
    assert source.source == DataSource.OQMD
    assert source.source_id == "oqmd-1216058"
    assert source.composition.elements == ("Lu",)
    assert source.get_property(PropertyName.BAND_GAP).value == 0.0
    assert source.get_property(PropertyName.ENERGY_ABOVE_HULL).value == 0.0125
    assert source.get_property(PropertyName.IS_METAL).value is True
    assert source.get_property(PropertyName.IS_STABLE).value is False
    assert source.symmetry is not None
    assert source.symmetry.space_group_symbol == "R-3m"
    assert source.symmetry.space_group_number == 166
    assert source.symmetry.crystal_system is not None
    dens = source.get_property(PropertyName.DENSITY)
    assert dens is not None
    assert dens.value > 0
    assert dens.unit == "g/cm^3"
    assert source.structure is not None
    assert source.structure.density == dens.value


def test_normalize_oqmd_spacegroup_without_underscore() -> None:
    doc = {**SAMPLE, "spacegroup": "P21/c", "entry_id": 1, "source_id": "oqmd-1"}
    sym = normalize_oqmd_document(doc, index=1).primary_source.symmetry
    assert sym is not None
    assert sym.space_group_number == 14

