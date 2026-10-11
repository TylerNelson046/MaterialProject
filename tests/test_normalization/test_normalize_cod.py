"""Tests for COD normalization."""

from __future__ import annotations

from config.constants import DataSource, PropertyName
from normalization.normalize_cod import normalize_cod_document, parse_cod_formula

SAMPLE = {
    "source_id": "cod-1000007",
    "file": "1000007",
    "cod_id": 1000007,
    "formula": "- Ca Mg O6 Si2 -",
    "calcformula": "- Ca4 Mg4 O24 Si8 -",
    "nel": "4",
    "sg": "C 1 2/c 1",
    "sgNumber": "15",
    "vol": "438.631",
    "a": "9.746",
    "b": "8.899",
    "c": "5.251",
    "alpha": "90",
    "beta": "105.63",
    "gamma": "90",
    "mineral": "Diopside",
    "structure_path": "data/structures/experimental/cod-1000007.cif",
}


def test_parse_cod_formula() -> None:
    amounts = parse_cod_formula("- Ca Mg O6 Si2 -")
    assert amounts["Ca"] == 1.0
    assert amounts["Mg"] == 1.0
    assert amounts["Si"] == 2.0
    assert amounts["O"] == 6.0


def test_normalize_cod_document() -> None:
    material = normalize_cod_document(SAMPLE, index=5)
    assert material.material_id == "MAT-000005"
    source = material.primary_source
    assert source.source == DataSource.COD
    assert source.source_id == "cod-1000007"
    assert source.composition.chemsys == "Ca-Mg-O-Si"
    assert source.symmetry.space_group_number == 15
    assert source.structure is not None
    assert source.structure.path.endswith("cod-1000007.cif")
    assert source.structure.format == "cif"
    assert source.get_property(PropertyName.THEORETICAL).value is False
    assert source.get_property(PropertyName.VOLUME).value == 438.631
    density = source.get_property(PropertyName.DENSITY)
    assert density is not None
    assert density.value > 0
    assert source.structure.nsites == 40  # Ca4 Mg4 O24 Si8
