"""Tests for AFLOW normalization."""

from __future__ import annotations

from config.constants import DataSource, PropertyName
from normalization.normalize_aflow import normalize_aflow_document

SAMPLE = {
    "source_id": "aflow-00026003d47ada0f",
    "auid": "aflow:00026003d47ada0f",
    "aurl": "aflowlib.duke.edu:AFLOWDATA/ICSD_WEB/MCLC/O2_ICSD_173933",
    "compound": "O2",
    "species": ["O"],
    "nspecies": 1,
    "natoms": 2,
    "density": 1.19243,
    "volume_cell": 44.5,
    "spacegroup_relax": 12,
    "Pearson_symbol_relax": "mS4",
    "Egap": 1.5421,
    "enthalpy_formation_atom": -0.0058465,
    "energy_atom": -4.93699,
    "ael_bulk_modulus_vrh": None,
    "ael_shear_modulus_vrh": None,
    "catalog": "ICSD",
}


def test_normalize_aflow_document() -> None:
    material = normalize_aflow_document(SAMPLE, index=3)
    assert material.material_id == "MAT-000003"
    source = material.primary_source
    assert source.source == DataSource.AFLOW
    assert source.source_id == "aflow-00026003d47ada0f"
    assert source.composition.elements == ("O",)
    assert source.symmetry.space_group_number == 12
    assert source.symmetry.space_group_symbol is not None
    assert source.get_property(PropertyName.BAND_GAP).value == 1.5421
    assert source.get_property(PropertyName.FORMATION_ENERGY_PER_ATOM).value == -0.0058465
    assert source.get_property(PropertyName.IS_METAL).value is False
    assert source.get_property(PropertyName.THEORETICAL).value is False
