"""Tests for Materials Project normalization."""

from __future__ import annotations

from config.constants import DataSource, PropertyName
from normalization.normalize_mp import normalize_mp_document


SAMPLE_DOC = {
    "material_id": "mp-149",
    "formula_pretty": "Si",
    "composition": {"Si": 1.0},
    "chemsys": "Si",
    "elements": ["Si"],
    "nelements": 1,
    "nsites": 2,
    "volume": 40.0,
    "density": 2.3,
    "formation_energy_per_atom": 0.0,
    "energy_above_hull": 0.0,
    "energy_per_atom": -5.4,
    "is_stable": True,
    "band_gap": 0.85,
    "is_gap_direct": True,
    "is_metal": False,
    "is_magnetic": False,
    "ordering": "NM",
    "total_magnetization": 0.0,
    "theoretical": False,
    "efermi": 5.5,
    "bulk_modulus": {"voigt": 100.0, "reuss": 96.0, "vrh": 98.0},
    "shear_modulus": {"voigt": 60.0, "reuss": 58.0, "vrh": 59.0},
    "homogeneous_poisson": 0.25,
    "e_total": 12.0,
    "n": 3.4,
    "symmetry": {
        "crystal_system": "Cubic",
        "symbol": "Fd-3m",
        "number": 227,
        "point_group": "m-3m",
    },
}


def test_normalize_mp_document_preserves_provenance() -> None:
    material = normalize_mp_document(SAMPLE_DOC, index=1)
    assert material.material_id == "MAT-000001"
    assert material.formula_pretty == "Si"

    source = material.primary_source
    assert source.source == DataSource.MATERIALS_PROJECT
    assert source.source_id == "mp-149"
    assert source.material_id == "MAT-000001"
    assert source.composition.elements == ("Si",)
    assert source.symmetry is not None
    assert source.symmetry.space_group_number == 227

    band_gap = source.get_property(PropertyName.BAND_GAP)
    assert band_gap is not None
    assert band_gap.value == 0.85
    assert band_gap.unit == "eV"
    assert band_gap.origin == "known"

    bulk = source.get_property(PropertyName.BULK_MODULUS_VRH)
    assert bulk is not None
    assert bulk.value == 98.0
    assert bulk.unit == "GPa"

    assert source.get_property(PropertyName.IS_GAP_DIRECT).value is True
    assert source.get_property(PropertyName.ORDERING).value == "NM"
    assert source.composition.reduced_formula == "Si"


def test_normalize_mp_elastic_falls_back_to_reuss_when_vrh_corrupt() -> None:
    """MP unstable tensors often diverge Voigt/VRH while Reuss stays physical."""
    doc = {
        **SAMPLE_DOC,
        "material_id": "mp-corrupt-elastic",
        "bulk_modulus": [
            ["voigt", -4.7e11],
            ["reuss", 211.9],
            ["vrh", -2.35e11],
        ],
        "shear_modulus": [
            ["voigt", 1e12],
            ["reuss", 80.0],
            ["vrh", 5e11],
        ],
        "homogeneous_poisson": 12.0,  # physically impossible
        "universal_anisotropy": -2.5e7,
    }
    source = normalize_mp_document(doc, index=9).primary_source
    assert source.get_property(PropertyName.BULK_MODULUS_VRH).value == 211.9
    assert source.get_property(PropertyName.SHEAR_MODULUS_VRH).value == 80.0
    assert source.get_property(PropertyName.HOMOGENEOUS_POISSON) is None
    assert source.get_property(PropertyName.UNIVERSAL_ANISOTROPY) is None


def test_normalize_mp_elastic_drops_fully_corrupt_tensor() -> None:
    doc = {
        **SAMPLE_DOC,
        "material_id": "mp-all-bad",
        "bulk_modulus": {"voigt": 1e9, "reuss": -1e8, "vrh": 5e8},
        "shear_modulus": {"voigt": -1e9, "reuss": 2e9, "vrh": 1e9},
    }
    source = normalize_mp_document(doc, index=10).primary_source
    assert source.get_property(PropertyName.BULK_MODULUS_VRH) is None
    assert source.get_property(PropertyName.SHEAR_MODULUS_VRH) is None


def test_normalize_mp_list_composition_reduced() -> None:
    """Arrow/JSONL often stores composition_reduced as list-of-pairs."""
    doc = {
        **SAMPLE_DOC,
        "material_id": "mp-aaabxbze",
        "formula_pretty": "LiEr2Ir",
        "composition": [["Li", 1.0], ["Er", 2.0], ["Ir", 1.0]],
        "composition_reduced": [["Li", 1.0], ["Er", 2.0], ["Ir", 1.0]],
        "elements": ["Li", "Er", "Ir"],
        "nelements": 3,
        "chemsys": "Er-Ir-Li",
    }
    material = normalize_mp_document(doc, index=2)
    reduced = material.primary_source.composition.reduced_formula
    assert reduced is not None
    assert not reduced.startswith("[")
    assert "Li" in reduced and "Er" in reduced and "Ir" in reduced
    assert material.primary_source.composition.composition == {
        "Li": 1.0,
        "Er": 2.0,
        "Ir": 1.0,
    }
