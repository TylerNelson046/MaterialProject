"""Tests for JARVIS normalization."""

from __future__ import annotations

from config.constants import DataSource, PropertyName
from normalization.normalize_jarvis import normalize_jarvis_document

SAMPLE = {
    "jid": "JVASP-1002",
    "formula": "Si",
    "search": "-Si",
    "formation_energy_peratom": 0.0,
    "ehull": 0.0,
    "optb88vdw_bandgap": 0.6,
    "optb88vdw_total_energy": -5.4,
    "density": 2.3,
    "nat": 2,
    "magmom_outcar": 0.0,
    "bulk_modulus_kv": 98.0,
    "shear_modulus_gv": 60.0,
    "poisson": 0.25,
    "spg_number": "227",
    "spg_symbol": "Fd-3m",
    "crys": "cubic",
    "atoms": {
        "elements": ["Si", "Si"],
        "lattice_mat": [[3.8, 0.0, 0.0], [0.0, 3.8, 0.0], [0.0, 0.0, 3.8]],
        "abc": [3.8, 3.8, 3.8],
        "angles": [90.0, 90.0, 90.0],
    },
    "mbj_bandgap": 1.1,
    "avg_elec_mass": 0.2,
    "avg_hole_mass": 0.3,
    "slme": 25.0,
    "Tc_supercon": "na",
    "exfoliation_energy": 80.0,
    "reference": "mp-149",
}


def test_normalize_jarvis_document() -> None:
    material = normalize_jarvis_document(SAMPLE, index=10)
    assert material.material_id == "MAT-000010"
    source = material.primary_source
    assert source.source == DataSource.JARVIS
    assert source.source_id == "JVASP-1002"
    assert source.composition.elements == ("Si",)
    assert source.get_property(PropertyName.BAND_GAP).value == 0.6
    assert source.get_property(PropertyName.BULK_MODULUS_VRH).value == 98.0
    assert source.get_property(PropertyName.IS_STABLE).value is True
    assert source.get_property(PropertyName.IS_METAL).value is False
    assert source.get_property(PropertyName.MBJ_BANDGAP).value == 1.1
    assert source.get_property(PropertyName.AVG_ELEC_MASS).value == 0.2
    assert source.get_property(PropertyName.SLME).value == 25.0
    assert source.get_property(PropertyName.EXFOLIATION_ENERGY).value == 80.0
    assert source.get_property(PropertyName.TC_SUPERCON) is None
    assert source.structure is not None
    assert source.structure.volume is not None
    assert abs(source.structure.volume - (3.8**3)) < 1e-6
    assert source.get_property(PropertyName.VOLUME).value == source.structure.volume


def test_normalize_jarvis_formula_only_no_atoms() -> None:
    """Bulk JSONL has formula but no atoms/elements — must still parse."""
    doc = {
        "jid": "JVASP-90856",
        "formula": "TiCuSiAs",
        "formation_energy_peratom": -0.4,
        "ehull": 0.01,
        "optb88vdw_bandgap": 0.0,
        "density": 5.0,
        "nat": 8,
        "spg_number": 129,
        "spg_symbol": "P4/nmm",
    }
    material = normalize_jarvis_document(doc, index=1)
    comp = material.primary_source.composition
    assert comp.elements == ("As", "Cu", "Si", "Ti")
    assert comp.nelements == 4
    assert comp.chemsys == "As-Cu-Si-Ti"
    assert comp.composition["Ti"] == 1.0
    assert comp.composition["Cu"] == 1.0
