"""Hull protocol unit tests (Phase 4.7)."""

from __future__ import annotations

import pytest

pytest.importorskip("pymatgen")

from ml.mace.hull import (  # noqa: E402
    PROTOCOL_MACE_PD_EFORM_V1,
    HullEntry,
    build_phase_diagram,
    e_above_hull_per_atom,
    predict_e_above_hull,
)


def test_stable_binary_on_hull() -> None:
    refs = [
        HullEntry("a", "Li", 0.0, 1),
        HullEntry("b", "O2", 0.0, 2),
        HullEntry("c", "Li2O", -2.0, 3),
        HullEntry("d", "Li2O2", -1.0, 4),
    ]
    # Li2O should be near/on hull relative to this toy set.
    e_hull = predict_e_above_hull(
        "Li2O", -2.0, refs, protocol=PROTOCOL_MACE_PD_EFORM_V1
    )
    assert e_hull >= -1e-8
    assert e_hull < 1e-6


def test_unstable_above_hull() -> None:
    refs = [
        HullEntry("c", "Li2O", -2.0, 3),
        HullEntry("d", "Li2O2", -0.1, 4),  # less stable
    ]
    e_hull = predict_e_above_hull(
        "Li2O2", -0.1, refs, protocol=PROTOCOL_MACE_PD_EFORM_V1
    )
    assert e_hull > 0.0


def test_build_phase_diagram_and_query() -> None:
    refs = [
        HullEntry("c", "MgO", -3.0, 2),
        HullEntry("d", "Mg2O", -1.0, 3),
    ]
    diagram = build_phase_diagram(refs, elements=["Mg", "O"])
    e_h = e_above_hull_per_atom("MgO", -3.0, diagram)
    assert e_h < 1e-6


def test_predict_e_above_hull_requires_covering_refs() -> None:
    with pytest.raises(ValueError, match="No covering hull reference"):
        predict_e_above_hull(
            "NaCl",
            -1.0,
            [HullEntry("x", "Li2O", -2.0, 3)],
            protocol=PROTOCOL_MACE_PD_EFORM_V1,
        )
