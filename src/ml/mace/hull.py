"""Named energy-above-hull protocol from MACE formation energies (Phase 4.7).

Protocol ``mace_pd_eform_v1``:
* Reference pool: structure-gated materials with MACE formation energies
  (eV/atom; default reference protocol ``elemental_bulk_v1``).
* For each chemical system, build a pymatgen ``PhaseDiagram`` from reference
  entries whose element set ⊆ the system, plus elemental references at E=0.
* Predicted E_hull = ``PhaseDiagram.get_e_above_hull`` (eV/atom).

This is **not** a single-structure regressor against export ``label``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

PROTOCOL_MACE_PD_EFORM_V1 = "mace_pd_eform_v1"
HULL_PROTOCOLS = frozenset({PROTOCOL_MACE_PD_EFORM_V1})


@dataclass(frozen=True)
class HullEntry:
    """One composition + formation energy for hull construction."""

    material_id: str
    formula: str
    e_form_per_atom: float
    n_atoms: int


def _composition(formula: str):
    from pymatgen.core import Composition

    return Composition(formula)


def _entry_energy_total(e_form_per_atom: float, composition) -> float:
    """Total formation energy for a Composition amount (elemental refs at 0)."""
    return float(e_form_per_atom) * float(composition.num_atoms)


def build_phase_diagram(
    entries: Iterable[HullEntry],
    *,
    elements: Iterable[str] | None = None,
):
    """Build a formation-energy ``PhaseDiagram`` (elemental refs at 0)."""
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition, Element

    pd_entries = []
    seen_elements: set[str] = set()
    for item in entries:
        comp = _composition(item.formula)
        seen_elements.update(str(el) for el in comp.elements)
        pd_entries.append(
            PDEntry(comp, _entry_energy_total(item.e_form_per_atom, comp))
        )

    if elements is not None:
        wanted = {str(e) for e in elements}
    else:
        wanted = seen_elements

    for el in sorted(wanted):
        pd_entries.append(PDEntry(Composition({Element(el): 1}), 0.0))

    if len(pd_entries) < 2:
        raise ValueError("Need at least one compound + elements for a phase diagram")
    return PhaseDiagram(pd_entries)


def e_above_hull_per_atom(
    formula: str,
    e_form_per_atom: float,
    diagram,
) -> float:
    """Return E_hull (eV/atom) for a composition with known E_form/atom."""
    from pymatgen.analysis.phase_diagram import PDEntry

    comp = _composition(formula)
    entry = PDEntry(comp, _entry_energy_total(e_form_per_atom, comp))
    return float(diagram.get_e_above_hull(entry))


def filter_entries_for_system(
    entries: list[HullEntry],
    formula: str,
) -> list[HullEntry]:
    """Keep reference entries whose elements are a subset of ``formula``'s."""
    query = set(str(el) for el in _composition(formula).elements)
    kept: list[HullEntry] = []
    for item in entries:
        els = set(str(el) for el in _composition(item.formula).elements)
        if els.issubset(query):
            kept.append(item)
    return kept


def predict_e_above_hull(
    formula: str,
    e_form_per_atom: float,
    reference_entries: list[HullEntry],
    *,
    protocol: str = PROTOCOL_MACE_PD_EFORM_V1,
) -> float:
    """Predict E_hull/atom under the named protocol."""
    if protocol != PROTOCOL_MACE_PD_EFORM_V1:
        raise ValueError(
            f"Unsupported hull protocol {protocol!r}; "
            f"expected one of {sorted(HULL_PROTOCOLS)}"
        )
    subset = filter_entries_for_system(reference_entries, formula)
    if not subset:
        # Without covering compounds, a diagram of (query + elemental E=0) makes
        # every e_form≤0 look "on hull". Refuse rather than invent stability.
        raise ValueError(
            f"No covering hull reference entries for formula {formula!r}; "
            "expand the E_form cache / reference pool for this chemical system."
        )
    # Include the query so the diagram is well-defined when it is among the
    # reference compounds (E_hull → 0 when it is the ground state).
    query_id = "__query__"
    subset = list(subset) + [
        HullEntry(
            material_id=query_id,
            formula=formula,
            e_form_per_atom=float(e_form_per_atom),
            n_atoms=int(round(_composition(formula).num_atoms)),
        )
    ]
    elements = [str(el) for el in _composition(formula).elements]
    diagram = build_phase_diagram(subset, elements=elements)
    return e_above_hull_per_atom(formula, e_form_per_atom, diagram)


def reference_entries_from_frame(frame: pd.DataFrame) -> list[HullEntry]:
    """Build ``HullEntry`` list from a predictions / pack frame.

    Required columns: ``material_id``, ``formula`` (or ``formula_pretty``),
    ``e_form_per_atom`` (or ``y_pred_formation``), optional ``n_atoms``.
    """
    formula_col = "formula" if "formula" in frame.columns else "formula_pretty"
    e_col = (
        "e_form_per_atom"
        if "e_form_per_atom" in frame.columns
        else "y_pred_formation"
    )
    required = {"material_id", formula_col, e_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Frame missing columns for hull reference: {sorted(missing)}")

    rows: list[HullEntry] = []
    for rec in frame.itertuples(index=False):
        formula = getattr(rec, formula_col)
        if formula is None or (isinstance(formula, float) and np.isnan(formula)):
            continue
        e_form = float(getattr(rec, e_col))
        n_atoms = getattr(rec, "n_atoms", None)
        if n_atoms is None or (isinstance(n_atoms, float) and np.isnan(n_atoms)):
            n_atoms = int(round(_composition(str(formula)).num_atoms))
        rows.append(
            HullEntry(
                material_id=str(rec.material_id),
                formula=str(formula),
                e_form_per_atom=e_form,
                n_atoms=int(n_atoms),
            )
        )
    return rows
