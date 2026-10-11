"""Normalize Alexandria on-hull (JARVIS figshare) documents."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from config.constants import DataSource, PropertyName
from normalization.physical_ranges import sanitize_density, sanitize_ehull, sanitize_volume
from normalization.schema import (
    CompositionInfo,
    SourceProperty,
    SourceRecord,
    StructureRef,
    SymmetryInfo,
    make_source_property,
    utc_now_iso,
)

SOURCE = DataSource.ALEXANDRIA
METHOD = "alexandria-pbe-hull"


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _composition_from_atoms(atoms: dict[str, Any]) -> CompositionInfo:
    elements = atoms.get("elements") or []
    counts = Counter(str(el) for el in elements if el)
    composition = {el: float(n) for el, n in sorted(counts.items())}
    try:
        from pymatgen.core import Composition

        comp = Composition(composition)
        formula = comp.reduced_formula
        chemsys = comp.chemical_system
    except Exception:  # noqa: BLE001
        formula = "".join(f"{el}{int(n) if n == int(n) else n}" for el, n in composition.items()) or "unknown"
        chemsys = "-".join(sorted(composition))
    return CompositionInfo(
        formula_pretty=formula,
        reduced_formula=formula,
        chemsys=chemsys,
        elements=tuple(sorted(composition)),
        nelements=len(composition),
        composition=composition,
    )


def _cell_metrics(atoms: dict[str, Any]) -> tuple[float | None, float | None, int | None]:
    lattice = atoms.get("lattice_mat")
    elements = atoms.get("elements") or []
    nsites = len(elements) if elements else None
    volume = None
    density = None
    if isinstance(lattice, list) and len(lattice) == 3:
        try:
            import numpy as np

            mat = np.array(lattice, dtype=float)
            volume = float(abs(np.linalg.det(mat)))
        except Exception:  # noqa: BLE001
            volume = None
    volume = sanitize_volume(volume)
    if volume and nsites and atoms.get("elements"):
        try:
            from pymatgen.core import Composition

            mass = float(Composition(Counter(atoms["elements"])).weight)
            # mass in amu, volume Å³ → g/cm³
            density = sanitize_density(mass / volume * 1.66053906660)
        except Exception:  # noqa: BLE001
            density = None
    return volume, density, nsites


def _symmetry_from_atoms(atoms: dict[str, Any]) -> SymmetryInfo:
    try:
        from pymatgen.core import Lattice, Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        structure = Structure(
            Lattice(atoms["lattice_mat"]),
            atoms["elements"],
            atoms["coords"],
            coords_are_cartesian=False,
        )
        for symprec in (0.1, 0.01, 0.2):
            try:
                sga = SpacegroupAnalyzer(structure, symprec=symprec)
                number = int(sga.get_space_group_number())
                symbol = str(sga.get_space_group_symbol())
                crystal = None
                for lo, hi, name in (
                    (1, 2, "Triclinic"),
                    (3, 15, "Monoclinic"),
                    (16, 74, "Orthorhombic"),
                    (75, 142, "Tetragonal"),
                    (143, 167, "Trigonal"),
                    (168, 194, "Hexagonal"),
                    (195, 230, "Cubic"),
                ):
                    if lo <= number <= hi:
                        crystal = name
                        break
                return SymmetryInfo(
                    crystal_system=crystal,
                    space_group_symbol=symbol,
                    space_group_number=number,
                )
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return SymmetryInfo()


def normalize_alexandria_document(doc: dict[str, Any]) -> SourceRecord | None:
    source_id = str(doc.get("id") or "").strip()
    if not source_id:
        return None
    atoms = doc.get("atoms")
    if not isinstance(atoms, dict):
        return None
    composition = _composition_from_atoms(atoms)
    volume, density, nsites = _cell_metrics(atoms)
    symmetry = _symmetry_from_atoms(atoms)
    energy_total = _num(doc.get("energy_corrected"))
    if energy_total is None:
        energy_total = _num(doc.get("energy_total"))
    energy_per_atom = None
    if energy_total is not None and nsites:
        energy_per_atom = energy_total / float(nsites)

    props: list[SourceProperty] = []
    for name, value in (
        (PropertyName.ENERGY_PER_ATOM, energy_per_atom),
        (PropertyName.ENERGY_ABOVE_HULL, sanitize_ehull(0.0)),  # on-hull slice
        (PropertyName.IS_STABLE, True),
        (PropertyName.DENSITY, density),
        (PropertyName.VOLUME, volume),
        (PropertyName.NSITES, nsites),
        (PropertyName.THEORETICAL, True),
    ):
        prop = make_source_property(name, value, SOURCE, source_id, method=METHOD)
        if prop is not None:
            props.append(prop)

    return SourceRecord(
        source=SOURCE,
        source_id=source_id,
        composition=composition,
        symmetry=symmetry,
        properties=tuple(props),
        structure=StructureRef(
            path=None,
            format=None,
            nsites=nsites,
            volume=volume,
            density=density,
        ),
        original_metadata={
            "id": source_id,
            "dataset": "alex_pbe_hull",
            "energy_corrected": doc.get("energy_corrected"),
            "energy_total": doc.get("energy_total"),
        },
        retrieved_at=utc_now_iso(),
    )


def iter_raw_alexandria_documents(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
