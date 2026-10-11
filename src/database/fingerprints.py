"""Composition / structure fingerprints for entity resolution."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

_ELEMENT_TOKEN = re.compile(r"([A-Z][a-z]*)(\d*(?:\.\d*)?)")


def _parse_composition_json(raw: Any) -> dict[str, float]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items() if v is not None}
    if isinstance(raw, str):
        text = raw.strip()
        if not text or text in {"{}", "null", "None"}:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return {str(k): float(v) for k, v in parsed.items() if v is not None}
    return {}


def _composition_from_formula(formula: str | None) -> dict[str, float]:
    if not formula:
        return {}
    try:
        from pymatgen.core import Composition

        return {str(el): float(amt) for el, amt in Composition(formula).items()}
    except Exception:  # noqa: BLE001
        out: dict[str, float] = {}
        for element, count in _ELEMENT_TOKEN.findall(formula):
            out[element] = out.get(element, 0.0) + float(count or 1)
        return out


def reduced_composition_key(
    *,
    composition: Any = None,
    formula_pretty: str | None = None,
    reduced_formula: str | None = None,
) -> str | None:
    """Canonical reduced-composition string, e.g. ``Ba1F4Mg1``."""
    amounts = _parse_composition_json(composition)
    if not amounts:
        if reduced_formula and not reduced_formula.strip().startswith("["):
            amounts = _composition_from_formula(reduced_formula)
        if not amounts:
            amounts = _composition_from_formula(formula_pretty)
    if not amounts:
        return None

    try:
        from pymatgen.core import Composition

        reduced, _factor = Composition(amounts).get_reduced_composition_and_factor()
        items = sorted((str(el), float(amt)) for el, amt in reduced.items())
    except Exception:  # noqa: BLE001
        values = [int(round(v)) if abs(v - round(v)) < 1e-6 else v for v in amounts.values()]
        if all(isinstance(v, int) for v in values):
            from functools import reduce
            from math import gcd

            g = reduce(gcd, values) or 1
            items = sorted((el, amounts[el] / g) for el in amounts)
        else:
            items = sorted((el, float(amt)) for el, amt in amounts.items())

    if not items:
        return None
    return "".join(f"{el}{amt:g}" for el, amt in items)


def _hm_symbol_candidates(symbol: str) -> tuple[str, ...]:
    """Generate Hermann–Mauguin spellings OQMD/AFLOW commonly omit underscores for.

    Examples: ``P21/c`` → ``P2_1/c``, ``P212121`` → ``P2_12_12_1``.
    """
    text = (symbol or "").strip()
    if not text:
        return ()
    candidates: list[str] = [text]
    current = text
    for _ in range(6):
        nxt = re.sub(
            r"(?<=[A-Za-z0-9])([2346])([12345])(?!\d|_)",
            r"\1_\2",
            current,
            count=1,
        )
        if nxt == current:
            break
        current = nxt
        if current not in candidates:
            candidates.append(current)
    return tuple(candidates)


@lru_cache(maxsize=1024)
def space_group_number_from_symbol(symbol: str) -> int | None:
    text = (symbol or "").strip()
    if not text:
        return None
    try:
        from pymatgen.symmetry.groups import SpaceGroup
    except Exception:  # noqa: BLE001
        return None

    for candidate in _hm_symbol_candidates(text):
        try:
            return int(SpaceGroup(candidate).int_number)
        except Exception:  # noqa: BLE001
            continue
    return None


def resolve_space_group_number(
    *,
    space_group_number: Any = None,
    space_group_symbol: str | None = None,
) -> int | None:
    if space_group_number is not None:
        try:
            number = int(space_group_number)
            if 1 <= number <= 230:
                return number
        except (TypeError, ValueError):
            pass
    if space_group_symbol:
        return space_group_number_from_symbol(str(space_group_symbol))
    return None


def structure_fingerprint(
    *,
    composition: Any = None,
    formula_pretty: str | None = None,
    reduced_formula: str | None = None,
    space_group_number: Any = None,
    space_group_symbol: str | None = None,
    nsites: Any = None,
) -> str | None:
    """Fingerprint: reduced composition + space-group number + site count.

    ``nsites`` avoids merging distinct cells that share only chemistry + SG
    (common for oxyfluorides and other polymorph-rich chemistries).
    Density/volume soft-matching is applied later in entity resolution.
    """
    comp_key = reduced_composition_key(
        composition=composition,
        formula_pretty=formula_pretty,
        reduced_formula=reduced_formula,
    )
    sg = resolve_space_group_number(
        space_group_number=space_group_number,
        space_group_symbol=space_group_symbol,
    )
    try:
        sites = int(nsites) if nsites is not None else None
    except (TypeError, ValueError):
        sites = None
    if not comp_key or sg is None or sites is None or sites <= 0:
        return None
    return f"{comp_key}|sg:{sg}|n:{sites}"


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def volume_per_atom(volume: Any, nsites: Any) -> float | None:
    vol = _finite_float(volume)
    try:
        sites = int(nsites) if nsites is not None else None
    except (TypeError, ValueError):
        sites = None
    if vol is None or sites is None or sites <= 0 or vol <= 0:
        return None
    return vol / float(sites)


def density_or_vpa_compatible(
    *,
    density_a: Any = None,
    density_b: Any = None,
    volume_a: Any = None,
    volume_b: Any = None,
    nsites_a: Any = None,
    nsites_b: Any = None,
    density_rel_tol: float = 0.05,
    density_abs_tol: float = 0.15,
    vpa_rel_tol: float = 0.05,
    vpa_abs_tol: float = 0.5,
) -> bool:
    """Return True if two records are compatible for cross-source merge.

    Prefer density when both sides have it; otherwise volume/atom.
    If neither metric is available on both sides, allow the merge (base
    fingerprint already matched composition + SG + nsites).
    """
    da = _finite_float(density_a)
    db = _finite_float(density_b)
    if da is not None and db is not None:
        mean = (da + db) / 2.0
        if mean <= 0:
            return abs(da - db) <= density_abs_tol
        return abs(da - db) <= max(density_abs_tol, density_rel_tol * mean)

    vpa_a = volume_per_atom(volume_a, nsites_a)
    vpa_b = volume_per_atom(volume_b, nsites_b)
    if vpa_a is not None and vpa_b is not None:
        mean = (vpa_a + vpa_b) / 2.0
        if mean <= 0:
            return abs(vpa_a - vpa_b) <= vpa_abs_tol
        return abs(vpa_a - vpa_b) <= max(vpa_abs_tol, vpa_rel_tol * mean)

    # Incomplete metrics: do not block on density/volume alone.
    return True
