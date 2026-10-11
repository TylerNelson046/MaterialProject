"""Composition feature vectors (no matminer dependency).

Used by the training-export builder and by LightGBM train/predict:

* ``FRAC_COLS`` — elemental fractions ``frac_H`` … ``frac_Og``
* ``COMP_STAT_COLS`` — cheap composition aggregates
* ``ELEMENT_DESC_COLS`` — weighted elemental descriptors from pymatgen
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

# Stable feature dimension: H (Z=1) … Og (Z=118).
_ELEMENT_SYMBOLS: tuple[str, ...] | None = None

_BLOCK_INDEX = {"s": 0.0, "p": 1.0, "d": 2.0, "f": 3.0}


def element_symbols() -> tuple[str, ...]:
    """Return H…Og symbols (Z=1–118), cached after first call."""
    global _ELEMENT_SYMBOLS
    if _ELEMENT_SYMBOLS is None:
        from pymatgen.core import Element

        symbols: list[str] = []
        for z in range(1, 119):
            try:
                symbols.append(Element.from_Z(z).symbol)
            except Exception:  # noqa: BLE001 — rare Z gaps / pymatgen edge cases
                symbols.append(f"Z{z}")
        _ELEMENT_SYMBOLS = tuple(symbols)
    return _ELEMENT_SYMBOLS


FRAC_COLS: tuple[str, ...] = tuple(f"frac_{sym}" for sym in element_symbols())
COMP_STAT_COLS: tuple[str, ...] = (
    "comp_mean_atomic_number",
    "comp_mean_atomic_mass",
    "comp_max_atomic_number",
    "comp_n_elements",
)
# Weighted elemental descriptors (computed from formula; not required in export parquet).
ELEMENT_DESC_COLS: tuple[str, ...] = (
    "desc_mean_X",
    "desc_mean_atomic_radius",
    "desc_mean_row",
    "desc_mean_group",
    "desc_mean_block",
    "desc_frac_metal",
    "desc_std_atomic_number",
    "desc_mean_ionization_energy",
)


def _parse_amounts(formula: Any) -> dict[str, float]:
    """Parse a formula into ``{element_symbol: amount}``; empty on failure."""
    if formula is None or (isinstance(formula, float) and np.isnan(formula)):
        return {}
    text = str(formula).strip()
    if not text:
        return {}
    try:
        from pymatgen.core import Composition

        return {str(el): float(amt) for el, amt in Composition(text).items()}
    except Exception:  # noqa: BLE001 — invalid / ambiguous formulas
        return {}


def _safe_float(value: Any) -> float | None:
    """Coerce to a finite float, else ``None``."""
    try:
        if value is None:
            return None
        out = float(value)
        if not np.isfinite(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
    """pairs = (weight, value); ignores non-finite values."""
    num = 0.0
    den = 0.0
    for weight, value in pairs:
        if weight <= 0 or not np.isfinite(value):
            continue
        num += weight * value
        den += weight
    if den <= 0:
        return float("nan")
    return float(num / den)


def composition_feature_row(formula: Any) -> dict[str, float]:
    """Fractional composition + composition stats + elemental descriptors."""
    from pymatgen.core import Element

    amounts = _parse_amounts(formula)
    total = sum(amounts.values())
    out: dict[str, float] = {col: 0.0 for col in FRAC_COLS}
    out.update({col: float("nan") for col in COMP_STAT_COLS})
    out.update({col: float("nan") for col in ELEMENT_DESC_COLS})
    if total <= 0:
        out["comp_n_elements"] = 0.0
        out["desc_frac_metal"] = 0.0
        return out

    zs: list[float] = []
    masses: list[float] = []
    z_vals: list[float] = []
    x_pairs: list[tuple[float, float]] = []
    radius_pairs: list[tuple[float, float]] = []
    row_pairs: list[tuple[float, float]] = []
    group_pairs: list[tuple[float, float]] = []
    block_pairs: list[tuple[float, float]] = []
    ie_pairs: list[tuple[float, float]] = []
    metal_weight = 0.0

    for el, amt in amounts.items():
        frac = amt / total
        col = f"frac_{el}"
        if col in out:
            out[col] = frac
        try:
            element = Element(el)
        except Exception:  # noqa: BLE001
            continue

        z = float(element.Z)
        zs.append(z * frac)
        z_vals.append(z)
        masses.append(float(element.atomic_mass) * frac)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            x_val = _safe_float(getattr(element, "X", None))
            radius = _safe_float(getattr(element, "atomic_radius", None))
            if radius is None:
                radius = _safe_float(getattr(element, "atomic_radius_calculated", None))
        if x_val is not None:
            x_pairs.append((frac, x_val))
        if radius is not None:
            radius_pairs.append((frac, radius))

        row = _safe_float(getattr(element, "row", None))
        if row is not None:
            row_pairs.append((frac, row))
        group = _safe_float(getattr(element, "group", None))
        if group is not None:
            group_pairs.append((frac, group))

        block = str(getattr(element, "block", "") or "").lower()
        if block in _BLOCK_INDEX:
            block_pairs.append((frac, _BLOCK_INDEX[block]))

        ie = None
        ion_energies = getattr(element, "ionization_energies", None)
        if ion_energies:
            ie = _safe_float(ion_energies[0] if len(ion_energies) else None)
        if ie is not None:
            ie_pairs.append((frac, ie))

        try:
            if bool(element.is_metal):
                metal_weight += frac
        except Exception:  # noqa: BLE001
            pass

    out["comp_n_elements"] = float(len(amounts))
    if zs:
        out["comp_mean_atomic_number"] = float(sum(zs))
        out["comp_mean_atomic_mass"] = float(sum(masses))
        out["comp_max_atomic_number"] = float(max(z_vals)) if z_vals else float("nan")
        out["desc_std_atomic_number"] = float(np.std(z_vals)) if len(z_vals) > 1 else 0.0

    out["desc_mean_X"] = _weighted_mean(x_pairs)
    out["desc_mean_atomic_radius"] = _weighted_mean(radius_pairs)
    out["desc_mean_row"] = _weighted_mean(row_pairs)
    out["desc_mean_group"] = _weighted_mean(group_pairs)
    out["desc_mean_block"] = _weighted_mean(block_pairs)
    out["desc_mean_ionization_energy"] = _weighted_mean(ie_pairs)
    out["desc_frac_metal"] = float(metal_weight)
    return out


def add_composition_features(
    frame: pd.DataFrame, *, formula_col: str = "formula_pretty"
) -> pd.DataFrame:
    """Append frac_* / composition-stat / descriptor columns from formula_pretty."""
    if formula_col not in frame.columns:
        raise KeyError(f"Missing {formula_col} for composition features")
    formulas = frame[formula_col].tolist()
    cache: dict[Any, dict[str, float]] = {}
    rows: list[dict[str, float]] = []
    for formula in formulas:
        key = formula if not (isinstance(formula, float) and np.isnan(formula)) else None
        if key not in cache:
            cache[key] = composition_feature_row(formula)
        rows.append(cache[key])
    feat = pd.DataFrame(rows, index=frame.index)
    return pd.concat([frame.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)


def add_element_descriptors(
    frame: pd.DataFrame, *, formula_col: str = "formula_pretty"
) -> pd.DataFrame:
    """Append only ELEMENT_DESC_COLS derived from formula_pretty."""
    if formula_col not in frame.columns:
        raise KeyError(f"Missing {formula_col} for element descriptors")
    cache: dict[Any, dict[str, float]] = {}
    rows: list[dict[str, float]] = []
    for formula in frame[formula_col].tolist():
        key = formula if not (isinstance(formula, float) and np.isnan(formula)) else None
        if key not in cache:
            full = composition_feature_row(formula)
            cache[key] = {col: full[col] for col in ELEMENT_DESC_COLS}
        rows.append(cache[key])
    feat = pd.DataFrame(rows, index=frame.index)
    out = frame.copy()
    for col in ELEMENT_DESC_COLS:
        out[col] = feat[col].to_numpy()
    return out
