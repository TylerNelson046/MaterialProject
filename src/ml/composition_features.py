"""Composition feature vectors for training exports (no matminer dependency)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Stable feature dimension: H (Z=1) … Og (Z=118).
_ELEMENT_SYMBOLS: tuple[str, ...] | None = None


def element_symbols() -> tuple[str, ...]:
    global _ELEMENT_SYMBOLS
    if _ELEMENT_SYMBOLS is None:
        from pymatgen.core import Element

        symbols: list[str] = []
        for z in range(1, 119):
            try:
                symbols.append(Element.from_Z(z).symbol)
            except Exception:  # noqa: BLE001
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


def _parse_amounts(formula: Any) -> dict[str, float]:
    if formula is None or (isinstance(formula, float) and np.isnan(formula)):
        return {}
    text = str(formula).strip()
    if not text:
        return {}
    try:
        from pymatgen.core import Composition

        return {str(el): float(amt) for el, amt in Composition(text).items()}
    except Exception:  # noqa: BLE001
        return {}


def composition_feature_row(formula: Any) -> dict[str, float]:
    """Fractional composition (sums to 1) + cheap composition stats."""
    from pymatgen.core import Element

    amounts = _parse_amounts(formula)
    total = sum(amounts.values())
    out: dict[str, float] = {col: 0.0 for col in FRAC_COLS}
    out.update({col: float("nan") for col in COMP_STAT_COLS})
    if total <= 0:
        out["comp_n_elements"] = 0.0
        return out

    zs: list[float] = []
    masses: list[float] = []
    for el, amt in amounts.items():
        frac = amt / total
        col = f"frac_{el}"
        if col in out:
            out[col] = frac
        try:
            element = Element(el)
            zs.append(float(element.Z) * frac)
            masses.append(float(element.atomic_mass) * frac)
        except Exception:  # noqa: BLE001
            continue

    out["comp_n_elements"] = float(len(amounts))
    if zs:
        out["comp_mean_atomic_number"] = float(sum(zs))
        out["comp_mean_atomic_mass"] = float(sum(masses))
        try:
            out["comp_max_atomic_number"] = float(
                max(Element(el).Z for el in amounts)
            )
        except Exception:  # noqa: BLE001
            out["comp_max_atomic_number"] = float("nan")
    return out


def add_composition_features(frame: pd.DataFrame, *, formula_col: str = "formula_pretty") -> pd.DataFrame:
    """Append frac_* and composition-stat columns derived from formula_pretty."""
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
