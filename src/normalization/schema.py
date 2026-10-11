"""Canonical material schema for the unified internal representation.

Design rules (from README):
- Preserve source provenance (source + source_id + original metadata).
- Never silently overwrite conflicting property values from different sources.
- Track units explicitly on every property value.
- Keep identity separate from source-tagged measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from config.constants import (
    INTERNAL_ID_PREFIX,
    PROPERTY_UNITS,
    DataSource,
    PropertyName,
)


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class CompositionInfo:
    """Normalized composition fields shared across sources."""

    formula_pretty: str
    reduced_formula: str | None = None
    chemsys: str | None = None
    elements: tuple[str, ...] = ()
    nelements: int | None = None
    # Element -> amount mapping, e.g. {"Si": 1.0, "O": 2.0}
    composition: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula_pretty": self.formula_pretty,
            "reduced_formula": self.reduced_formula,
            "chemsys": self.chemsys,
            "elements": list(self.elements),
            "nelements": self.nelements,
            "composition": dict(self.composition),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompositionInfo:
        elements = data.get("elements") or []
        return cls(
            formula_pretty=data["formula_pretty"],
            reduced_formula=data.get("reduced_formula"),
            chemsys=data.get("chemsys"),
            elements=tuple(elements),
            nelements=data.get("nelements"),
            composition={k: float(v) for k, v in (data.get("composition") or {}).items()},
        )


@dataclass(frozen=True)
class SymmetryInfo:
    """Crystal symmetry descriptors when available."""

    crystal_system: str | None = None
    space_group_symbol: str | None = None
    space_group_number: int | None = None
    point_group: str | None = None
    number_of_species: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SymmetryInfo | None:
        if not data:
            return None
        return cls(
            crystal_system=data.get("crystal_system"),
            space_group_symbol=data.get("space_group_symbol"),
            space_group_number=data.get("space_group_number"),
            point_group=data.get("point_group"),
            number_of_species=data.get("number_of_species"),
        )


@dataclass(frozen=True)
class StructureRef:
    """Reference to a stored structure file (CIF/JSON), not the full structure blob.

    Full structures live under data/structures/; the DuckDB row stores the path.
    """

    path: str | None = None
    format: str | None = None  # e.g. "cif", "json"
    nsites: int | None = None
    volume: float | None = None  # Å^3
    density: float | None = None  # g/cm^3
    lattice_a: float | None = None
    lattice_b: float | None = None
    lattice_c: float | None = None
    lattice_alpha: float | None = None
    lattice_beta: float | None = None
    lattice_gamma: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> StructureRef | None:
        if not data:
            return None
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class SourceProperty:
    """A single measured/computed property tagged with its originating source.

    Different databases may report different values for the "same" material.
    Each value is stored independently rather than collapsed to a single "best".
    """

    name: str
    value: float | int | bool | str | None
    unit: str
    source: DataSource
    source_id: str
    # How the value was obtained: database observation vs later ML prediction.
    origin: str = "known"  # known | predicted | generated
    method: str | None = None
    uncertainty: float | None = None

    def __post_init__(self) -> None:
        if self.name in {p.value for p in PropertyName}:
            expected = PROPERTY_UNITS[PropertyName(self.name)]
            if self.unit != expected:
                raise ValueError(
                    f"Unit mismatch for {self.name}: got {self.unit!r}, expected {expected!r}"
                )

    @classmethod
    def from_canonical(
        cls,
        name: PropertyName,
        value: float | int | bool | str | None,
        source: DataSource,
        source_id: str,
        *,
        origin: str = "known",
        method: str | None = None,
        uncertainty: float | None = None,
    ) -> SourceProperty:
        """Build a property using the canonical unit for ``name``."""
        return cls(
            name=name.value,
            value=value,
            unit=PROPERTY_UNITS[name],
            source=source,
            source_id=source_id,
            origin=origin,
            method=method,
            uncertainty=uncertainty,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "source": self.source.value,
            "source_id": self.source_id,
            "origin": self.origin,
            "method": self.method,
            "uncertainty": self.uncertainty,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceProperty:
        return cls(
            name=data["name"],
            value=data.get("value"),
            unit=data["unit"],
            source=DataSource(data["source"]),
            source_id=data["source_id"],
            origin=data.get("origin", "known"),
            method=data.get("method"),
            uncertainty=data.get("uncertainty"),
        )


@dataclass(frozen=True)
class SourceRecord:
    """One observation of a material from a single external database.

    Multiple SourceRecords may later link to one internal material ID via
    entity resolution. Conflicting property values are kept as separate rows.
    """

    source: DataSource
    source_id: str
    composition: CompositionInfo
    properties: tuple[SourceProperty, ...] = ()
    symmetry: SymmetryInfo | None = None
    structure: StructureRef | None = None
    # Opaque source-specific payload retained for provenance / debugging.
    original_metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_at: str = field(default_factory=utc_now_iso)
    # Optional link assigned after entity resolution / material creation.
    material_id: str | None = None

    def property_map(self) -> dict[str, SourceProperty]:
        """Map property name -> SourceProperty for this source record."""
        return {prop.name: prop for prop in self.properties}

    def get_property(self, name: PropertyName | str) -> SourceProperty | None:
        key = name.value if isinstance(name, PropertyName) else name
        return self.property_map().get(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "source_id": self.source_id,
            "material_id": self.material_id,
            "composition": self.composition.to_dict(),
            "properties": [p.to_dict() for p in self.properties],
            "symmetry": self.symmetry.to_dict() if self.symmetry else None,
            "structure": self.structure.to_dict() if self.structure else None,
            "original_metadata": dict(self.original_metadata),
            "retrieved_at": self.retrieved_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceRecord:
        return cls(
            source=DataSource(data["source"]),
            source_id=data["source_id"],
            composition=CompositionInfo.from_dict(data["composition"]),
            properties=tuple(
                SourceProperty.from_dict(p) for p in data.get("properties") or []
            ),
            symmetry=SymmetryInfo.from_dict(data.get("symmetry")),
            structure=StructureRef.from_dict(data.get("structure")),
            original_metadata=dict(data.get("original_metadata") or {}),
            retrieved_at=data.get("retrieved_at") or utc_now_iso(),
            material_id=data.get("material_id"),
        )


@dataclass(frozen=True)
class MaterialRecord:
    """Internal material identity that may aggregate one or more source records.

    Stage 1 typically has one SourceRecord per MaterialRecord (MP only).
    Entity resolution later may attach additional SourceRecords without
    deleting or overwriting their property values.
    """

    material_id: str
    sources: tuple[SourceRecord, ...]
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("MaterialRecord requires at least one SourceRecord")
        if not self.material_id.startswith(f"{INTERNAL_ID_PREFIX}-"):
            raise ValueError(
                f"material_id must start with '{INTERNAL_ID_PREFIX}-', got {self.material_id!r}"
            )

    @property
    def primary_source(self) -> SourceRecord:
        return self.sources[0]

    @property
    def formula_pretty(self) -> str:
        return self.primary_source.composition.formula_pretty

    @property
    def elements(self) -> tuple[str, ...]:
        return self.primary_source.composition.elements

    def all_properties(self) -> tuple[SourceProperty, ...]:
        """Flatten all source-tagged properties (conflicts preserved)."""
        props: list[SourceProperty] = []
        for record in self.sources:
            props.extend(record.properties)
        return tuple(props)

    def properties_for(self, name: PropertyName | str) -> tuple[SourceProperty, ...]:
        key = name.value if isinstance(name, PropertyName) else name
        return tuple(p for p in self.all_properties() if p.name == key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "material_id": self.material_id,
            "sources": [s.to_dict() for s in self.sources],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaterialRecord:
        return cls(
            material_id=data["material_id"],
            sources=tuple(SourceRecord.from_dict(s) for s in data["sources"]),
            created_at=data.get("created_at") or utc_now_iso(),
            updated_at=data.get("updated_at") or utc_now_iso(),
            notes=data.get("notes"),
        )


def make_material_id(index: int) -> str:
    """Format an internal material ID, e.g. MAT-000001."""
    if index < 1:
        raise ValueError("material index must be >= 1")
    return f"{INTERNAL_ID_PREFIX}-{index:06d}"


def make_source_property(
    name: PropertyName,
    value: float | int | bool | str | None,
    source: DataSource,
    source_id: str,
    **kwargs: Any,
) -> SourceProperty | None:
    """Create a canonical SourceProperty, skipping null values."""
    if value is None:
        return None
    return SourceProperty.from_canonical(
        name=name,
        value=value,
        source=source,
        source_id=source_id,
        **kwargs,
    )
