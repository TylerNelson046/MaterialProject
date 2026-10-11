"""Normalization package: unified schema and source-specific mappers."""

from normalization.schema import (
    CompositionInfo,
    MaterialRecord,
    SourceProperty,
    SourceRecord,
    StructureRef,
    SymmetryInfo,
    make_material_id,
    make_source_property,
)

__all__ = [
    "CompositionInfo",
    "MaterialRecord",
    "SourceProperty",
    "SourceRecord",
    "StructureRef",
    "SymmetryInfo",
    "make_material_id",
    "make_source_property",
]
