"""Database package for unified DuckDB storage and queries."""

from database.duckdb import MaterialsDatabase
from database.queries import MaterialSearchFilters, database_stats, search_materials

__all__ = [
    "MaterialsDatabase",
    "MaterialSearchFilters",
    "database_stats",
    "search_materials",
]
