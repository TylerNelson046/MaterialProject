"""Data ingestion package."""

from ingestion.ingest_aflow import ingest_aflow
from ingestion.ingest_cod import ingest_cod
from ingestion.ingest_mp import ingest_materials_project
from ingestion.ingest_jarvis import ingest_jarvis
from ingestion.ingest_oqmd import ingest_oqmd

__all__ = [
    "ingest_aflow",
    "ingest_cod",
    "ingest_materials_project",
    "ingest_jarvis",
    "ingest_oqmd",
]
