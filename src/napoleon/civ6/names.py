"""Civ VI type names to the short names agents see, and back."""

from __future__ import annotations

PREFIXES = (
    "TECHNOLOGY_",
    "CIVIC_",
    "UNIT_",
    "BUILDING_",
    "DISTRICT_",
    "PROJECT_",
    "TERRAIN_",
    "FEATURE_",
    "RESOURCE_",
)


def short(type_name: str | None) -> str | None:
    """``UNIT_WARRIOR`` -> ``warrior``; ``TERRAIN_GRASS_HILLS`` -> ``grass_hills``."""
    if not type_name or type_name.upper() in ("NONE", "NOTHING"):
        return None
    name = type_name.upper()
    for prefix in PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.lower()


def tech_type(name: str) -> str:
    return "TECHNOLOGY_" + name.upper()


def civic_type(name: str) -> str:
    return "CIVIC_" + name.upper()
