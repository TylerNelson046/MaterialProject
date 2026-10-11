"""Stable project constants: sources, properties, units, and version info."""

from __future__ import annotations

from enum import StrEnum

PROJECT_NAME = "materials-predictor"
PROJECT_VERSION = "0.1.0"


class DataSource(StrEnum):
    """External materials databases supported by the project."""

    MATERIALS_PROJECT = "materials_project"
    JARVIS = "jarvis"
    OQMD = "oqmd"
    AFLOW = "aflow"
    COD = "cod"
    ALEXANDRIA = "alexandria"
    DERIVED = "derived"


STAGE1_SOURCES: tuple[DataSource, ...] = (DataSource.MATERIALS_PROJECT,)
PLANNED_SOURCES: tuple[DataSource, ...] = tuple(DataSource)


class PropertyName(StrEnum):
    """Canonical property names used in the unified schema."""

    # Thermodynamic / stability
    FORMATION_ENERGY_PER_ATOM = "formation_energy_per_atom"
    ENERGY_ABOVE_HULL = "energy_above_hull"
    ENERGY_PER_ATOM = "energy_per_atom"
    UNCORRECTED_ENERGY_PER_ATOM = "uncorrected_energy_per_atom"
    EQUILIBRIUM_REACTION_ENERGY_PER_ATOM = "equilibrium_reaction_energy_per_atom"
    IS_STABLE = "is_stable"
    THEORETICAL = "theoretical"

    # Electronic / semiconductor
    BAND_GAP = "band_gap"
    IS_GAP_DIRECT = "is_gap_direct"
    IS_METAL = "is_metal"
    EFERMI = "efermi"
    CBM = "cbm"
    VBM = "vbm"

    # Structure geometry
    DENSITY = "density"
    VOLUME = "volume"
    NSITES = "nsites"

    # Magnetism
    IS_MAGNETIC = "is_magnetic"
    TOTAL_MAGNETIZATION = "total_magnetization"
    TOTAL_MAGNETIZATION_NORMALIZED_VOL = "total_magnetization_normalized_vol"
    NUM_MAGNETIC_SITES = "num_magnetic_sites"
    ORDERING = "ordering"

    # Mechanical / elastic
    BULK_MODULUS_VRH = "bulk_modulus_vrh"
    SHEAR_MODULUS_VRH = "shear_modulus_vrh"
    HOMOGENEOUS_POISSON = "homogeneous_poisson"
    UNIVERSAL_ANISOTROPY = "universal_anisotropy"

    # Dielectric / optical
    DIELECTRIC_TOTAL = "dielectric_total"
    DIELECTRIC_IONIC = "dielectric_ionic"
    DIELECTRIC_ELECTRONIC = "dielectric_electronic"
    REFRACTIVE_INDEX = "refractive_index"

    # Surfaces
    WEIGHTED_SURFACE_ENERGY = "weighted_surface_energy"
    WEIGHTED_WORK_FUNCTION = "weighted_work_function"

    # JARVIS specialty / advanced electronic & functional
    MBJ_BANDGAP = "mbj_bandgap"
    AVG_ELEC_MASS = "avg_elec_mass"
    AVG_HOLE_MASS = "avg_hole_mass"
    SLME = "slme"
    TC_SUPERCON = "tc_supercon"
    EXFOLIATION_ENERGY = "exfoliation_energy"

    # Nuclear screening (element-table derived)
    NEUTRON_ABSORPTION_CROSS_SECTION = "neutron_absorption_cross_section"
    MELTING_POINT_ELEMENT_MIX = "melting_point_element_mix"
    MACROSCOPIC_ABSORPTION = "macroscopic_absorption"
    HAS_STRONG_NEUTRON_ABSORBER = "has_strong_neutron_absorber"
    RADIATION_HARDNESS_SCORE = "radiation_hardness_score"

    # Mechanical (derived)
    YOUNGS_MODULUS_VRH = "youngs_modulus_vrh"

    # Battery / electrode (Materials Project insertion electrodes)
    AVERAGE_VOLTAGE = "average_voltage"
    CAPACITY_GRAV = "capacity_grav"
    CAPACITY_VOL = "capacity_vol"
    ENERGY_GRAV = "energy_grav"
    WORKING_ION = "working_ion"
    BATTERY_FORMULA = "battery_formula"

    # Thermal transport
    THERMAL_CONDUCTIVITY = "thermal_conductivity"
    THERMAL_CONDUCTIVITY_PROXY = "thermal_conductivity_proxy"
    DEBYE_TEMPERATURE = "debye_temperature"
    HEAT_CAPACITY_CV = "heat_capacity_cv"

    # Corrosion / radiation proxies
    CORROSION_RESISTANCE_SCORE = "corrosion_resistance_score"
    OXIDATION_TENDENCY = "oxidation_tendency"
    DISPLACEMENT_ENERGY_MIX = "displacement_energy_mix"
    RADIATION_DAMAGE_SUSCEPTIBILITY = "radiation_damage_susceptibility"

    # Cheap calculated KPIs
    ENERGY_VOL = "energy_vol"
    SPECIFIC_STIFFNESS = "specific_stiffness"
    SPECIFIC_BULK_MODULUS = "specific_bulk_modulus"
    NEUTRON_MEAN_FREE_PATH = "neutron_mean_free_path"
    MACROSCOPIC_ABSORPTION_PER_MASS = "macroscopic_absorption_per_mass"
    MELTING_TO_DENSITY_RATIO = "melting_to_density_ratio"
    THERMAL_TO_DENSITY_RATIO = "thermal_to_density_ratio"
    IS_NEAR_HULL = "is_near_hull"
    IS_SEMICONDUCTOR_WINDOW = "is_semiconductor_window"
    ELASTIC_ANISOTROPY_ABS = "elastic_anisotropy_abs"

    # Provenance / quality flags
    DEPRECATED = "deprecated"


PROPERTY_UNITS: dict[PropertyName, str] = {
    PropertyName.FORMATION_ENERGY_PER_ATOM: "eV/atom",
    PropertyName.ENERGY_ABOVE_HULL: "eV/atom",
    PropertyName.ENERGY_PER_ATOM: "eV/atom",
    PropertyName.UNCORRECTED_ENERGY_PER_ATOM: "eV/atom",
    PropertyName.EQUILIBRIUM_REACTION_ENERGY_PER_ATOM: "eV/atom",
    PropertyName.IS_STABLE: "boolean",
    PropertyName.THEORETICAL: "boolean",
    PropertyName.BAND_GAP: "eV",
    PropertyName.IS_GAP_DIRECT: "boolean",
    PropertyName.IS_METAL: "boolean",
    PropertyName.EFERMI: "eV",
    PropertyName.CBM: "eV",
    PropertyName.VBM: "eV",
    PropertyName.DENSITY: "g/cm^3",
    PropertyName.VOLUME: "Å^3",
    PropertyName.NSITES: "count",
    PropertyName.IS_MAGNETIC: "boolean",
    PropertyName.TOTAL_MAGNETIZATION: "μB",
    PropertyName.TOTAL_MAGNETIZATION_NORMALIZED_VOL: "μB/Å^3",
    PropertyName.NUM_MAGNETIC_SITES: "count",
    PropertyName.ORDERING: "categorical",
    PropertyName.BULK_MODULUS_VRH: "GPa",
    PropertyName.SHEAR_MODULUS_VRH: "GPa",
    PropertyName.HOMOGENEOUS_POISSON: "dimensionless",
    PropertyName.UNIVERSAL_ANISOTROPY: "dimensionless",
    PropertyName.DIELECTRIC_TOTAL: "dimensionless",
    PropertyName.DIELECTRIC_IONIC: "dimensionless",
    PropertyName.DIELECTRIC_ELECTRONIC: "dimensionless",
    PropertyName.REFRACTIVE_INDEX: "dimensionless",
    PropertyName.WEIGHTED_SURFACE_ENERGY: "J/m^2",
    PropertyName.WEIGHTED_WORK_FUNCTION: "eV",
    PropertyName.MBJ_BANDGAP: "eV",
    PropertyName.AVG_ELEC_MASS: "m_e",
    PropertyName.AVG_HOLE_MASS: "m_e",
    PropertyName.SLME: "percent",
    PropertyName.TC_SUPERCON: "K",
    PropertyName.EXFOLIATION_ENERGY: "meV/atom",
    PropertyName.NEUTRON_ABSORPTION_CROSS_SECTION: "barns",
    PropertyName.MELTING_POINT_ELEMENT_MIX: "K",
    PropertyName.MACROSCOPIC_ABSORPTION: "1/cm",
    PropertyName.HAS_STRONG_NEUTRON_ABSORBER: "boolean",
    PropertyName.RADIATION_HARDNESS_SCORE: "dimensionless",
    PropertyName.YOUNGS_MODULUS_VRH: "GPa",
    PropertyName.AVERAGE_VOLTAGE: "V",
    PropertyName.CAPACITY_GRAV: "mAh/g",
    PropertyName.CAPACITY_VOL: "Ah/L",
    PropertyName.ENERGY_GRAV: "Wh/kg",
    PropertyName.WORKING_ION: "categorical",
    PropertyName.BATTERY_FORMULA: "categorical",
    PropertyName.THERMAL_CONDUCTIVITY: "W/m·K",
    PropertyName.THERMAL_CONDUCTIVITY_PROXY: "W/m·K",
    PropertyName.DEBYE_TEMPERATURE: "K",
    PropertyName.HEAT_CAPACITY_CV: "kB/cell",
    PropertyName.CORROSION_RESISTANCE_SCORE: "dimensionless",
    PropertyName.OXIDATION_TENDENCY: "dimensionless",
    PropertyName.DISPLACEMENT_ENERGY_MIX: "eV",
    PropertyName.RADIATION_DAMAGE_SUSCEPTIBILITY: "dimensionless",
    PropertyName.ENERGY_VOL: "Wh/L",
    PropertyName.SPECIFIC_STIFFNESS: "GPa·cm^3/g",
    PropertyName.SPECIFIC_BULK_MODULUS: "GPa·cm^3/g",
    PropertyName.NEUTRON_MEAN_FREE_PATH: "cm",
    PropertyName.MACROSCOPIC_ABSORPTION_PER_MASS: "cm^2/g",
    PropertyName.MELTING_TO_DENSITY_RATIO: "K·cm^3/g",
    PropertyName.THERMAL_TO_DENSITY_RATIO: "W·cm^3/(m·K·g)",
    PropertyName.IS_NEAR_HULL: "boolean",
    PropertyName.IS_SEMICONDUCTOR_WINDOW: "boolean",
    PropertyName.ELASTIC_ANISOTROPY_ABS: "dimensionless",
    PropertyName.DEPRECATED: "boolean",
}

# Numeric properties where cross-source disagreement is scientifically meaningful.
# Policy: never average into search_index; keep all source rows; expose spread in detail views.
CONFLICT_TRACKED_PROPERTIES: tuple[str, ...] = (
    PropertyName.BAND_GAP.value,
    PropertyName.FORMATION_ENERGY_PER_ATOM.value,
    PropertyName.ENERGY_ABOVE_HULL.value,
    PropertyName.DENSITY.value,
    PropertyName.VOLUME.value,
    PropertyName.BULK_MODULUS_VRH.value,
    PropertyName.SHEAR_MODULUS_VRH.value,
    PropertyName.TOTAL_MAGNETIZATION.value,
    PropertyName.MBJ_BANDGAP.value,
)


DOMAIN_PACKS: tuple[str, ...] = (
    "nuclear",
    "batteries",
    "magnets",
    "semiconductors",
    "general",
)

VALIDATION_STATUS = (
    "known",
    "predicted",
    "generated",
    "potentially_novel",
    "validated",
)

INTERNAL_ID_PREFIX = "MAT"
