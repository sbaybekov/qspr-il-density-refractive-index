"""Registry of the 8 available QSPR prediction models (property x solvent/pure)."""

from dataclasses import dataclass
from pathlib import Path

_MODELS_ROOT = Path(__file__).resolve().parent / "models"


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for one trained model (a property predicted for a solvent system)."""

    key: str
    property_name: str
    property_short: str
    solvent: str
    is_pure: bool
    model_dir: Path
    target_column: str
    description: str
    default_smiles_col: str = "SMILES"
    default_mole_fraction_col: str | None = "Mole_fraction_IL"
    default_temp_col: str | None = None

    @property
    def label(self) -> str:
        return f"{self.property_name} in {self.solvent}"


REGISTRY: dict[str, ModelSpec] = {
    "1": ModelSpec(
        key="1",
        property_name="Refractive index",
        property_short="n",
        solvent="ethanol",
        is_pure=False,
        model_dir=_MODELS_ROOT / "ri_ethanol" / "RI_ethanol_ensemble_model",
        target_column="Refractive index (Na D-line)",
        description="Predict refractive index of an ionic liquid in ethanol.",
    ),
    "2": ModelSpec(
        key="2",
        property_name="Refractive index",
        property_short="n",
        solvent="isopropanol",
        is_pure=False,
        model_dir=_MODELS_ROOT / "ri_isopropanol" / "RI_isopropanol_ensemble_model",
        target_column="Refractive index (Na D-line)",
        description="Predict refractive index of an ionic liquid in isopropanol.",
    ),
    "3": ModelSpec(
        key="3",
        property_name="Refractive index",
        property_short="n",
        solvent="water",
        is_pure=False,
        model_dir=_MODELS_ROOT / "ri_water" / "RI_water_ensemble_model",
        target_column="Refractive index (Na D-line)",
        description="Predict refractive index of an ionic liquid in water.",
    ),
    "4": ModelSpec(
        key="4",
        property_name="Refractive index",
        property_short="n",
        solvent="pure ionic liquid",
        is_pure=True,
        model_dir=_MODELS_ROOT / "ri_pure" / "RI_pure_ensemble_model",
        target_column="Refractive index (Na D-line)",
        description="Predict refractive index of a pure ionic liquid.",
        default_mole_fraction_col=None,
    ),
    "5": ModelSpec(
        key="5",
        property_name="Density",
        property_short="dens",
        solvent="ethanol",
        is_pure=False,
        model_dir=_MODELS_ROOT / "density_ethanol" / "density_ethanol_ensemble_model",
        target_column="Density (kg/m3)",
        description="Predict density of an ionic liquid in ethanol.",
    ),
    "6": ModelSpec(
        key="6",
        property_name="Density",
        property_short="dens",
        solvent="isopropanol",
        is_pure=False,
        model_dir=_MODELS_ROOT / "density_isopropanol" / "density_isopropanol_ensemble_model",
        target_column="Density (kg/m3)",
        description="Predict density of an ionic liquid in isopropanol.",
    ),
    "7": ModelSpec(
        key="7",
        property_name="Density",
        property_short="dens",
        solvent="water",
        is_pure=False,
        model_dir=_MODELS_ROOT / "density_water" / "density_water_ensemble_model",
        target_column="Density (kg/m3)",
        description="Predict density of an ionic liquid in water.",
    ),
    "8": ModelSpec(
        key="8",
        property_name="Density",
        property_short="dens",
        solvent="pure ionic liquid",
        is_pure=True,
        model_dir=_MODELS_ROOT / "density_pure" / "density_pure_ensemble_model",
        target_column="Density (kg/m3)",
        description="Predict density of a pure ionic liquid.",
        default_mole_fraction_col=None,
    ),
}


def get(key: str) -> ModelSpec:
    """Look up a model spec by its registry key ("1".."8")."""
    try:
        return REGISTRY[key]
    except KeyError as exc:
        valid = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown model key {key!r}. Valid keys: {valid}.") from exc


def iter_specs() -> list[ModelSpec]:
    """All model specs, in registry-key order."""
    return [REGISTRY[key] for key in sorted(REGISTRY)]


def find(property_name: str, solvent: str) -> ModelSpec:
    """Look up a model spec by property name and solvent (case-insensitive)."""
    for spec in iter_specs():
        if spec.property_name.lower() == property_name.lower() and spec.solvent.lower() == solvent.lower():
            return spec
    raise KeyError(f"No model registered for property={property_name!r}, solvent={solvent!r}.")
