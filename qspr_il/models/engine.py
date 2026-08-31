"""Consolidated QSPR prediction engine.

Replaces the 8 near-duplicate ``apply_*.py`` scripts that used to live under
``models/<name>/``. All of them shared the same standardization, descriptor
calculation, and ensemble-prediction logic; this module keeps that logic in
one place, parametrized by a :class:`~qspr_il.registry.ModelSpec`.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from mordred import Calculator, descriptors as mordred_descriptors
from mordred.error import Missing
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

if TYPE_CHECKING:
    from qspr_il.registry import ModelSpec

DEFAULT_TEMPERATURE_K = 298.15


def standardize_molecule(smi: str) -> tuple[str, str]:
    """Reionize, normalize functional groups, and strip stereochemistry from a SMILES string.

    Returns ``(standardized_smiles, change_summary)``. Invalid or unparseable SMILES are
    returned unchanged along with a description of what went wrong.
    """
    changes = []
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi, "Invalid SMILES"

    try:
        parts = smi.split(".")
        reionized_parts = []
        for part in parts:
            part_mol = Chem.MolFromSmiles(part)
            if part_mol:
                reionized = rdMolStandardize.Reionize(part_mol)
                if Chem.MolToSmiles(reionized) != Chem.MolToSmiles(part_mol):
                    changes.append("Reionized")
                reionized_parts.append(Chem.MolToSmiles(reionized))
            else:
                reionized_parts.append(part)
        reionized_smi = ".".join(reionized_parts)
        if reionized_smi != smi:
            mol = Chem.MolFromSmiles(reionized_smi)
        else:
            mol = Chem.MolFromSmiles(smi)
    except Exception as e:
        return smi, f"Reionization Error: {e}"

    try:
        normalizer = rdMolStandardize.Normalizer()
        normalized = normalizer.normalize(mol)
        if Chem.MolToSmiles(normalized) != Chem.MolToSmiles(mol):
            changes.append("Functional groups normalized")
        mol = normalized
    except Exception as e:
        return smi, f"Normalization Error: {e}"

    try:
        smi_before_stereo = Chem.MolToSmiles(mol, isomericSmiles=True)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
        smi_after_stereo = Chem.MolToSmiles(mol, isomericSmiles=True)
        if smi_before_stereo != smi_after_stereo:
            changes.append("Stereo removed")
    except Exception as e:
        return smi, f"Stereo Cleanup Error: {e}"

    standardized_smi = Chem.MolToSmiles(mol)
    change_summary = ", ".join(changes) if changes else "No changes"
    return standardized_smi, change_summary


def reorder_charged_species(df: pd.DataFrame, smiles_col: str = "Standardized_IL_SMILES") -> pd.DataFrame:
    """Reorder each multi-component SMILES so cations come before anions."""

    def reorder_smiles(smi):
        parts = smi.split(".")
        cations = []
        anions = []
        for part in parts:
            if re.search(r"\+[0-9]*", part):
                cations.append(part)
            elif re.search(r"\-[0-9]*", part):
                anions.append(part)
        return ".".join(cations + anions)

    df[smiles_col] = df[smiles_col].apply(reorder_smiles)
    return df


@dataclass
class LoadedEnsemble:
    """The 5 trained models + their metadata for one :class:`ModelSpec`."""

    models: list
    metadata: list[dict]


def load_models_and_metadata(model_dir: str | Path) -> LoadedEnsemble:
    """Load an ensemble's ``model_N.joblib`` + ``metadata_N.json`` pairs (N = 1, 2, 3, ...,
    a contiguous run starting at 1) from a directory.

    Ensembles shipped with this project always have exactly 5; a freshly trained one (see
    :mod:`qspr_il.models.training`) may have fewer if there weren't enough unique compounds for
    a full 5-fold split. Raises :class:`FileNotFoundError` if there's no ``model_1.joblib`` at
    all, or if the numbering has a gap (e.g. ``model_1`` and ``model_3`` exist but ``model_2``
    doesn't) -- a valid ensemble is always a contiguous run from 1, never a sparse one.
    """
    model_dir = Path(model_dir)
    indices = [int(m.group(1)) for p in model_dir.glob(
        "model_*.joblib") if (m := re.match(r"model_(\d+)$", p.stem))]
    if not indices:
        raise FileNotFoundError(
            f"No model_N.joblib files found in {model_dir}.")

    models = []
    model_metadata = []
    for i in range(1, max(indices) + 1):
        model_path = model_dir / f"model_{i}.joblib"
        metadata_path = model_dir / f"metadata_{i}.json"
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Model or metadata file for model {i} not found in {model_dir}.")
        models.append(joblib.load(model_path))
        with open(metadata_path, "r") as f:
            model_metadata.append(json.load(f))
    return LoadedEnsemble(models=models, metadata=model_metadata)


@lru_cache(maxsize=16)
def _filtered_calculator(required: tuple[str, ...]) -> Calculator:
    """A Mordred calculator restricted to ``required`` descriptor names.

    Constructing a full ``Calculator`` registers ~1800 descriptors and is by far the
    largest fixed cost of a prediction run -- it used to be paid once per SMILES
    component per ensemble member. Cache the filtered calculators (keyed by the
    canonicalized descriptor set) so it's paid once per distinct set instead. The
    returned calculator is reused read-only across calls.
    """
    wanted = set(required)
    calc = Calculator(mordred_descriptors, ignore_3D=True)
    calc.descriptors = [d for d in calc.descriptors if str(d) in wanted]
    return calc


def _component_descriptor_vectors(components, calc: Calculator) -> dict[str, np.ndarray]:
    """Descriptor vector (in ``calc`` order) for each RDKit-parseable component SMILES.

    ``components`` is deduplicated by the caller; each distinct component is parsed and
    run through Mordred exactly once. Unparseable components are simply absent from the
    returned mapping.
    """
    vectors: dict[str, np.ndarray] = {}
    for smi in components:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        result = calc(mol)
        vectors[smi] = np.fromiter(
            (np.nan if isinstance(v, Missing) else v for v in result.values()),
            dtype=np.float64,
            count=len(calc.descriptors),
        )
    return vectors


def _descriptor_matrix(smiles_list, required_descriptors) -> tuple[np.ndarray, list[str]]:
    """Mean Mordred descriptor vectors for a column of (multi-component) SMILES.

    Returns ``(matrix, names)`` where ``matrix`` has shape ``(len(smiles_list), len(names))``
    and ``names`` is the calculator's own ordering of ``required_descriptors``. Each distinct
    component is standardized and calculated only once regardless of how many rows repeat it
    -- prediction inputs are typically a few ionic liquids swept over many temperature /
    mole-fraction points. Rows with no parseable component come back all-NaN.
    """
    calc = _filtered_calculator(tuple(sorted(set(required_descriptors))))
    names = [str(d) for d in calc.descriptors]

    row_components = [str(s).split(".") for s in smiles_list]
    unique_components = {c for parts in row_components for c in parts}
    vectors = _component_descriptor_vectors(sorted(unique_components), calc)

    matrix = np.full((len(smiles_list), len(names)), np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="Mean of empty slice")
        for i, parts in enumerate(row_components):
            present = [vectors[c] for c in parts if c in vectors]
            if present:
                matrix[i] = np.nanmean(np.vstack(present), axis=0)
    return matrix, names


def calculate_descriptors(smiles: str, required_descriptors: list[str]) -> tuple[np.ndarray | None, list[str] | None]:
    """Compute the mean Mordred descriptor vector across all components of a multi-part SMILES."""
    calc = _filtered_calculator(tuple(sorted(set(required_descriptors))))
    vectors = _component_descriptor_vectors(smiles.split("."), calc)
    if not vectors:
        return None, None
    names = [str(d) for d in calc.descriptors]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="Mean of empty slice")
        mean_descriptor_vector = np.nanmean(np.vstack(list(vectors.values())), axis=0)
    return mean_descriptor_vector, names


def process_il_smiles_list(smiles_list: list[str], required_descriptors: list[str]) -> np.ndarray:
    """Compute descriptor vectors for a whole column of (possibly multi-component) SMILES."""
    matrix, _ = _descriptor_matrix(smiles_list, required_descriptors)
    return matrix


def prepare_input(
    data: pd.DataFrame,
    smiles_col: str,
    temp_col: str | None,
    mole_fraction_col: str | None,
    default_temp: float = DEFAULT_TEMPERATURE_K,
    progress_callback=None,
) -> pd.DataFrame:
    """Standardize SMILES and normalize the Temperature/Mole_fraction columns in place.

    ``mole_fraction_col`` should be ``None`` for pure-IL model specs (no mixture composition).
    ``progress_callback``, if given, is called with a one-line status string as this runs --
    used by the CLI and Streamlit app to show what's happening instead of a blank wait.
    """

    def _report(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    data = data.copy()

    if smiles_col not in data.columns:
        raise ValueError(
            f"SMILES column '{smiles_col}' not found in input data. Available columns: {list(data.columns)}")

    _report(f"Standardizing {len(data)} SMILES...")
    standardized_smiles = []
    changes = []
    for smi in data[smiles_col]:
        standardized_smi, change_summary = standardize_molecule(smi)
        standardized_smiles.append(standardized_smi)
        changes.append(change_summary)
    data["Standardized_IL_SMILES"] = standardized_smiles
    data["Changes"] = changes
    data = reorder_charged_species(data, smiles_col="Standardized_IL_SMILES")
    invalid_count = sum(1 for c in changes if c == "Invalid SMILES")
    _report(
        f"Standardization complete"
        + (f" ({invalid_count} of {len(data)} SMILES could not be parsed)" if invalid_count else "")
        + "."
    )

    if mole_fraction_col is not None:
        if mole_fraction_col not in data.columns:
            raise ValueError(
                f"Mole fraction column '{mole_fraction_col}' not found in input data. "
                f"Available columns: {list(data.columns)}"
            )
        data.rename(columns={mole_fraction_col: "Mole_fraction"}, inplace=True)

    if temp_col is None or temp_col not in data.columns:
        data["Temperature"] = default_temp
    elif data[temp_col].isna().any():
        data["Temperature"] = data[temp_col].fillna(default_temp)
    else:
        data.rename(columns={temp_col: "Temperature"}, inplace=True)

    return data


def predict(
    data: pd.DataFrame,
    ensemble: LoadedEnsemble,
    smiles_col: str = "Standardized_IL_SMILES",
    temp_col: str = "Temperature",
    mole_fraction_col: str | None = "Mole_fraction",
    progress_callback=None,
) -> pd.DataFrame:
    """Run the 5-model ensemble over already-prepared ``data`` and append prediction columns.

    ``mole_fraction_col=None`` skips the mole-fraction feature (pure-IL models). If an
    individual ensemble member fails, its predictions are recorded as NaN so the remaining
    models can still contribute to the consensus. ``progress_callback``, if given, is called
    with a one-line status string per ensemble member -- inference is what varies per member,
    so it's what's reported.

    Mordred descriptors are computed once for the union of every member's required set (and
    once per *distinct* SMILES component within that), then sliced per member -- the shipped
    ensembles all share a single descriptor list, so this replaces N full descriptor passes
    with one.
    """

    def _report(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    data = data.copy()
    all_smiles_list = data[smiles_col].tolist()
    predictions = []
    n_models = len(ensemble.models)

    union_descriptors = sorted(
        {d for md in ensemble.metadata for d in md["descriptors"]})
    n_unique = len({c for s in all_smiles_list for c in str(s).split(".")})
    _report(
        f"Calculating Mordred descriptors for {len(all_smiles_list)} row(s) "
        f"({n_unique} unique component(s), {len(union_descriptors)} descriptor(s))..."
    )
    union_matrix, union_names = _descriptor_matrix(
        all_smiles_list, union_descriptors)
    col_of = {name: j for j, name in enumerate(union_names)}

    temperature_array = data[temp_col].values.reshape(-1, 1)
    mole_fraction_array = (
        data[mole_fraction_col].values.reshape(-1, 1) if mole_fraction_col is not None else None
    )

    for i, (model, metadata) in enumerate(zip(ensemble.models, ensemble.metadata), 1):
        _report(f"Model {i}/{n_models}: predicting...")
        try:
            wanted = set(metadata["descriptors"])
            cols = [col_of[name] for name in union_names if name in wanted]
            feature_arrays = [union_matrix[:, cols], temperature_array]
            if mole_fraction_array is not None:
                feature_arrays.append(mole_fraction_array)
            preds = model.predict(np.hstack(feature_arrays))
            predictions.append(pd.Series(preds))
        except Exception as e:
            print(f"Error processing Model {i}: {e}")
            _report(
                f"Model {i}/{n_models}: prediction failed ({e}) -- recorded as NaN.")
            predictions.append(pd.Series([np.nan] * len(all_smiles_list)))

    _report("Combining ensemble predictions into mean/std...")
    data["prediction_mean"] = pd.concat(
        predictions, axis=1).mean(axis=1).round(4)
    data["prediction_std"] = pd.concat(
        predictions, axis=1).std(axis=1).round(4)
    return data


def run_prediction(
    data: pd.DataFrame,
    spec: "ModelSpec",
    ensemble: LoadedEnsemble | None = None,
    smiles_col: str | None = None,
    temp_col: str | None = None,
    mole_fraction_col: str | None = None,
    progress_callback=None,
) -> pd.DataFrame:
    """Single in-process entry point used by the CLI, the Streamlit app, and tests.

    Any column argument left as ``None`` falls back to ``spec``'s defaults. Pass an
    already-loaded ``ensemble`` (e.g. from a cache) to avoid reloading the joblib files.
    ``progress_callback``, if given, is called with a one-line status string at each stage
    (standardization, ensemble loading, the shared descriptor pass, and per-member inference)
    -- used by the CLI and Streamlit app to show what's happening during a run instead of a
    blank wait, especially for larger inputs where descriptor calculation dominates runtime.
    """

    def _report(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    smiles_col = smiles_col or spec.default_smiles_col
    temp_col = temp_col or spec.default_temp_col
    if spec.is_pure:
        mole_fraction_col = None
    else:
        mole_fraction_col = mole_fraction_col or spec.default_mole_fraction_col

    prepared = prepare_input(
        data,
        smiles_col=smiles_col,
        temp_col=temp_col,
        mole_fraction_col=mole_fraction_col,
        default_temp=DEFAULT_TEMPERATURE_K,
        progress_callback=progress_callback,
    )

    resolved_mole_fraction_col = "Mole_fraction" if mole_fraction_col is not None else None

    if ensemble is None:
        _report(f"Loading trained ensemble from {spec.model_dir}...")
        ensemble = load_models_and_metadata(spec.model_dir)

    return predict(
        prepared,
        ensemble,
        smiles_col="Standardized_IL_SMILES",
        temp_col="Temperature",
        mole_fraction_col=resolved_mole_fraction_col,
        progress_callback=progress_callback,
    )
