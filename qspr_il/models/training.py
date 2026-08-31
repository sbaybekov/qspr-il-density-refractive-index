"""Train a new XGBoost ensemble on curated data for a property with no existing model.

This is **not** a reimplementation of this project's original training methodology (five
independently shuffled dataset variants, each with its own hyperparameter search -- see
README.md) -- that's a much larger undertaking outside this module's scope. It's a practical,
honest alternative: group k-fold cross-validation (grouped by IL SMILES, so no compound leaks
between train and validation) produces several train/validation splits, and one XGBoost
regressor with fixed, reasonable hyperparameters is fit per split. The result has the exact
same shape as the existing trained ensembles (``model_i.joblib`` + ``metadata_i.json``, loaded
the same way by :func:`qspr_il.models.engine.load_models_and_metadata`), so a model trained
here works with the rest of the prediction pipeline unmodified -- it just wasn't tuned as
rigorously as the ones shipped with the project.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from mordred import Calculator, descriptors as mordred_descriptors
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from qspr_il.models.engine import LoadedEnsemble, process_il_smiles_list

DEFAULT_HYPERPARAMETERS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
DEFAULT_N_MODELS = 5

_all_2d_descriptor_names_cache: list[str] | None = None


def all_2d_descriptor_names() -> list[str]:
    """Every 2D Mordred descriptor name, computed once and cached for the process lifetime."""
    global _all_2d_descriptor_names_cache
    if _all_2d_descriptor_names_cache is None:
        _all_2d_descriptor_names_cache = [str(d) for d in Calculator(
            mordred_descriptors, ignore_3D=True).descriptors]
    return _all_2d_descriptor_names_cache


def select_usable_descriptors(
    descriptor_matrix: np.ndarray,
    descriptor_names: list[str],
    max_missing_frac: float = 0.2,
    min_std: float = 1e-8,
) -> list[str]:
    """Drop descriptor columns that are mostly missing or near-constant across the training
    set -- keeps the resulting model from depending on data too sparse or uninformative to
    have learned anything real from.
    """
    df = pd.DataFrame(descriptor_matrix, columns=descriptor_names)
    missing_frac = df.isna().mean()
    std = df.std(skipna=True)
    keep = (missing_frac <= max_missing_frac) & (std > min_std)
    return [name for name, ok in keep.items() if bool(ok)]


def train_ensemble(
    curated_df: pd.DataFrame,
    property_value_col: str,
    smiles_col: str,
    temp_col: str,
    mole_fraction_col: str | None,
    output_dir: str | Path,
    n_models: int = DEFAULT_N_MODELS,
    hyperparameters: dict | None = None,
    progress_callback=None,
) -> tuple[LoadedEnsemble, list[dict]]:
    """Train a new ``n_models``-member XGBoost ensemble and save it under ``output_dir`` in the
    same ``model_i.joblib`` / ``metadata_i.json`` shape :func:`~qspr_il.models.engine.load_models_and_metadata`
    expects.

    ``curated_df`` is typically the output of :func:`qspr_il.data.cleaning.fetch_curated_dataset`.
    ``mole_fraction_col=None`` trains a pure-IL model (no mixture-composition feature), matching
    :func:`qspr_il.models.engine.predict`'s convention. Returns the trained (in-memory) ensemble
    plus a list of per-model validation metrics (RMSE, R2, fold sizes) so the caller can report
    how well it did before trusting it.
    """

    def _report(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    hyperparameters = hyperparameters or DEFAULT_HYPERPARAMETERS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = [property_value_col, smiles_col, temp_col] + \
        ([mole_fraction_col] if mole_fraction_col else [])
    df = curated_df.dropna(subset=required_cols).reset_index(drop=True)
    if len(df) < 4:
        raise ValueError(
            f"Not enough usable rows ({len(df)}) to train a model -- need at least 4.")

    _report(f"Computing Mordred descriptors for {len(df)} rows...")
    smiles_list = df[smiles_col].tolist()
    all_names = all_2d_descriptor_names()
    descriptor_matrix = process_il_smiles_list(smiles_list, all_names)
    usable_descriptors = select_usable_descriptors(
        descriptor_matrix, all_names)
    if not usable_descriptors:
        raise ValueError(
            "No usable Mordred descriptors survived filtering -- check the input SMILES.")
    _report(
        f"Selected {len(usable_descriptors)} usable descriptors (of {len(all_names)} computed).")

    keep_idx = [all_names.index(n) for n in usable_descriptors]
    feature_blocks = [descriptor_matrix[:, keep_idx],
                      df[[temp_col]].to_numpy(dtype=float)]
    if mole_fraction_col:
        feature_blocks.append(df[[mole_fraction_col]].to_numpy(dtype=float))
    X = np.hstack(feature_blocks)
    y = df[property_value_col].to_numpy(dtype=float)
    groups = df[smiles_col].astype(str).to_numpy()

    n_unique_groups = len(set(groups))
    n_splits = min(n_models, n_unique_groups) if n_unique_groups >= 2 else 1
    if n_splits < n_models:
        _report(
            f"Only {n_unique_groups} unique IL SMILES available -- training {n_splits} model(s), not {n_models}.")
    splits = (
        [(np.arange(len(X)), np.arange(len(X)))]
        if n_splits == 1
        else list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    )

    models, metadata_list, metrics = [], [], []
    for i, (train_idx, val_idx) in enumerate(splits, 1):
        _report(
            f"Model {i}/{len(splits)}: fitting XGBoost on {len(train_idx)} row(s)...")
        model = XGBRegressor(**hyperparameters)
        model.fit(X[train_idx], y[train_idx])

        val_pred = model.predict(X[val_idx])
        rmse = float(np.sqrt(np.mean((val_pred - y[val_idx]) ** 2)))
        ss_res = float(np.sum((y[val_idx] - val_pred) ** 2))
        ss_tot = float(np.sum((y[val_idx] - y[val_idx].mean()) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        _report(
            f"Model {i}/{len(splits)}: validation RMSE={rmse:.4g}, R2={r2:.3f}")

        metadata = {"hyperparameters": hyperparameters,
                    "descriptors": usable_descriptors}
        joblib.dump(model, output_dir / f"model_{i}.joblib")
        with open(output_dir / f"metadata_{i}.json", "w") as f:
            json.dump(metadata, f)

        models.append(model)
        metadata_list.append(metadata)
        metrics.append(
            {"model": i, "rmse": rmse, "r2": r2, "n_train": int(
                len(train_idx)), "n_val": int(len(val_idx))}
        )

    _report(
        f"Training complete. Saved {len(models)} model(s) to {output_dir}.")
    return LoadedEnsemble(models=models, metadata=metadata_list), metrics
