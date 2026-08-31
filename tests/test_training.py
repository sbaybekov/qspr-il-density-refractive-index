import pandas as pd
import pytest

from qspr_il.models.engine import load_models_and_metadata, predict, prepare_input
from qspr_il.models.training import DEFAULT_HYPERPARAMETERS, select_usable_descriptors, train_ensemble

# A handful of real, distinct, valid IL SMILES so Mordred descriptor calculation succeeds and
# GroupKFold has enough unique groups to split on.
_SMILES_POOL = [
    "CCCCn1cc[n+](C)c1.F[B-](F)(F)F",
    "CCN1C=C[N+](=C1)C.F[B-](F)(F)F",
    "CCCCN1C=C[N+](=C1)C.[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "CCCCCCN1C=C[N+](=C1)C.F[P-](F)(F)(F)(F)F",
    "CCCCN1C=C[N+](=C1)C.F[P-](F)(F)(F)(F)F",
    "CCN1C=C[N+](=C1)C.[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
]
_FAST_HYPERPARAMETERS = {"max_depth": 3,
                         "n_estimators": 15, "learning_rate": 0.3}


def _pure_dataset(repeats: int = 3) -> pd.DataFrame:
    smiles = _SMILES_POOL * repeats
    temps = [298.15, 303.15, 308.15] * (len(_SMILES_POOL) * repeats // 3)
    values = [900.0 + 10.0 * i for i in range(len(smiles))]
    return pd.DataFrame({"IL_SMILES": smiles, "Temperature (K)": temps, "Property_value": values})


def _mixture_dataset(repeats: int = 3) -> pd.DataFrame:
    df = _pure_dataset(repeats)
    df["Mole_fraction_IL"] = [0.1 + 0.05 * (i % 15) for i in range(len(df))]
    return df


def test_select_usable_descriptors_drops_missing_and_constant_columns():
    import numpy as np

    matrix = np.array(
        [
            [1.0, np.nan, 5.0],
            [2.0, np.nan, 5.0],
            [3.0, 1.0, 5.0],
        ]
    )
    kept = select_usable_descriptors(
        matrix, ["varies", "mostly_missing", "constant"])
    assert kept == ["varies"]


def test_train_ensemble_pure_produces_requested_number_of_models(tmp_path):
    df = _pure_dataset(repeats=3)
    messages = []
    ensemble, metrics = train_ensemble(
        df,
        property_value_col="Property_value",
        smiles_col="IL_SMILES",
        temp_col="Temperature (K)",
        mole_fraction_col=None,
        output_dir=tmp_path,
        n_models=3,
        hyperparameters=_FAST_HYPERPARAMETERS,
        progress_callback=messages.append,
    )
    assert len(ensemble.models) == 3
    assert len(ensemble.metadata) == 3
    assert len(metrics) == 3
    assert all("rmse" in m and "r2" in m for m in metrics)
    assert any("Computing Mordred descriptors" in msg for msg in messages)
    assert any("Training complete" in msg for msg in messages)

    # saved artifacts are a valid, loadable ensemble
    reloaded = load_models_and_metadata(tmp_path)
    assert len(reloaded.models) == 3
    for meta in reloaded.metadata:
        assert meta["hyperparameters"] == _FAST_HYPERPARAMETERS
        assert isinstance(meta["descriptors"], list) and meta["descriptors"]


def test_train_ensemble_mixture_uses_mole_fraction_feature(tmp_path):
    df = _mixture_dataset(repeats=3)
    ensemble, metrics = train_ensemble(
        df,
        property_value_col="Property_value",
        smiles_col="IL_SMILES",
        temp_col="Temperature (K)",
        mole_fraction_col="Mole_fraction_IL",
        output_dir=tmp_path,
        n_models=2,
        hyperparameters=_FAST_HYPERPARAMETERS,
    )
    assert len(ensemble.models) == 2

    reloaded = load_models_and_metadata(tmp_path)
    data = pd.DataFrame({"SMILES": [_SMILES_POOL[0]], "Temperature": [
                        298.15], "Mole_fraction": [0.3]})
    prepared = prepare_input(data, smiles_col="SMILES",
                             temp_col="Temperature", mole_fraction_col="Mole_fraction")
    result = predict(prepared, reloaded, mole_fraction_col="Mole_fraction")
    assert not result["prediction_mean"].isna().any()


def test_train_ensemble_reduces_model_count_for_few_unique_compounds(tmp_path):
    # Only 2 unique IL SMILES -> GroupKFold can produce at most 2 splits, even though 5 was
    # requested; the saved ensemble must still be a valid, loadable, contiguous run.
    df = pd.DataFrame(
        {
            "IL_SMILES": [_SMILES_POOL[0], _SMILES_POOL[1]] * 4,
            "Temperature (K)": [298.15, 303.15] * 4,
            "Property_value": [900.0 + i for i in range(8)],
        }
    )
    messages = []
    ensemble, metrics = train_ensemble(
        df,
        property_value_col="Property_value",
        smiles_col="IL_SMILES",
        temp_col="Temperature (K)",
        mole_fraction_col=None,
        output_dir=tmp_path,
        n_models=5,
        hyperparameters=_FAST_HYPERPARAMETERS,
        progress_callback=messages.append,
    )
    assert len(ensemble.models) <= 2
    assert any("training" in msg.lower() and "not" in msg.lower()
               for msg in messages) or len(ensemble.models) == 2
    reloaded = load_models_and_metadata(tmp_path)
    assert len(reloaded.models) == len(ensemble.models)


def test_train_ensemble_raises_for_too_few_rows(tmp_path):
    df = _pure_dataset(repeats=1).head(2)
    with pytest.raises(ValueError, match="Not enough"):
        train_ensemble(
            df,
            property_value_col="Property_value",
            smiles_col="IL_SMILES",
            temp_col="Temperature (K)",
            mole_fraction_col=None,
            output_dir=tmp_path,
            hyperparameters=_FAST_HYPERPARAMETERS,
        )


def test_train_ensemble_drops_rows_with_missing_required_values(tmp_path):
    df = _pure_dataset(repeats=3)
    df.loc[0, "Property_value"] = None
    ensemble, _ = train_ensemble(
        df,
        property_value_col="Property_value",
        smiles_col="IL_SMILES",
        temp_col="Temperature (K)",
        mole_fraction_col=None,
        output_dir=tmp_path,
        n_models=2,
        hyperparameters=_FAST_HYPERPARAMETERS,
    )
    assert len(ensemble.models) == 2


def test_default_hyperparameters_used_when_none_given(tmp_path):
    df = _pure_dataset(repeats=3)
    _, _ = train_ensemble(
        df,
        property_value_col="Property_value",
        smiles_col="IL_SMILES",
        temp_col="Temperature (K)",
        mole_fraction_col=None,
        output_dir=tmp_path,
        n_models=2,
        hyperparameters=None,
    )
    import json

    meta = json.load(open(tmp_path / "metadata_1.json"))
    assert meta["hyperparameters"] == DEFAULT_HYPERPARAMETERS
