"""Shared pytest fixtures: fake ensembles, sample data, and mocked HTTP responses.

None of these fixtures touch the network or load the real (multi-MB) trained models —
tests should run fast and offline.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest


class _StubModel:
    """A minimal stand-in for a trained estimator: predicts a fixed value per row."""

    def __init__(self, value: float = 1.5):
        self.value = value

    def predict(self, X):
        return [self.value] * len(X)


# Small, fast-to-compute Mordred descriptor names used by the fake ensemble metadata.
FAKE_DESCRIPTORS = ["nAtom", "nHeavyAtom"]


@pytest.fixture
def fake_ensemble_dir(tmp_path):
    """Build a tiny 5-model ensemble directory compatible with engine.load_models_and_metadata."""
    model_dir = tmp_path / "fake_ensemble_model"
    model_dir.mkdir()
    for i in range(1, 6):
        joblib.dump(_StubModel(value=1.0 * i), model_dir / f"model_{i}.joblib")
        metadata = {
            "hyperparameters": {"n_estimators": 10},
            "descriptors": FAKE_DESCRIPTORS,
        }
        with open(model_dir / f"metadata_{i}.json", "w") as f:
            json.dump(metadata, f)
    return model_dir


@pytest.fixture
def sample_prediction_df():
    """A tiny mixture-model-shaped input DataFrame."""
    return pd.DataFrame(
        {
            "SMILES": ["CCO.[Cl-]", "C[N+](C)(C)C.[Cl-]"],
            "Mole_fraction": [0.3, 0.5],
            "Temperature": [298.15, 310.0],
        }
    )


@pytest.fixture
def sample_pure_prediction_df():
    """A tiny pure-IL-model-shaped input DataFrame (no mole fraction column)."""
    return pd.DataFrame(
        {
            "SMILES": ["C[N+](C)(C)C.[Cl-]", "CCCC[N+]1=CC=CC=C1.F[B-](F)(F)F"],
            "Temperature": [298.15, 298.15],
        }
    )


@pytest.fixture
def mock_ilthermo_responses(monkeypatch):
    """Stub every HTTP path ionics.client uses so it never makes a real network call.

    Covers both the module-level ``requests.get`` (used by ``getIdsets``) and the pooled
    ``requests.Session`` the concurrent ``download_idsets`` fetches through.
    """
    import requests

    calls = []

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload
            self.content = json.dumps(payload).encode("utf-8")

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, *args, **kwargs):
        calls.append(url)
        if "ilsearch" in url:
            return _FakeResponse({"res": [["12345", "some set"]]})
        if "ilset" in url:
            return _FakeResponse(
                {
                    "dhead": [["Temperature", "K"], ["Pressure", "kPa"], ["Density", "kg/m3"]],
                    "data": [[298.15, 101.3, 1050.2]],
                    "components": [{"idout": "C001", "name": "Test IL", "formula": "C1CCOC1"}],
                    "ref": "Smith, J. et al., 2020",
                }
            )
        raise AssertionError(f"Unexpected URL in test: {url}")

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(
        requests.Session, "get", lambda self, url, *a, **kw: fake_get(url, *a, **kw))
    return calls
