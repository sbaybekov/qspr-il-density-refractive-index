Training New Models
=====================

The 8 shipped models (density/refractive index x 4 systems) were trained
using this project's original methodology -- five independently shuffled
dataset variants, each with its own hyperparameter search (see the README's
"Datasets and Models" section). :mod:`qspr_il.models.training` is a smaller,
*honest* alternative for properties that don't have a shipped model yet: it
does **not** reimplement that original methodology. It trains a real,
usable ensemble with the same file shape the rest of the project expects,
using group k-fold cross-validation (grouped by IL SMILES, so no compound
leaks between train and validation) with fixed, reasonable XGBoost
hyperparameters -- and always reports the resulting validation metrics so you
can judge whether to trust it before using it.

.. mermaid::

   graph TD
       subgraph "Curated data (qspr_il.data.cleaning)"
           DF["Property_value, IL_SMILES, Temperature (K)[, Mole_fraction_IL]"]
       end

       subgraph "qspr_il.models.training"
           DESC["Compute all 2D Mordred descriptors"]
           SELECT["select_usable_descriptors() - drop sparse/constant columns"]
           SPLIT["GroupKFold by IL SMILES"]
           FIT["Fit one XGBRegressor per fold"]
       end

       DF --> DESC --> SELECT --> SPLIT --> FIT
       FIT --> OUT["model_1..N.joblib + metadata_1..N.json"]
       FIT --> METRICS["Per-fold RMSE / R2"]

.. automodule:: qspr_il.models.training
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

::

   from qspr_il.data.cleaning import fetch_curated_dataset
   from qspr_il.models.training import train_ensemble

   df = fetch_curated_dataset("viscosity", None)  # no trained model exists for this yet
   ensemble, metrics = train_ensemble(
       df,
       property_value_col="Property_value",
       smiles_col="IL_SMILES",
       temp_col="Temperature (K)",
       mole_fraction_col=None,  # or "Mole_fraction_IL" for a mixture dataset
       output_dir="results/custom_models/viscosity_pure_ensemble_model",
   )
   for m in metrics:
       print(m)  # {"model": 1, "rmse": ..., "r2": ..., "n_train": ..., "n_val": ...}

The saved directory is immediately usable with the rest of the prediction
pipeline -- ``load_models_and_metadata()`` and ``run_prediction()`` (see
:doc:`engine`) don't care whether an ensemble came from this module or was
shipped with the project::

   from qspr_il.models.engine import load_models_and_metadata, run_prediction

   ensemble = load_models_and_metadata("results/custom_models/viscosity_pure_ensemble_model")

:func:`~qspr_il.models.engine.load_models_and_metadata` accepts any
contiguous run of ``model_N.joblib`` files starting at 1 -- not just exactly
5 -- since :func:`~qspr_il.models.training.train_ensemble` trains fewer than
requested when there aren't enough unique compounds for a full k-fold split.

In the CLI and app
-------------------

Both ``python qspr.py``'s "Both" action and the Streamlit app's "Both" mode
offer this automatically: fetch data for a property with no trained model,
and you're asked whether to train one on the spot, then optionally run it
immediately on the same data.

Known limitations
------------------

* Hyperparameters are fixed (:data:`~qspr_il.models.training.DEFAULT_HYPERPARAMETERS`),
  not searched -- pass your own via the ``hyperparameters`` argument if the
  defaults perform poorly for a given property.
* Validation R2 can be legitimately poor or negative for properties with few
  unique compounds or strongly nonlinear temperature dependence (e.g.
  viscosity) -- this is the group k-fold split honestly reporting that the
  model doesn't generalize well to unseen compounds, not a bug. Check the
  reported metrics before trusting a model trained this way.
