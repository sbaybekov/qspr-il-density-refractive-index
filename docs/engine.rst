Prediction Engine
==================

All 8 prediction models (2 properties x 4 solvent systems) share a single
implementation: :mod:`qspr_il.models.engine`, parametrized by a
:class:`~qspr_il.registry.ModelSpec` from :mod:`qspr_il.registry`. This
replaces what used to be 8 near-identical, independently-maintained scripts.

.. mermaid::

   graph TD
       subgraph Input
           A["IL SMILES"]
           B["Temperature (K)"]
           C["Mole fraction (mixture models only)"]
       end

       subgraph "qspr_il.models.engine"
           D["standardize_molecule()"]
           E["reorder_charged_species()"]
           F["calculate_descriptors() - Mordred"]
           G["predict() - 5-model ensemble"]
       end

       subgraph "qspr_il/models/&lt;name&gt;/*_ensemble_model/"
           H["model_1..5.joblib"]
           I["metadata_1..5.json"]
       end

       A --> D --> E --> F --> G
       B --> G
       C --> G
       H -.-> G
       I -.-> G
       G --> J["prediction_mean / prediction_std"]

Registry
--------

The registry is a simple cross-product of 2 properties and 4 systems, each
mapped to one trained ensemble directory:

.. mermaid::

   graph TD
       subgraph Properties
           DENS["Density"]
           RI["Refractive index"]
       end

       subgraph Systems
           PURE["Pure IL"]
           WATER["+ water"]
           ETOH["+ ethanol"]
           IPA["+ isopropanol"]
       end

       DENS --> PURE & WATER & ETOH & IPA
       RI --> PURE & WATER & ETOH & IPA
       PURE & WATER & ETOH & IPA --> REG["qspr_il.registry.REGISTRY - 8 ModelSpecs"]

.. automodule:: qspr_il.registry
   :members:
   :undoc-members:
   :show-inheritance:

Engine
------

``standardize_molecule()`` standardizes each (possibly multi-component) IL
SMILES before descriptor calculation:

.. mermaid::

   graph TD
       Input["Input SMILES"] --> Split["Split on '.' (components)"]
       Split --> Reion["rdMolStandardize.Reionize() per component"]
       Reion --> Norm["rdMolStandardize.Normalizer()"]
       Norm --> Stereo["Remove stereochemistry"]
       Stereo --> Rejoin["Rejoin components"]
       Rejoin --> Reorder["reorder_charged_species() - cations before anions"]
       Reorder --> Out["Standardized_IL_SMILES + Changes summary"]

``load_models_and_metadata()`` reads one ensemble directory into a
:class:`~qspr_il.models.engine.LoadedEnsemble`:

.. mermaid::

   classDiagram
       class ModelDirectory {
           &lt;&lt;qspr_il/models/&lt;name&gt;/*_ensemble_model/&gt;&gt;
           model_1.joblib .. model_5.joblib
           metadata_1.json .. metadata_5.json
       }
       class LoadedEnsemble {
           &lt;&lt;qspr_il.models.engine&gt;&gt;
           models: list~XGBRegressor~
           metadata: list~dict~
       }
       class ModelMetadata {
           hyperparameters
           descriptors: list~str~
       }
       ModelDirectory --> LoadedEnsemble : load_models_and_metadata()
       LoadedEnsemble --> ModelMetadata

.. automodule:: qspr_il.models.engine
   :members:
   :undoc-members:
   :show-inheritance:

Command-line interface
-----------------------

.. automodule:: qspr_il.cli
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

From the project root::

   python qspr.py

...or, once the package is installed (``pip install -e .``), the console
script::

   qspr-il-predict --model 5 --input_csv datasets/external_test_set.csv \
       --smiles_col IL_SMILES --mole_fraction_col Mole_fraction_IL

Both are equivalent thin wrappers around :func:`qspr_il.cli.main`, which in
turn calls :func:`qspr_il.models.engine.run_prediction` in-process -- there is
no subprocess dispatch and no ``sys.path`` manipulation involved.

Running with no flags at all asks what to do first::

   $ python qspr.py

   What would you like to do?
     1. Run a prediction model (density / refractive index)
     2. Fetch & clean ILThermo data for any property
     3. Both -- fetch data, then run the matching prediction model if one exists
   Choice [1]:

Option 2 walks through :doc:`data_pipeline` interactively (property, system,
temperature/pressure range, output path) without touching prediction at all.
Option 3 does the same, then -- if the fetched property has a trained model
(currently density or refractive index) -- offers to run it immediately on
the freshly fetched, in-memory data. If it doesn't, it offers to train one on
the spot instead (see :doc:`training`), then optionally run that. Passing
``--model`` (as in the example above) skips this prompt and goes straight to
prediction, so existing scripted/non-interactive usage is unaffected. The
Streamlit app (:doc:`streamlit_app`) exposes the same three choices as a
sidebar selector.
