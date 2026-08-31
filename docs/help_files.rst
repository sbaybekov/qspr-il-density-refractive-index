Command-Line Reference
=======================

The single CLI entry point, :func:`qspr_il.cli.main`, replaces what used to
be 8 separate scripts each with their own ``--help`` text. Run either::

   python qspr.py --help

or, once the package is installed (``pip install -e .``)::

   qspr-il-predict --help

which prints::

   usage: qspr-il-predict [-h] [--action {model,data,both}]
                           [--model {1,2,3,4,5,6,7,8}] [--input_csv INPUT_CSV]
                           [--smiles_col SMILES_COL]
                           [--mole_fraction_col MOLE_FRACTION_COL] [--temp_col TEMP_COL]
                           [--model_dir MODEL_DIR] [--output_csv OUTPUT_CSV]

   Predict density or refractive index of ionic liquids.

   options:
     -h, --help            show this help message and exit
     --action {model,data,both}
                            What to do: run a prediction model, fetch & clean
                            ILThermo data, or both. Defaults to 'model' if --model
                            is given, otherwise you are prompted interactively.
     --model {1,2,3,4,5,6,7,8}
                            Model to run (1-8).
     --input_csv INPUT_CSV
                            Path to the input CSV file (if omitted, you will be prompted).
     --smiles_col SMILES_COL
                            Name of the SMILES column in the input CSV.
     --mole_fraction_col MOLE_FRACTION_COL
                            Name of the mole fraction column (mixture models only).
     --temp_col TEMP_COL   Name of the temperature column.
     --model_dir MODEL_DIR
                            Directory containing the ensemble of models and metadata.
     --output_csv OUTPUT_CSV
                            Path to save the output CSV with predictions.

If ``--action`` is omitted and no ``--model`` is given either, you are asked
interactively whether to run a prediction model, fetch & clean data (for any
ILThermo property -- see :doc:`data_pipeline`), or both. Passing ``--model``
(as in the example below) skips that prompt and goes straight to prediction.

Model numbers
-------------

.. list-table::
   :header-rows: 1
   :widths: 1 3 3

   * - ``--model``
     - Property
     - System
   * - 1
     - Refractive index
     - Ethanol
   * - 2
     - Refractive index
     - Isopropanol
   * - 3
     - Refractive index
     - Water
   * - 4
     - Refractive index
     - Pure ionic liquid
   * - 5
     - Density
     - Ethanol
   * - 6
     - Density
     - Isopropanol
   * - 7
     - Density
     - Water
   * - 8
     - Density
     - Pure ionic liquid

Any option left unset falls back to an interactive prompt (press Enter to
accept the shown default), except ``--mole_fraction_col``, which is never
asked for models 4 and 8 (pure ionic liquids have no mixture composition).

Example
-------

::

   python qspr.py --model 1 \
       --input_csv datasets/external_test_set.csv \
       --smiles_col IL_SMILES \
       --mole_fraction_col Mole_fraction_IL \
       --output_csv results/ri_ethanol_prediction.csv

Output columns are the original input columns plus ``Standardized_IL_SMILES``,
``Changes``, ``prediction_mean``, and ``prediction_std``.
