Dataset Statistics
==================

The tables below summarize the CSV files currently stored in ``datasets/``.
Counts and ranges were calculated directly from the repository files. A range
is shown as minimum--maximum; missing values count empty CSV cells.

Training datasets
-----------------

.. list-table:: Training-set dimensions and coverage
   :header-rows: 1
   :widths: 28 10 10 14 18 18 14

   * - Dataset
     - Rows
     - Columns
     - Unique IL SMILES
     - Temperature (K)
     - Target range
     - Missing cells
   * - ``density_ethanol.csv``
     - 7,439
     - 26
     - 97
     - 278.15--348.15
     - 739.8--1538.23 kg/m3
     - 35,809
   * - ``density_isopropanol.csv``
     - 1,804
     - 26
     - 32
     - 278.15--338.15
     - 744.02--1474.44 kg/m3
     - 8,837
   * - ``density_pure.csv``
     - 31,935
     - 16
     - 1,503
     - 253.15--573.00
     - 900.1--2150 kg/m3
     - 532
   * - ``density_water.csv``
     - 20,558
     - 26
     - 296
     - 269.10--373.20
     - 844.5--1654 kg/m3
     - 79,429
   * - ``ri_ethanol.csv``
     - 1,182
     - 27
     - 38
     - 278.15--343.15
     - 1.34252--1.55065
     - 6,449
   * - ``ri_isopropanol.csv``
     - 734
     - 27
     - 17
     - 288.15--338.15
     - 1.35599--1.50285
     - 4,102
   * - ``ri_pure.csv``
     - 6,649
     - 17
     - 566
     - 278.15--368.10
     - 1.33517--1.7
     - 6,678
   * - ``ri_water.csv``
     - 4,985
     - 27
     - 128
     - 288.15--353.15
     - 1.325--1.5824
     - 19,056

Mixture composition
-------------------

The mixture training files contain a ``Mole fraction`` column ranging from
0 to 1. Their ``Weight fraction`` columns also range from 0 to 1 where values
are available, but many cells are empty: 7,379 in density-ethanol,
15,943 in density-water, 1,114 in refractive-index ethanol, 721 in
refractive-index isopropanol, and 2,429 in refractive-index water.

Pure ionic-liquid datasets do not contain mixture composition columns.

External test set
-----------------

``datasets/external_test_set.csv`` contains 86 rows and 13 columns, covering
31 unique IL SMILES and three solvents: water, ethanol, and isopropanol. The
file contains no missing cells. Its observed target ranges are:

* Density: 777.3--1016 kg/m3
* Refractive index: 1.3509--1.427

The external file does not provide a temperature column. The application
scripts therefore use the documented default temperature of 298.15 K unless a
different temperature column is supplied by the user.

Interpretation
--------------

The training row count is the number of measurements, not the number of unique
ionic liquids. Repeated IL SMILES can occur at different temperatures or
compositions. Missing-cell counts include all columns, including metadata
fields that are not required by the prediction scripts.

Provenance
----------

These CSVs were curated from the ILThermo database. A reproducible,
configurable version of that fetch-and-clean process is now implemented in
:mod:`qspr_il.data.ionics` and :mod:`qspr_il.data.cleaning` -- see
:doc:`data_pipeline` for how to (re)generate a dataset shaped like the ones
summarized above, and for known limitations of the underlying data source.

The static training-set schema and the pipeline's generic curated-output
schema line up directly (``Property``/``Property_value`` replaces the
static files' fixed ``Density (kg/m3)`` / ``Refractive index (Na D-line)``
column, since :func:`~qspr_il.data.cleaning.fetch_curated_dataset` isn't
limited to those two properties):

.. mermaid::

   classDiagram
       class TrainingSetCSV {
           &lt;&lt;datasets/training_sets/*.csv&gt;&gt;
           setid
           Standardized_IL_SMILES
           Temperature (K)
           Pressure (kPa)
           Mole fraction
           Density (kg/m3) or Refractive index (Na D-line)
           Record_ID
           Data_quality_flag
       }
       class CuratedOutput {
           &lt;&lt;qspr_il.data.cleaning.fetch_curated_dataset()&gt;&gt;
           setid
           Standardized_IL_SMILES
           Temperature (K)
           Pressure (kPa)
           Mole fraction
           Mole_fraction_IL
           Property
           Property_value
           Record_ID
           Data_quality_flag
       }
       TrainingSetCSV <|-- CuratedOutput : same shape, generic property columns