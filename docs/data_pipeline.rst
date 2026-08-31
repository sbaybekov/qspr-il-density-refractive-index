Data Acquisition and Cleaning
==============================

Training datasets are sourced from the `ILThermo database
<https://ilthermo.boulder.nist.gov/>`_. This project used to depend on an
external tool, `pyIonics <https://github.com/kammmran/pyionics>`_, for that
step; pyIonics is being deprecated as a standalone package, so its
functionality is now vendored permanently inside :mod:`qspr_il.data.ionics`.
A second module, :mod:`qspr_il.data.cleaning`, adds the "duplicate removal,
consistency checks, and standardization" step the project has always
described but never actually implemented in code until now.

This pipeline is **not** limited to density and refractive index -- the two
properties this project has trained models for. Call
:func:`qspr_il.data.cleaning.list_available_properties` for the full list of
~55 ILThermo properties (viscosity, surface tension, thermal conductivity,
and more) that can be fetched and cleaned the same way, over any
temperature/pressure range you choose. The curated output uses a generic
``Property``/``Property_value`` column pair rather than a property-specific
column name, so the same code path handles all of them. The interactive
``python qspr.py`` CLI and the Streamlit app both expose this directly --
see :doc:`engine` and :doc:`streamlit_app`.

.. mermaid::

   graph LR
       subgraph "ILThermo (NIST)"
           ILT["ilthermo.boulder.nist.gov"]
       end

       subgraph "qspr_il.data.ionics"
           SEARCH["getIdsets() / download_idsets()"]
           CONV["convert2csv()"]
           SMI["addSmiles() - keydata lookup"]
       end

       subgraph "qspr_il.data.cleaning"
           RESOLVE["resolve_property_display_name()"]
           PUBCHEM["resolve_smiles_by_name() - PubChem fallback"]
           FORMULA["formula_matches() - reject wrong matches"]
           CLEAN["build_curated_dataset() - dedupe, T/P filter"]
       end

       ILT --> SEARCH --> CONV --> SMI --> CLEAN
       RESOLVE -.-> SEARCH
       SMI -. "unresolved SMILES" .-> PUBCHEM --> FORMULA --> CLEAN
       CLEAN --> OUT["Curated DataFrame - Property / Property_value schema"]

Fetch layer: ``qspr_il.data.ionics``
-------------------------------------

Pure fetch/reshape -- no cleaning logic. Downloads raw measurement data from
ILThermo, flattens it into CSV/TSV, and joins in SMILES strings for known
compound ids.

.. automodule:: qspr_il.data.ionics.client
   :members:
   :undoc-members:
   :show-inheritance:

Cleaning layer: ``qspr_il.data.cleaning``
-------------------------------------------

Deduplication, temperature/pressure filtering, SMILES standardization, and
column normalization into the same shape as ``datasets/training_sets/*.csv``.

.. automodule:: qspr_il.data.cleaning
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

To (re)build a curated CSV for any property/solvent combination::

   from qspr_il.data.cleaning import fetch_curated_dataset, list_available_properties

   list_available_properties()   # -> ['activity', 'density', 'refractive-index', 'viscosity', ...]

   df = fetch_curated_dataset("density", "ethanol")     # density, IL in ethanol
   df = fetch_curated_dataset("refractive-index", None)  # refractive index, pure IL
   df = fetch_curated_dataset("viscosity", "water")      # viscosity, IL in water -- no trained
                                                          # model exists for this yet, but the
                                                          # curated dataset is still produced

``property_query`` can be the hyphenated short name from
:func:`~qspr_il.data.cleaning.list_available_properties` (e.g.
``"refractive-index"``) or ILThermo's own display name (e.g.
``"Refractive index"``) -- it's resolved fuzzily against the live search
results. ``solvent_name`` is one of ``"water"``, ``"ethanol"``,
``"isopropanol"`` (solvents with reliable SMILES resolution -- see
limitations below), or ``None`` for a pure ionic liquid. Pass
``pressure_range`` (or ``None`` to skip pressure filtering entirely),
``temp_range``, or ``property_range`` to narrow the accepted conditions.
``year``, ``author``, and ``keyword`` are the same server-side ILThermo
``ilsearch`` filters ``pyionics`` exposes (a publication year, an author
surname, a free-text keyword); each narrows the query before any cleaning, and
the interactive CLI (``qspr.py``) and Streamlit app both prompt for them.

This module only fetches and cleans data; it does not train any model itself,
and never touches the existing trained ``.joblib`` ensembles under
``qspr_il/models/<name>/``. If the property you fetched has no trained model
yet (anything other than density or refractive index, today), a separate
module -- :mod:`qspr_il.models.training`, see :doc:`training` -- can train a
new one directly on the curated output. Both the CLI's "Both" action and the
Streamlit app's "Both" mode offer this automatically when they hit a property
with no matching model.

Known data-source limitations
------------------------------

Two staleness issues were found (empirically, against the live API) while
building this pipeline:

* The bundled ``property_idsets.csv`` lookup table's internal ILThermo
  property ids no longer match what the live search API expects.
  :func:`qspr_il.data.cleaning.fetch_curated_dataset` works around this by
  searching broadly and filtering client-side on the exact property display
  name instead.
* The bundled ``smiles.csv`` compound-id lookup table has sparse-to-nonexistent
  coverage of actual ionic-liquid compounds (it appears to mostly cover common
  small molecules/solvents). Solvent SMILES are therefore resolved by name
  (:data:`qspr_il.data.cleaning.SOLVENT_SMILES_BY_NAME`) rather than by id,
  and IL SMILES fall back to :func:`~qspr_il.data.cleaning.resolve_smiles_by_name`
  (a PubChem name lookup, cross-checked against ILThermo's own reported
  molecular formula via :func:`~qspr_il.data.cleaning.formula_matches` before
  being trusted -- a single, unambiguous-looking PubChem name match can still
  be the *wrong* compound, confirmed against a real IL salt name). Only a
  compound that fails both the id-based lookup and the formula-checked
  PubChem fallback is dropped; pass ``use_pubchem_fallback=False`` to stay
  fully offline (at the cost of recovering fewer rows).
