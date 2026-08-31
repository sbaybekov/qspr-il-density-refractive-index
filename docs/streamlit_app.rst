Streamlit App
=============

A hosted instance is deployed at
https://qspr-il-density-refractive-index.streamlit.app/

:mod:`qspr_il.app` is a Streamlit GUI over the same prediction engine and data
pipeline used by the CLI (:mod:`qspr_il.models.engine`,
:mod:`qspr_il.data.cleaning`). A sidebar selector offers three modes:

* **Run prediction model** -- upload a CSV or enter a single ionic-liquid
  SMILES, and get density/refractive-index predictions (the original app
  behavior).
* **Fetch & clean data** -- pull and curate ILThermo data for *any* of the
  ~55 properties it tracks (not just density/refractive index), any of the
  supported solvent systems or a pure IL, and any temperature/pressure range,
  then download the resulting CSV. An **"Advanced ILThermo search filters
  (pyionics)"** expander adds the same server-side ``ilsearch`` filters the
  ``pyionics`` client exposes -- publication year, author surname, and
  free-text keyword -- each narrowing the query before cleaning. See
  :doc:`data_pipeline`.
* **Both** -- fetch data as above, then, if a trained model exists for that
  property (currently density or refractive index), run it directly on the
  freshly fetched data with one more click, no re-upload needed. If no model
  exists yet, offers to train one on the spot instead (see :doc:`training`),
  with live progress and validation metrics, then optionally run it too.

Every prediction result includes, right below the results table:

* A **CSV download** and a **PDF report download** (a summary, distribution
  and uncertainty charts rendered fresh via matplotlib, and a results table
  -- not a screenshot of the page).
* For the 6 mixture models (density/refractive index x
  water/ethanol/isopropanol -- there's no pure-IL variant), a collapsed
  **"Training data vs. external test set (UMAP)"** expander showing the
  matching pre-generated visualization from :doc:`results`.

Running the app
---------------

Install the ``gui`` extra and start the app::

   python -m pip install -e ".[gui]"
   streamlit run qspr_il/app.py

``qspr_il/app.py`` adds the repository root to ``sys.path`` on startup, so the
same command works as the main module on Streamlit Community Cloud -- point the
app at this repository and set the main module path to ``qspr_il/app.py``.
Streamlit Cloud clones the whole repository, so ``qspr_il`` and its bundled
trained model artifacts are used directly from the checkout.
