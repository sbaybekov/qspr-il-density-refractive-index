Streamlit App
=============

:mod:`qspr_il.app` is a Streamlit GUI over the same prediction engine and data
pipeline used by the CLI (:mod:`qspr_il.models.engine`,
:mod:`qspr_il.data.cleaning`). A sidebar selector offers three modes:

* **Run prediction model** -- upload a CSV or enter a single ionic-liquid
  SMILES, and get density/refractive-index predictions (the original app
  behavior).
* **Fetch & clean data** -- pull and curate ILThermo data for *any* of the
  ~55 properties it tracks (not just density/refractive index), any of the
  supported solvent systems or a pure IL, and any temperature/pressure range,
  then download the resulting CSV. See :doc:`data_pipeline`.
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

Running locally
----------------

Install the ``gui`` extra and start the app::

   python -m pip install -e ".[gui]"
   streamlit run qspr_il/app.py

Deploying to Hugging Face Spaces
----------------------------------

The app is deployed to `Hugging Face Spaces <https://huggingface.co/spaces>`_
rather than GitHub, using the standalone export in ``huggingface_space/`` at
the repository root:

.. code-block:: text

   huggingface_space/
     README.md          # Spaces config frontmatter (sdk: streamlit, app_file: app.py, ...)
     app.py              # thin shim: from qspr_il.app import main; main()
     requirements.txt    # streamlit + qspr_il installed from this GitHub repo

A separate folder is used (rather than syncing the whole repository as a
Space) because Hugging Face Spaces reads its configuration from a
``README.md`` at the Space's own root -- reusing the project's real
``README.md`` for that would overwrite its purpose as project documentation.

To deploy:

1. Create a new Space on Hugging Face with the Streamlit SDK.
2. Push the contents of ``huggingface_space/`` to the Space's git remote, e.g.::

      git clone https://huggingface.co/spaces/<user>/<space-name> hf-space
      cp -r huggingface_space/* hf-space/
      cd hf-space
      git add . && git commit -m "Deploy QSPR Streamlit app" && git push

The Space's ``requirements.txt`` installs ``qspr_il`` directly from this
GitHub repository (including its bundled trained model artifacts, shipped as
package data), so the trained models are never duplicated into the Space
export itself.
