Results and Visualizations
============================

``results/`` holds two kinds of generated output:

* ``results/*_prediction.csv`` -- predictions from a CLI or app run (gitignored
  by default; regenerated on demand, see :doc:`engine`).
* ``results/interactive_umap/`` -- one self-contained interactive `Bokeh
  <https://bokeh.org/>`_ scatter plot per model, projecting the Mordred
  descriptor space to 2D with UMAP (``n_neighbors=5``, ``min_dist=0.1``) to
  visually compare the training data against the external test set:

  .. list-table::
     :header-rows: 1
     :widths: 2 3

     * - File
       - Model
     * - ``density_ethanol_neighbors5_dist01.html``
       - Density, IL in ethanol
     * - ``density_isopropanol_neighbors5_dist01.html``
       - Density, IL in isopropanol
     * - ``density_water_neighbors5_dist01.html``
       - Density, IL in water
     * - ``ri_ethanol_neighbors5_dist01.html``
       - Refractive index, IL in ethanol
     * - ``ri_isopropanol_neighbors5_dist01.html``
       - Refractive index, IL in isopropanol
     * - ``ri_water_neighbors5_dist01.html``
       - Refractive index, IL in water

Viewing them
------------

Open any file directly in a browser -- each is fully self-contained (data
embedded inline, only a Bokeh JS CDN script tag as an external dependency).

Or, from the Streamlit app: run any mixture model (density/refractive index
x water/ethanol/isopropanol -- there's no pure-IL variant) via "Run
prediction model" or "Both", and the matching UMAP visualization appears as
a collapsed **"Training data vs. external test set (UMAP)"** expander right
below the prediction results::

   streamlit run qspr_il/app.py

This reads directly from ``results/interactive_umap/`` in the local
repository checkout; it isn't bundled into the installed ``qspr_il`` package
(the files are several MB of static HTML each), so it's only available when
running from a full clone -- not, for example, from the Hugging Face Spaces
deployment described in :doc:`streamlit_app`.
