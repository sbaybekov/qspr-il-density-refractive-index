"""Streamlit GUI for the QSPR density/refractive-index prediction models.

Run locally or on Streamlit Community Cloud with::

    streamlit run qspr_il/app.py

See :doc:`/streamlit_app` in the Sphinx docs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# When Streamlit runs this file as a script (``streamlit run qspr_il/app.py``),
# only this file's directory -- not the repo root -- is placed on ``sys.path``,
# so ``import qspr_il`` would fail. Add the repo root before importing anything
# from the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from qspr_il.models.engine import load_models_and_metadata, run_prediction
from qspr_il.registry import ModelSpec, iter_specs
from qspr_il.registry import find as find_spec

_BANNER_IMAGE = Path(__file__).resolve().parent / "assets" / "il_github.png"
_UMAP_DIR = Path(__file__).resolve().parent.parent / \
    "results" / "interactive_umap"
DEFAULT_TEMP_RANGE = (253.0, 573.0)
DEFAULT_PRESSURE_RANGE = (90.0, 110.0)
KNOWN_DATA_SOLVENTS = ["Pure ionic liquid",
                       "water", "ethanol", "isopropanol", "Other..."]


@st.cache_resource(show_spinner="Loading trained ensemble...")
def _load_ensemble(model_dir: str):
    return load_models_and_metadata(model_dir)


def _select_spec() -> ModelSpec:
    specs = iter_specs()
    properties = sorted({s.property_name for s in specs})
    property_name = st.sidebar.selectbox("Property", properties)
    solvents = [s.solvent for s in specs if s.property_name == property_name]
    solvent = st.sidebar.selectbox("System", solvents)
    return next(s for s in specs if s.property_name == property_name and s.solvent == solvent)


def _run_prediction_with_status(
    data: pd.DataFrame,
    spec: ModelSpec,
    ensemble,
    smiles_col: str,
    temp_col: str | None,
    mole_fraction_col: str | None,
    label: str,
) -> pd.DataFrame | None:
    """Run a prediction with a live step-by-step status log instead of a blank wait.

    Shows what's actually happening (standardization, ensemble loading, per-model descriptor
    calculation and inference) and, on failure, a clean error plus a "full details" expander.
    Returns the result DataFrame, or ``None`` if prediction failed (the caller should stop).
    """
    status = st.status(label, expanded=True)

    def _on_progress(message: str) -> None:
        status.write(message)

    try:
        result = run_prediction(
            data,
            spec,
            ensemble=ensemble,
            smiles_col=smiles_col,
            temp_col=temp_col,
            mole_fraction_col=mole_fraction_col,
            progress_callback=_on_progress,
        )
    except Exception as e:
        status.update(label="Prediction failed", state="error")
        st.error(f"Prediction failed: {e}")
        with st.expander("Full error details (for debugging)"):
            st.exception(e)
        return None

    status.update(
        label=f"Done: {len(result)} row(s) predicted.", state="complete")
    return result


def _render_report(result: pd.DataFrame, spec: ModelSpec) -> None:
    """Summary analysis of a batch prediction run: KPIs, distribution, and uncertainty."""
    st.subheader("Prediction report")

    valid = result["prediction_mean"].notna()
    invalid_smiles = (result.get("Changes") == "Invalid SMILES") if "Changes" in result.columns else pd.Series(
        False, index=result.index
    )
    std_series = result.loc[valid, "prediction_std"]
    high_uncertainty_threshold = std_series.quantile(
        0.75) if len(std_series) else float("nan")

    cols = st.columns(4)
    cols[0].metric("Rows predicted", f"{int(valid.sum())} / {len(result)}")
    cols[1].metric(f"Mean {spec.target_column}",
                   f"{result.loc[valid, 'prediction_mean'].mean():.4f}" if valid.any() else "n/a")
    cols[2].metric("Mean ensemble std",
                   f"{std_series.mean():.4f}" if len(std_series) else "n/a")
    cols[3].metric("Unparseable SMILES", int(invalid_smiles.sum()))

    if valid.sum() == 0:
        st.warning(
            "No valid predictions to analyze -- check the SMILES/column settings above.")
        return

    st.markdown(f"**Distribution of predicted {spec.target_column}**")
    counts, bin_edges = np.histogram(
        result.loc[valid, "prediction_mean"], bins=min(20, max(3, valid.sum())))
    hist_df = pd.DataFrame(
        {"count": counts},
        index=[
            f"{bin_edges[i]:.3g}-{bin_edges[i + 1]:.3g}" for i in range(len(bin_edges) - 1)],
    )
    st.bar_chart(hist_df)

    st.markdown("**Prediction vs. ensemble uncertainty**")
    st.scatter_chart(result.loc[valid],
                     x="prediction_mean", y="prediction_std")

    top_uncertain = result.loc[valid].sort_values(
        "prediction_std", ascending=False).head(5)
    with st.expander(f"5 highest-uncertainty predictions (std > {high_uncertainty_threshold:.4g})"):
        st.dataframe(top_uncertain)

    if invalid_smiles.any():
        with st.expander(f"{int(invalid_smiles.sum())} row(s) with unparseable SMILES (excluded above)"):
            st.dataframe(result.loc[invalid_smiles])


_PDF_TABLE_COLUMNS = [
    "Standardized_IL_SMILES",
    "Temperature",
    "Mole_fraction",
    "prediction_mean",
    "prediction_std",
    "Changes",
]
_PDF_MAX_TABLE_ROWS = 200


def _build_pdf_report(result: pd.DataFrame, spec: ModelSpec) -> bytes:
    """Render a PDF with a summary, distribution/uncertainty charts (matplotlib, rendered
    fresh -- not a screenshot), and the results as a table."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    valid = result["prediction_mean"].notna()
    styles = getSampleStyleSheet()
    elements = [
        Paragraph(f"QSPR Predictions: {spec.label}", styles["Title"]),
        Paragraph(spec.description, styles["Normal"]),
        Spacer(1, 0.15 * inch),
        Paragraph(
            f"Rows predicted: {int(valid.sum())} / {len(result)}", styles["Normal"]),
    ]
    if valid.any():
        elements.append(
            Paragraph(
                f"Mean {spec.target_column}: {result.loc[valid, 'prediction_mean'].mean():.4f} "
                f"(mean ensemble std {result.loc[valid, 'prediction_std'].mean():.4f})",
                styles["Normal"],
            )
        )
    elements.append(Spacer(1, 0.2 * inch))

    chart_width, chart_height = 6.5 * inch, 3.4 * inch
    if int(valid.sum()) >= 2:
        fig1, ax1 = plt.subplots(figsize=(6.5, 3.4))
        ax1.hist(result.loc[valid, "prediction_mean"], bins=min(
            20, max(3, int(valid.sum()))), color="#4C78A8")
        ax1.set_title(f"Distribution of predicted {spec.target_column}")
        ax1.set_xlabel(spec.target_column)
        ax1.set_ylabel("Count")
        fig1.tight_layout()
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png", dpi=150)
        plt.close(fig1)
        buf1.seek(0)
        elements += [Image(buf1, width=chart_width,
                           height=chart_height), Spacer(1, 0.15 * inch)]

        fig2, ax2 = plt.subplots(figsize=(6.5, 3.4))
        ax2.scatter(
            result.loc[valid, "prediction_mean"], result.loc[valid, "prediction_std"], s=14, alpha=0.7, color="#F58518"
        )
        ax2.set_title("Prediction vs. ensemble uncertainty")
        ax2.set_xlabel(spec.target_column)
        ax2.set_ylabel("Ensemble std")
        fig2.tight_layout()
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", dpi=150)
        plt.close(fig2)
        buf2.seek(0)
        elements += [Image(buf2, width=chart_width,
                           height=chart_height), Spacer(1, 0.2 * inch)]

    elements.append(Paragraph("Results", styles["Heading2"]))
    table_cols = [c for c in _PDF_TABLE_COLUMNS if c in result.columns]
    table_df = result[table_cols].head(_PDF_MAX_TABLE_ROWS)
    table_data = [table_cols] + table_df.astype(str).values.tolist()
    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C78A8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#F0F0F0")]),
            ]
        )
    )
    elements.append(table)
    if len(result) > _PDF_MAX_TABLE_ROWS:
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(
            Paragraph(
                f"... {len(result) - _PDF_MAX_TABLE_ROWS} more row(s) not shown "
                "(see the CSV download for the full result).",
                styles["Italic"],
            )
        )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            title=f"{spec.label} - QSPR predictions")
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def _render_download_buttons(result: pd.DataFrame, spec: ModelSpec, file_stub: str) -> None:
    """CSV + PDF download buttons for a prediction result (table + charts in the PDF)."""
    st.download_button(
        "Download predictions CSV",
        result.to_csv(index=False),
        file_name=f"{file_stub}.csv",
        mime="text/csv",
    )
    try:
        pdf_bytes = _build_pdf_report(result, spec)
    except Exception as e:
        st.caption(f"PDF report unavailable: {e}")
        return
    st.download_button(
        "Download PDF report",
        pdf_bytes,
        file_name=f"{file_stub}.pdf",
        mime="application/pdf",
    )


def _umap_file_for_spec(spec: ModelSpec) -> Path | None:
    """The pre-generated UMAP visualization matching ``spec``, if one exists.

    Only the 6 mixture models (density/RI x water/ethanol/isopropanol) have one -- there's no
    pure-IL variant in ``results/interactive_umap/``.
    """
    if spec.is_pure:
        return None
    property_token = "density" if spec.property_name == "Density" else "ri"
    path = _UMAP_DIR / \
        f"{property_token}_{spec.solvent.lower()}_neighbors5_dist01.html"
    return path if path.exists() else None


def _render_umap_expander(spec: ModelSpec) -> None:
    """Show the matching UMAP visualization (training data vs. external test set) for this
    model's predictions, collapsed by default so the heavy HTML file isn't loaded unless asked
    for."""
    umap_file = _umap_file_for_spec(spec)
    if umap_file is None:
        return
    with st.expander("Training data vs. external test set (UMAP)", expanded=False):
        st.caption(
            "2D UMAP projection (n_neighbors=5, min_dist=0.1) of the Mordred descriptor space "
            f"for {spec.label}."
        )
        try:
            st.iframe(umap_file, height=700)
        except Exception as e:
            st.error(f"Could not load {umap_file.name}: {e}")


def _render_auto_train(df: pd.DataFrame, resolved_property: str, solvent_name: str | None) -> None:
    """No trained model exists for this property -- offer to train a new one on the data just
    fetched (group k-fold XGBoost ensemble, see qspr_il.models.training), then optionally run
    it immediately on the same data.
    """
    st.info(
        f"No trained prediction model exists yet for '{resolved_property}'.")
    system_label = solvent_name or "pure"
    default_dir = f"results/custom_models/{resolved_property.lower().replace(' ', '_')}_{system_label}_ensemble_model"

    col1, col2 = st.columns([1, 2])
    with col1:
        n_models = st.number_input(
            "Ensemble members to train", min_value=1, max_value=5, value=5)
    with col2:
        output_dir = st.text_input("Save trained model to", value=default_dir)

    if st.button("Train a new model on this data", type="primary"):
        from qspr_il.models.training import train_ensemble

        mole_fraction_col = None if solvent_name is None else "Mole_fraction_IL"
        status = st.status(
            f"Training a new model for '{resolved_property}'...", expanded=True)

        def _on_progress(message: str) -> None:
            status.write(message)

        try:
            ensemble, metrics = train_ensemble(
                df,
                property_value_col="Property_value",
                smiles_col="IL_SMILES",
                temp_col="Temperature (K)",
                mole_fraction_col=mole_fraction_col,
                output_dir=output_dir,
                n_models=int(n_models),
                progress_callback=_on_progress,
            )
        except Exception as e:
            status.update(label="Training failed", state="error")
            st.error(f"Training failed: {e}")
            with st.expander("Full error details (for debugging)"):
                st.exception(e)
            return

        status.update(
            label=f"Done: trained {len(ensemble.models)} model(s).", state="complete")
        spec = ModelSpec(
            key="custom",
            property_name=resolved_property,
            property_short="",
            solvent=solvent_name or "pure ionic liquid",
            is_pure=solvent_name is None,
            model_dir=Path(output_dir),
            target_column=resolved_property,
            description=f"Custom-trained model for '{resolved_property}'"
            + (f" in {solvent_name}." if solvent_name else " (pure ionic liquid)."),
            default_mole_fraction_col=mole_fraction_col,
        )
        st.session_state["trained_model"] = (
            ensemble, spec, mole_fraction_col, metrics, resolved_property, solvent_name)

    cached = st.session_state.get("trained_model")
    if cached is None:
        return
    ensemble, spec, mole_fraction_col, metrics, cached_property, cached_solvent = cached
    if cached_property != resolved_property or cached_solvent != solvent_name:
        return  # stale -- from a different property/solvent fetch

    st.markdown(
        "**Validation metrics** (group k-fold by IL SMILES, not this project's original tuning methodology)")
    st.dataframe(pd.DataFrame(metrics))

    if st.button(f"Run this newly trained model on the fetched data", type="primary"):
        result = _run_prediction_with_status(
            df, spec, ensemble, "IL_SMILES", "Temperature (K)", mole_fraction_col, "Running newly trained model..."
        )
        if result is None:
            return
        st.dataframe(result)
        _render_download_buttons(
            result, spec, f"{resolved_property.lower().replace(' ', '_')}_{system_label}_predictions_from_custom_model")
        _render_report(result, spec)


def _run_data_mode(also_run_model: bool) -> None:
    """UI for fetching + cleaning ILThermo data for any property, optionally chaining into
    the matching trained prediction model (for "Both" mode) when one exists."""
    from qspr_il.data.cleaning import fetch_curated_dataset, list_available_properties

    st.subheader("Fetch & clean ILThermo data")
    st.caption(
        "Not limited to density/refractive index -- pick any ILThermo property, any of the "
        "supported solvent systems (or a pure ionic liquid), and a temperature/pressure range."
    )

    properties = list_available_properties()
    default_idx = properties.index("density") if "density" in properties else 0
    property_query = st.selectbox("Property", properties, index=default_idx)

    solvent_choice = st.selectbox("System", KNOWN_DATA_SOLVENTS)
    if solvent_choice == "Pure ionic liquid":
        solvent_name = None
    elif solvent_choice == "Other...":
        solvent_name = st.text_input(
            "Solvent name (as it appears in ILThermo)", value="") or None
        st.caption(
            "SMILES resolution is only reliable for water/ethanol/isopropanol -- other solvents are best-effort.")
    else:
        solvent_name = solvent_choice

    col1, col2 = st.columns(2)
    with col1:
        temp_min, temp_max = st.slider(
            "Temperature (K) range", 200.0, 700.0, DEFAULT_TEMP_RANGE)
    with col2:
        limit_pressure = st.checkbox("Limit pressure range", value=True)
        pressure_range = None
        if limit_pressure:
            pressure_min, pressure_max = st.slider(
                "Pressure (kPa) range", 0.0, 500.0, DEFAULT_PRESSURE_RANGE)
            pressure_range = (pressure_min, pressure_max)

    max_datasets = st.number_input(
        "Max ILThermo datasets to download", min_value=1, value=30, step=5)

    with st.expander("Advanced ILThermo search filters (pyionics)"):
        st.caption(
            "The same server-side `ilsearch` filters the `pyionics` client exposes -- each "
            "one narrows the ILThermo query before any client-side cleaning. Leave blank to "
            "skip."
        )
        fcol1, fcol2, fcol3 = st.columns(3)
        year_filter = fcol1.text_input("Publication year", value="").strip()
        author_filter = fcol2.text_input("Author surname", value="").strip()
        keyword_filter = fcol3.text_input("Keyword", value="").strip()
    search_filters = (year_filter, author_filter, keyword_filter)

    if st.button("Fetch data", type="primary"):
        status = st.status(
            f"Fetching '{property_query}' data from ILThermo...", expanded=True)

        def _on_progress(message: str) -> None:
            status.write(message)

        try:
            df = fetch_curated_dataset(
                property_query,
                solvent_name,
                max_datasets=int(max_datasets),
                year=year_filter,
                author=author_filter,
                keyword=keyword_filter,
                temp_range=(temp_min, temp_max),
                pressure_range=pressure_range,
                progress_callback=_on_progress,
            )
        except Exception as e:
            status.update(label="Fetch failed", state="error")
            st.error(f"Fetch failed: {e}")
            with st.expander("Full error details (for debugging)"):
                st.exception(e)
            return

        status.update(
            label=f"Done: fetched {len(df)} curated row(s)." if not df.empty else "Done: no usable rows found.",
            state="complete",
        )
        resolved_property = df.loc[0, "Property"] if not df.empty else None
        st.session_state["data_result"] = (
            df, resolved_property, solvent_name, property_query, search_filters)

    cached = st.session_state.get("data_result")
    if cached is None:
        return
    df, resolved_property, solvent_name, cached_query, cached_filters = cached
    if cached_query != property_query or cached_filters != search_filters:
        return  # stale result from a different property selection or filter set

    if df.empty:
        st.warning(
            "No usable rows were found for this property/system/condition combination. This can "
            "happen if the bundled SMILES lookup table doesn't cover the compounds ILThermo "
            "returned -- see the data pipeline docs for known limitations."
        )
        return

    st.success(f"Fetched {len(df)} curated rows for '{resolved_property}'.")
    st.dataframe(df)
    st.download_button(
        "Download curated CSV",
        df.to_csv(index=False),
        file_name=f"{resolved_property.lower().replace(' ', '_')}_{solvent_name or 'pure'}.csv",
        mime="text/csv",
    )

    if not also_run_model:
        return

    solvent_label = "pure ionic liquid" if solvent_name is None else solvent_name
    try:
        spec = find_spec(resolved_property, solvent_label)
    except KeyError:
        _render_auto_train(df, resolved_property, solvent_name)
        return

    st.markdown(f"**A trained model exists for this data: {spec.label}**")
    if st.button(f"Run '{spec.label}' model on this data", type="primary"):
        mole_fraction_col = None if spec.is_pure else "Mole_fraction_IL"
        ensemble = _load_ensemble(str(spec.model_dir))
        result = _run_prediction_with_status(
            df, spec, ensemble, "IL_SMILES", "Temperature (K)", mole_fraction_col, f"Running '{spec.label}'..."
        )
        if result is None:
            return
        st.dataframe(result)
        _render_download_buttons(
            result, spec, f"{spec.property_name.lower().replace(' ', '_')}_{spec.solvent.replace(' ', '_')}_predictions_from_fetched_data"
        )
        _render_report(result, spec)
        _render_umap_expander(spec)


def _run_csv_mode(spec: ModelSpec) -> None:
    uploaded = st.file_uploader("Input CSV", type="csv")
    smiles_col = st.text_input("SMILES column", value=spec.default_smiles_col)
    mole_fraction_col = None
    if not spec.is_pure:
        mole_fraction_col = st.text_input(
            "Mole fraction column", value=spec.default_mole_fraction_col)
    temp_col = st.text_input(
        "Temperature column (leave empty for 298.15 K default)", value="")

    if uploaded is None:
        st.info("Upload a CSV to run predictions.")
        return

    if st.button("Run prediction", type="primary"):
        data = pd.read_csv(uploaded)
        ensemble = _load_ensemble(str(spec.model_dir))
        result = _run_prediction_with_status(
            data, spec, ensemble, smiles_col, temp_col or None, mole_fraction_col, "Running prediction..."
        )
        if result is None:
            return
        st.session_state["csv_result"] = (result, spec.key)

    cached = st.session_state.get("csv_result")
    if cached is None or cached[1] != spec.key:
        return
    result, _ = cached

    st.dataframe(result)
    _render_download_buttons(
        result, spec, f"{spec.property_name.lower().replace(' ', '_')}_{spec.solvent.replace(' ', '_')}_predictions")
    _render_report(result, spec)
    _render_umap_expander(spec)


def _run_single_entry_mode(spec: ModelSpec) -> None:
    smiles = st.text_input("Ionic liquid SMILES", value="")
    temperature = st.number_input("Temperature (K)", value=298.15)
    mole_fraction = None
    if not spec.is_pure:
        mole_fraction = st.number_input(
            "IL mole fraction", min_value=0.0, max_value=1.0, value=0.5)

    if st.button("Predict", type="primary"):
        if not smiles.strip():
            st.error("Enter a SMILES string.")
            return
        row = {"SMILES": [smiles], "Temperature": [temperature]}
        if mole_fraction is not None:
            row["Mole_fraction"] = [mole_fraction]
        data = pd.DataFrame(row)

        ensemble = _load_ensemble(str(spec.model_dir))
        result = _run_prediction_with_status(
            data,
            spec,
            ensemble,
            "SMILES",
            "Temperature",
            "Mole_fraction" if mole_fraction is not None else None,
            "Predicting...",
        )
        if result is None:
            return

        mean = result.loc[0, "prediction_mean"]
        std = result.loc[0, "prediction_std"]
        st.metric(f"Predicted {spec.target_column}", f"{mean} ± {std}")
        if result.loc[0, "Changes"] != "No changes":
            st.caption(f"SMILES standardization: {result.loc[0, 'Changes']}")
        _render_download_buttons(
            result, spec, f"{spec.property_name.lower().replace(' ', '_')}_{spec.solvent.replace(' ', '_')}_prediction")
        _render_umap_expander(spec)


def main() -> None:
    st.set_page_config(
        page_title="QSPR: IL Density & Refractive Index", page_icon="\U0001F9EA")
    if _BANNER_IMAGE.exists():
        st.image(str(_BANNER_IMAGE), width="stretch")
    st.title("QSPR: Ionic-Liquid Density & Refractive Index")
    st.write(
        "Predict density (kg/m3) and refractive index (Na D-line) of ionic-liquid "
        "mixtures and pure ionic liquids using trained XGBoost ensemble models."
    )

    action = st.sidebar.radio(
        "What do you want to do?",
        ["Run prediction model", "Fetch & clean data", "Both"],
        help="'Both' fetches ILThermo data for any property, then offers to run the "
        "matching trained model on it if one exists (currently: density, refractive index). "
        "Running a model shows its matching UMAP visualization in the results, if one exists.",
    )

    if action == "Fetch & clean data":
        _run_data_mode(also_run_model=False)
        return
    if action == "Both":
        _run_data_mode(also_run_model=True)
        return

    spec = _select_spec()
    st.subheader(spec.label)
    st.caption(spec.description)

    mode = st.radio("Input mode", ["Upload CSV",
                    "Single IL entry"], horizontal=True)
    if mode == "Upload CSV":
        _run_csv_mode(spec)
    else:
        _run_single_entry_mode(spec)


if __name__ == "__main__":
    main()
