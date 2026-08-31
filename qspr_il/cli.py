"""Command-line entry point for running QSPR predictions and fetching ILThermo data.

Replaces the old ``qspr.py`` menu + ``subprocess`` dispatch and the 8 duplicated
``apply_*.py`` argparse blocks. Everything runs in-process against
:mod:`qspr_il.models.engine` and :mod:`qspr_il.data.cleaning`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qspr_il.models.engine import load_models_and_metadata, run_prediction
from qspr_il.registry import ModelSpec, iter_specs
from qspr_il.registry import find as find_spec
from qspr_il.registry import get as get_spec

DEFAULT_TEMP_RANGE = (253.0, 573.0)
DEFAULT_PRESSURE_RANGE = (90.0, 110.0)


def _prompt(label: str, default: str = "") -> str:
    if default:
        value = input(f"{label} [{default}]: ").strip()
        return value or default
    return input(f"{label}: ").strip()


def _interactive_choose_model() -> ModelSpec:
    print("\nChoose a prediction model:")
    for spec in iter_specs():
        print(f"  {spec.key}. {spec.label}")
    while True:
        choice = _prompt("Model", "1")
        try:
            return get_spec(choice)
        except KeyError:
            print("Please select a number from 1 to 8.")


def resolve_args(args: argparse.Namespace, spec: ModelSpec) -> argparse.Namespace:
    """Fill in any unset CLI arguments interactively, using ``spec`` for defaults.

    Only asks for a mole-fraction column when the model is not a pure-IL model.
    """
    if args.input_csv is None:
        print("\nPrediction settings (press Enter to use the default in brackets).")
        args.input_csv = _prompt(
            "Input CSV path", "datasets/external_test_set.csv")
        while not args.input_csv:
            print("Input CSV path is required.")
            args.input_csv = _prompt("Input CSV path", "")
        args.smiles_col = _prompt(
            "SMILES column", args.smiles_col or spec.default_smiles_col)
        if not spec.is_pure:
            args.mole_fraction_col = _prompt(
                "Mole fraction column", args.mole_fraction_col or spec.default_mole_fraction_col
            )
        args.temp_col = _prompt(
            "Temperature column (leave empty for 298.15 K)", args.temp_col or "")
        args.model_dir = _prompt(
            "Model directory", args.model_dir or str(spec.model_dir))
        default_output = f"results/{Path(spec.model_dir).parent.name}_prediction.csv"
        args.output_csv = _prompt(
            "Output CSV path", args.output_csv or default_output)
        print()
    else:
        if not args.smiles_col:
            args.smiles_col = spec.default_smiles_col
        if not spec.is_pure and not args.mole_fraction_col:
            args.mole_fraction_col = spec.default_mole_fraction_col
        if not args.model_dir:
            args.model_dir = str(spec.model_dir)
        if not args.output_csv:
            args.output_csv = f"results/{Path(spec.model_dir).parent.name}_prediction.csv"

    if spec.is_pure:
        args.mole_fraction_col = None

    return args


def _run_prediction_action(args: argparse.Namespace, spec: ModelSpec | None = None) -> int:
    """Run the existing prediction workflow: pick a model, load an input CSV, predict."""
    spec = spec or (get_spec(args.model)
                    if args.model else _interactive_choose_model())
    args = resolve_args(args, spec)

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        return 1

    data = pd.read_csv(input_path)
    return _predict_and_save(data, spec, args.model_dir, args.smiles_col, args.temp_col, args.mole_fraction_col, args.output_csv)


def _predict_and_save(
    data: pd.DataFrame,
    spec: ModelSpec,
    model_dir: str,
    smiles_col: str | None,
    temp_col: str | None,
    mole_fraction_col: str | None,
    output_csv: str,
) -> int:
    try:
        ensemble = load_models_and_metadata(model_dir)
    except FileNotFoundError as e:
        print(f"Error loading models or metadata: {e}")
        return 1

    try:
        result = run_prediction(
            data,
            spec,
            ensemble=ensemble,
            smiles_col=smiles_col,
            temp_col=temp_col,
            mole_fraction_col=mole_fraction_col,
            progress_callback=lambda message: print(f"  {message}"),
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return 0


def _prompt_action() -> str:
    print("\nWhat would you like to do?")
    print("  1. Run a prediction model (density / refractive index)")
    print("  2. Fetch & clean ILThermo data for any property")
    print("  3. Both -- fetch data, then run the matching prediction model if one exists")
    while True:
        choice = _prompt("Choice", "1")
        if choice in ("1", "model"):
            return "model"
        if choice in ("2", "data"):
            return "data"
        if choice in ("3", "both"):
            return "both"
        print("Please enter 1, 2, or 3.")


def _prompt_range(label: str, default: tuple[float, float] | None) -> tuple[float, float] | None:
    default_str = f"{default[0]:g}-{default[1]:g}" if default else "none"
    raw = _prompt(
        f"{label} range (min-max, or 'none' for no limit)", default_str)
    if raw.strip().lower() in ("none", "any", ""):
        return None if raw.strip().lower() != "" else default
    try:
        lo, hi = raw.split("-", 1)
        return (float(lo), float(hi))
    except ValueError:
        print(
            f"Could not parse {raw!r} as 'min-max'; using default {default}.")
        return default


def _print_numbered_columns(items: list[str], columns: int = 3) -> None:
    rows = -(-len(items) // columns)  # ceil division
    labels = [f"{i + 1}. {item}" for i, item in enumerate(items)]
    col_widths = [
        max((len(labels[r + c * rows]) for r in range(rows)
            if r + c * rows < len(labels)), default=0) + 2
        for c in range(columns)
    ]
    for r in range(rows):
        line = ""
        for c in range(columns):
            idx = r + c * rows
            if idx < len(items):
                line += labels[idx].ljust(col_widths[c])
        print("  " + line.rstrip())


def _prompt_property() -> str:
    from qspr_il.data.cleaning import list_available_properties

    properties = list_available_properties()
    print(f"\n{len(properties)} ILThermo properties are available:")
    _print_numbered_columns(properties)
    default_number = properties.index(
        "density") + 1 if "density" in properties else 1
    while True:
        choice = _prompt(
            "Property number (or type a property name)", str(default_number))
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(properties):
                return properties[index - 1]
            print(f"Please enter a number from 1 to {len(properties)}.")
            continue
        if choice:
            return choice
        print("Please enter a number or a property name.")


def _prompt_data_settings() -> dict:
    property_query = _prompt_property()

    print("\nSystem: name a solvent for a binary IL-solvent mixture, or leave blank for a pure ionic liquid.")
    print("  Solvents with reliable SMILES resolution: water, ethanol, isopropanol")
    solvent_name = _prompt("Solvent (blank = pure IL)", "") or None

    temp_range = _prompt_range("Temperature (K)", DEFAULT_TEMP_RANGE)
    pressure_range = _prompt_range("Pressure (kPa)", DEFAULT_PRESSURE_RANGE)

    print(
        "\nOptional ILThermo search filters (same as pyionics; blank = skip):")
    year = _prompt("  Publication year", "")
    author = _prompt("  Author surname", "")
    keyword = _prompt("  Keyword", "")

    max_datasets_raw = _prompt(
        "Max ILThermo datasets to download (blank = no limit)", "30")
    max_datasets = int(max_datasets_raw) if max_datasets_raw.strip() else None

    default_output = f"datasets/custom/{property_query.replace(' ', '_')}_{solvent_name or 'pure'}.csv"
    output_csv = _prompt("Save curated CSV to", default_output)

    return {
        "property_query": property_query,
        "solvent_name": solvent_name,
        "temp_range": temp_range,
        "pressure_range": pressure_range,
        "year": year,
        "author": author,
        "keyword": keyword,
        "max_datasets": max_datasets,
        "output_csv": output_csv,
    }


def _run_data_action() -> tuple[pd.DataFrame, str | None, str | None, str] | None:
    """Interactively fetch + clean ILThermo data for any property and save it to disk.

    Returns ``(curated_df, resolved_property_name, solvent_name, output_csv)`` on success
    (``resolved_property_name`` is ``None`` if the fetch returned no rows), or ``None`` if the
    fetch failed outright (e.g. the requested property matched nothing).
    """
    from qspr_il.data.cleaning import fetch_curated_dataset

    settings = _prompt_data_settings()
    print(
        f"\nFetching '{settings['property_query']}' data from ILThermo "
        f"({'pure ionic liquid' if settings['solvent_name'] is None else settings['solvent_name']})..."
    )
    try:
        df = fetch_curated_dataset(
            settings["property_query"],
            settings["solvent_name"],
            max_datasets=settings["max_datasets"],
            year=settings["year"],
            author=settings["author"],
            keyword=settings["keyword"],
            temp_range=settings["temp_range"],
            pressure_range=settings["pressure_range"],
            progress_callback=lambda message: print(f"  {message}"),
        )
    except Exception as e:
        print(f"Error: {e}")
        return None

    resolved_property = df.loc[0, "Property"] if not df.empty else None
    if df.empty:
        print(
            "No usable rows were found for this property/system/condition combination "
            "(this can happen if the bundled SMILES lookup table doesn't cover the "
            "compounds ILThermo returned -- see qspr_il.data.cleaning module docs)."
        )
    else:
        output_path = Path(settings["output_csv"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(
            f"Saved {len(df)} curated rows ({resolved_property}) to {output_path}")

    return df, resolved_property, settings["solvent_name"], settings["output_csv"]


def _offer_auto_train(df: pd.DataFrame, resolved_property: str, solvent_name: str | None) -> int:
    """No trained model exists for this property -- offer to train a new one on the data
    just fetched, then optionally run it immediately.
    """
    print(
        f"\nNo trained prediction model exists yet for '{resolved_property}'.")
    train_now = _prompt(
        "Train a new model on this freshly fetched data now? [y/N]", "N")
    if not train_now.strip().lower().startswith("y"):
        print("Data-only -- no model trained.")
        return 0

    from qspr_il.models.training import train_ensemble

    mole_fraction_col = None if solvent_name is None else "Mole_fraction_IL"
    system_label = solvent_name or "pure"
    default_dir = f"results/custom_models/{resolved_property.lower().replace(' ', '_')}_{system_label}_ensemble_model"
    output_dir = _prompt("Save trained model to", default_dir)
    try:
        n_models = int(_prompt("Number of ensemble members to train", "5"))
    except ValueError:
        n_models = 5

    print(
        f"\nTraining a new model for '{resolved_property}' "
        f"({'pure ionic liquid' if solvent_name is None else solvent_name})..."
    )
    try:
        _ensemble, metrics = train_ensemble(
            df,
            property_value_col="Property_value",
            smiles_col="IL_SMILES",
            temp_col="Temperature (K)",
            mole_fraction_col=mole_fraction_col,
            output_dir=output_dir,
            n_models=n_models,
            progress_callback=lambda message: print(f"  {message}"),
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print("\nValidation metrics per ensemble member (group k-fold by IL SMILES):")
    for m in metrics:
        print(
            f"  Model {m['model']}: RMSE={m['rmse']:.4g}, R2={m['r2']:.3f} (train={m['n_train']}, val={m['n_val']})")
    print(
        f"\nModel saved to {output_dir}. Reuse it later with:\n"
        f"  qspr-il-predict --model_dir {output_dir} --input_csv <your.csv> --smiles_col <col>"
        f"{' --mole_fraction_col <col>' if mole_fraction_col else ''} --output_csv <out.csv>"
    )

    run_now = _prompt(
        "\nRun this newly trained model on the fetched data now? [Y/n]", "Y")
    if run_now.strip().lower().startswith("n"):
        return 0

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
    default_output = f"results/{resolved_property.lower().replace(' ', '_')}_{system_label}_prediction.csv"
    output_csv = _prompt("Save predictions to", default_output)
    return _predict_and_save(df, spec, output_dir, "IL_SMILES", "Temperature (K)", mole_fraction_col, output_csv)


def _run_both_action() -> int:
    outcome = _run_data_action()
    if outcome is None:
        return 1
    df, resolved_property, solvent_name, _output_csv = outcome
    if df.empty or resolved_property is None:
        return 0

    solvent_label = "pure ionic liquid" if solvent_name is None else solvent_name
    try:
        spec = find_spec(resolved_property, solvent_label)
    except KeyError:
        return _offer_auto_train(df, resolved_property, solvent_name)

    run_now = _prompt(
        f"\nRun the matching '{spec.label}' prediction model on this data now? [Y/n]", "Y")
    if run_now.strip().lower().startswith("n"):
        return 0

    mole_fraction_col = None if spec.is_pure else "Mole_fraction_IL"
    default_output = f"results/{Path(spec.model_dir).parent.name}_prediction.csv"
    output_csv = _prompt("Save predictions to", default_output)
    return _predict_and_save(df, spec, str(spec.model_dir), "IL_SMILES", "Temperature (K)", mole_fraction_col, output_csv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict density or refractive index of ionic liquids.")
    parser.add_argument(
        "--action",
        choices=["model", "data", "both"],
        help="What to do: run a prediction model, fetch & clean ILThermo data, or both. "
        "Defaults to 'model' if --model is given, otherwise you are prompted interactively.",
    )
    parser.add_argument(
        "--model", choices=sorted(r.key for r in iter_specs()), help="Model to run (1-8).")
    parser.add_argument(
        "--input_csv", help="Path to the input CSV file (if omitted, you will be prompted).")
    parser.add_argument(
        "--smiles_col", help="Name of the SMILES column in the input CSV.")
    parser.add_argument("--mole_fraction_col",
                        help="Name of the mole fraction column (mixture models only).")
    parser.add_argument("--temp_col", help="Name of the temperature column.")
    parser.add_argument(
        "--model_dir", help="Directory containing the ensemble of models and metadata.")
    parser.add_argument(
        "--output_csv", help="Path to save the output CSV with predictions.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    action = args.action or (
        "model" if args.model else None) or _prompt_action()

    if action == "data":
        return 0 if _run_data_action() is not None else 1
    if action == "both":
        return _run_both_action()
    return _run_prediction_action(args)


if __name__ == "__main__":
    raise SystemExit(main())
