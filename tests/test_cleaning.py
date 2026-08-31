import numpy as np
import pandas as pd
import pytest

from qspr_il.data import cleaning

# A real, valid SMILES for an imidazolium-based IL and its cation/anion parts,
# used across tests so RDKit standardization succeeds.
IL_SMILES = "CCCCn1cc[n+](C)c1.F[B-](F)(F)F"
ETHANOL_SMILES = "CCO"


def _pure_raw_row(**overrides):
    row = {
        "idset": "SET1",
        "author": "Doe et al.",
        "Temperature, K": 298.15,
        "Pressure, kPa": 101.0,
        "Specific density, kg/m3 - Liquid": 1050.0,
        "component 1 idout": "C001",
        "component 1 name": "1-butyl-3-methylimidazolium tetrafluoroborate",
        "component 1 SMILES": IL_SMILES,
    }
    row.update(overrides)
    return row


def _mixture_raw_row(**overrides):
    row = {
        "idset": "SET2",
        "author": "Doe et al.",
        "Temperature, K": 298.15,
        "Pressure, kPa": 101.0,
        "Mole fraction of ethanol": 0.4,
        "Specific density, kg/m3 - Liquid": 900.0,
        "component 1 idout": "C002",
        "component 1 name": "ethanol",
        "component 1 SMILES": ETHANOL_SMILES,
        "component 2 idout": "C001",
        "component 2 name": "1-butyl-3-methylimidazolium tetrafluoroborate",
        "component 2 SMILES": IL_SMILES,
    }
    row.update(overrides)
    return row


def test_compute_molecular_fields_valid_smiles():
    fields = cleaning.compute_molecular_fields(IL_SMILES)
    assert fields["IL_component_number"] == 2
    assert fields["IL_cation_SMILES"]
    assert fields["IL_anion_SMILES"]
    assert fields["IL_MW"] > 0
    assert fields["Contains_Metal"] is False


def test_compute_molecular_fields_detects_metal():
    # Sodium acetate-like fragment: contains a metal cation.
    fields = cleaning.compute_molecular_fields("[Na+].CC(=O)[O-]")
    assert fields["Contains_Metal"] is True


def test_flag_duplicates_exact_duplicate_dropped():
    df = pd.DataFrame(
        {
            "IL_SMILES": ["A", "A", "B"],
            "Temperature (K)": [298.15, 298.15, 300.0],
            "Property_value": [1000.0, 1000.0, 900.0],
        }
    )
    flags = cleaning.flag_duplicates(
        df, ["IL_SMILES", "Temperature (K)"], "Property_value")
    assert flags.tolist() == ["unique", "duplicate_dropped", "unique"]


def test_flag_duplicates_conflicting_values_flagged():
    df = pd.DataFrame(
        {
            "IL_SMILES": ["A", "A"],
            "Temperature (K)": [298.15, 298.15],
            "Property_value": [1000.0, 1200.0],  # >2% apart
        }
    )
    flags = cleaning.flag_duplicates(
        df, ["IL_SMILES", "Temperature (K)"], "Property_value")
    assert (flags == "conflicting_duplicate").all()


def test_filter_conditions_drops_out_of_range_pressure():
    df = pd.DataFrame(
        {
            "Temperature (K)": [298.15, 298.15, 298.15],
            "Pressure (kPa)": [101.0, 500.0, np.nan],
        }
    )
    filtered = cleaning.filter_conditions(df, allow_missing_pressure=True)
    assert len(filtered) == 2  # 101.0 kept, 500.0 dropped, NaN kept


def test_filter_conditions_pressure_range_none_skips_pressure_filter():
    df = pd.DataFrame(
        {
            "Temperature (K)": [298.15, 298.15],
            "Pressure (kPa)": [101.0, 5000.0],
        }
    )
    filtered = cleaning.filter_conditions(df, pressure_range=None)
    assert len(filtered) == 2


def test_filter_property_range():
    df = pd.DataFrame({"Property_value": [500.0, 5000.0, 1200.0]})
    filtered = cleaning.filter_property_range(
        df, "Property_value", (300.0, 3000.0))
    assert filtered["Property_value"].tolist() == [500.0, 1200.0]


def test_parse_numeric_plain_value():
    assert cleaning._parse_numeric("298.15") == 298.15


def test_parse_numeric_strips_uncertainty_suffix():
    # flatten_idset joins a [value, uncertainty] cell into "value;uncertainty" -- confirmed
    # against a real ILThermo response; only the primary value should be kept.
    assert cleaning._parse_numeric("914.7;1.8") == 914.7


def test_parse_numeric_unparseable_returns_nan():
    assert np.isnan(cleaning._parse_numeric("not a number"))
    assert np.isnan(cleaning._parse_numeric(None))


def test_build_curated_dataset_handles_value_with_uncertainty_suffix():
    # Regression test: a real ILThermo response's property-value cell of "914.7;1.8" used to
    # crash flag_duplicates's float conversion further down the pipeline.
    raw_df = pd.DataFrame(
        [_pure_raw_row(**{"Specific density, kg/m3 - Liquid": "914.7;1.8"})])
    curated = cleaning.build_curated_dataset(
        raw_df, "Density", solvent_name=None)
    assert len(curated) == 1
    assert curated.loc[0, "Property_value"] == 914.7


def test_build_curated_dataset_reports_progress():
    raw_df = pd.DataFrame([_pure_raw_row(), _pure_raw_row(
        **{"idset": "SET2", "Temperature, K": 300.0})])
    messages = []
    cleaning.build_curated_dataset(
        raw_df, "Density", solvent_name=None, progress_callback=messages.append)
    assert any("Standardizing" in m for m in messages)
    assert any("temperature/pressure" in m for m in messages)
    assert any("Deduplication" in m for m in messages)


def test_build_curated_dataset_pure_columns_match_target_schema():
    raw_df = pd.DataFrame([_pure_raw_row()])
    curated = cleaning.build_curated_dataset(
        raw_df, "Density", solvent_name=None)
    assert list(curated.columns) == cleaning.GENERIC_PURE_COLUMNS
    assert len(curated) == 1
    assert curated.loc[0, "Property"] == "Density"
    assert curated.loc[0, "Property_value"] == 1050.0
    assert curated.loc[0, "phases"] == "Liquid"
    assert curated.loc[0, "Data_quality_flag"] == "unique"


def test_build_curated_dataset_mixture_columns_match_target_schema():
    raw_df = pd.DataFrame([_mixture_raw_row()])
    curated = cleaning.build_curated_dataset(
        raw_df, "Density", solvent_name="ethanol")
    assert list(curated.columns) == cleaning.GENERIC_MIXTURE_COLUMNS
    assert len(curated) == 1
    assert curated.loc[0, "Solvent_name"] == "ethanol"
    assert curated.loc[0,
                       "IL_name"] == "1-butyl-3-methylimidazolium tetrafluoroborate"
    assert curated.loc[0, "Property_value"] == 900.0


def test_build_curated_dataset_mole_fraction_il_when_raw_value_is_solvents():
    # ILThermo reported this row's "Mole fraction" against the solvent (ethanol) -- confirmed
    # via the "Mole fraction of ethanol" raw column name in the fixture.
    raw_df = pd.DataFrame(
        [_mixture_raw_row(**{"Mole fraction of ethanol": 0.4})])
    curated = cleaning.build_curated_dataset(
        raw_df, "Density", solvent_name="ethanol")
    assert curated.loc[0, "Mole fraction of solvent"] == 0.4
    assert curated.loc[0, "Mole_fraction_IL"] == pytest.approx(0.6)


def test_build_curated_dataset_mole_fraction_il_when_raw_value_is_ils():
    # Same underlying composition, but this time ILThermo reported the raw "Mole fraction"
    # against the IL directly -- Mole_fraction_IL must come out the same either way.
    raw_df = pd.DataFrame(
        [
            _mixture_raw_row(
                **{
                    "Mole fraction of ethanol": None,
                    "Mole fraction of 1-butyl-3-methylimidazolium tetrafluoroborate": 0.6,
                }
            )
        ]
    )
    raw_df = raw_df.drop(columns=["Mole fraction of ethanol"])
    curated = cleaning.build_curated_dataset(
        raw_df, "Density", solvent_name="ethanol")
    assert curated.loc[0, "Mole_fraction_IL"] == pytest.approx(0.6)
    assert curated.loc[0, "Mole fraction of solvent"] == pytest.approx(0.4)


def test_build_curated_dataset_handles_concatenated_idsets_with_different_column_names():
    # Regression test: raw_df is normally the concatenation of several idsets, and ILThermo
    # embeds the specific compound name directly in condition column headers -- different
    # idsets therefore use *differently-named* "Mole fraction of <compound>" columns. After
    # concatenation, each row must pull its own data from its own idset's columns, not get
    # matched against some other row's compound/column by a single global column pick.
    # a second, different, valid IL
    other_il_smiles = "CCN1C=C[N+](=C1)C.F[B-](F)(F)F"
    row_a = _mixture_raw_row(
        **{
            "idset": "SET_A",
            "Mole fraction of ethanol": 0.3,
            "Specific density, kg/m3 - Liquid": 1000.0,
        }
    )
    row_b = _mixture_raw_row(
        **{
            "idset": "SET_B",
            "component 2 name": "1-ethyl-3-methylimidazolium tetrafluoroborate",
            "component 2 SMILES": other_il_smiles,
        }
    )
    del row_b["Mole fraction of ethanol"]
    row_b["Mole fraction of 1-ethyl-3-methylimidazolium tetrafluoroborate"] = 0.7
    row_b["Specific density, kg/m3 - Liquid"] = 1100.0

    raw_df = pd.concat(
        [pd.DataFrame([row_a]), pd.DataFrame([row_b])], ignore_index=True)
    curated = cleaning.build_curated_dataset(
        raw_df, "Density", solvent_name="ethanol")

    assert len(curated) == 2
    by_setid = curated.set_index("setid")
    assert by_setid.loc["SET_A",
                        "IL_name"] == "1-butyl-3-methylimidazolium tetrafluoroborate"
    assert by_setid.loc["SET_A", "Property_value"] == 1000.0
    assert by_setid.loc["SET_A", "Mole_fraction_IL"] == pytest.approx(
        0.7)  # 0.3 was the solvent's
    assert by_setid.loc["SET_B",
                        "IL_name"] == "1-ethyl-3-methylimidazolium tetrafluoroborate"
    assert by_setid.loc["SET_B", "Property_value"] == 1100.0
    assert by_setid.loc["SET_B", "Mole_fraction_IL"] == pytest.approx(
        0.7)  # reported directly against the IL


def test_build_curated_dataset_works_for_a_non_density_property():
    # Confirms the pipeline is property-agnostic: viscosity works the same way as density.
    raw_df = pd.DataFrame([_pure_raw_row(
        **{"Specific density, kg/m3 - Liquid": None, "Viscosity, mPa*s - Liquid": 12.5})])
    raw_df = raw_df.drop(columns=["Specific density, kg/m3 - Liquid"])
    curated = cleaning.build_curated_dataset(
        raw_df, "Viscosity", solvent_name=None)
    assert len(curated) == 1
    assert curated.loc[0, "Property"] == "Viscosity"
    assert curated.loc[0, "Property_value"] == 12.5


def test_build_curated_dataset_drops_unparseable_smiles_when_pubchem_also_fails(monkeypatch):
    # a bare formula, not SMILES
    raw_df = pd.DataFrame(
        [_pure_raw_row(**{"component 1 SMILES": "C12H19F6N3O4S2"})])
    monkeypatch.setattr(cleaning, "resolve_smiles_by_name",
                        lambda name, timeout=10.0: None)  # PubChem: no match either

    curated = cleaning.build_curated_dataset(
        raw_df, "Density", solvent_name=None)
    assert curated.empty


def test_build_curated_dataset_filters_by_temperature_range():
    raw_df = pd.DataFrame([_pure_raw_row(**{"Temperature, K": 1000.0})])
    curated = cleaning.build_curated_dataset(
        raw_df, "Density", solvent_name=None, temp_range=(253.0, 573.0))
    assert curated.empty


def test_list_available_properties_includes_more_than_density_and_ri():
    properties = cleaning.list_available_properties()
    assert "density" in properties
    assert "refractive-index" in properties
    assert "viscosity" in properties
    # not just the 2 properties this project trains models for
    assert len(properties) > 10


def test_resolve_property_display_name_matches_hyphenated_short_name():
    idsets_json = {"res": [["id1", "auth", "Density", "Liquid"], [
        "id2", "auth", "Viscosity", "Liquid"]]}
    assert cleaning.resolve_property_display_name(
        idsets_json, "density") == "Density"
    assert cleaning.resolve_property_display_name(
        idsets_json, "refractive-index") is None


def test_resolve_property_display_name_fuzzy_substring_match():
    idsets_json = {"res": [["id1", "auth", "Refractive index", "Liquid"]]}
    assert cleaning.resolve_property_display_name(
        idsets_json, "refractive") == "Refractive index"


def test_resolve_smiles_by_name_success(monkeypatch):
    class _FakeResponse:
        status_code = 200
        text = "CCO\n"

    monkeypatch.setattr(cleaning.requests, "get", lambda url,
                        timeout=None: _FakeResponse())
    assert cleaning.resolve_smiles_by_name("ethanol") == "CCO"


def test_resolve_smiles_by_name_not_found_returns_none(monkeypatch):
    class _FakeResponse:
        status_code = 404
        text = ""

    monkeypatch.setattr(cleaning.requests, "get", lambda url,
                        timeout=None: _FakeResponse())
    assert cleaning.resolve_smiles_by_name("not-a-real-compound") is None


def test_resolve_smiles_by_name_tries_bracket_variant(monkeypatch):
    calls = []

    class _FakeResponse:
        def __init__(self, ok):
            self.status_code = 200 if ok else 404
            self.text = "CCO\n" if ok else ""

    def fake_get(url, timeout=None):
        calls.append(url)
        # Fail on the first (bracketed) variant, succeed on the plain-parens variant.
        return _FakeResponse(ok=len(calls) == 2)

    monkeypatch.setattr(cleaning.requests, "get", fake_get)
    result = cleaning.resolve_smiles_by_name(
        "bis[(trifluoromethyl)sulfonyl]imide")
    assert result == "CCO"
    assert len(calls) == 2


def test_resolve_smiles_by_name_rejects_ambiguous_multi_match(monkeypatch):
    # Regression test: PubChem returning several distinct compound matches for one name
    # search (a fuzzy/synonym match, not several representations of the same compound) used
    # to silently take the first one -- confirmed against a real full IL-salt name that
    # matched several different real salts with different alkyl chain lengths.
    class _FakeResponse:
        status_code = 200
        text = (
            "CCCCN1C=C[N+](=C1C)C.C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F\n"
            "CCCCCCCCCCN1C=C[N+](=C1)C.C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F\n"
        )

    monkeypatch.setattr(cleaning.requests, "get", lambda url,
                        timeout=None: _FakeResponse())
    assert cleaning.resolve_smiles_by_name(
        "bis[(trifluoromethyl)sulfonyl]imide") is None


def test_resolve_smiles_by_name_empty_input_returns_none():
    assert cleaning.resolve_smiles_by_name("") is None
    assert cleaning.resolve_smiles_by_name("   ") is None


def test_formula_matches_same_compound():
    assert cleaning.formula_matches(IL_SMILES, "C8H15BF4N2")


def test_formula_matches_rejects_different_compound():
    # ethanol vs the IL's real formula
    assert not cleaning.formula_matches("CCO", "C8H15BF4N2")


def test_formula_matches_no_expected_formula_passes():
    # Nothing to contradict -- treated as a pass (caller has no ground truth either way).
    assert cleaning.formula_matches(IL_SMILES, "")


def test_formula_matches_invalid_smiles_fails():
    assert not cleaning.formula_matches("not a smiles", "C8H15BF4N2")


# The real molecular formula of IL_SMILES (C8H15BF4N2) -- used as the "expected_formula"
# ILThermo would have reported for this compound, so a mocked PubChem candidate that matches
# IL_SMILES also passes the formula cross-check in _resolve_component_smiles.
IL_FORMULA = "C8H15BF4N2"


def test_build_curated_dataset_falls_back_to_pubchem_for_unresolved_il(monkeypatch):
    # The id-based addSmiles lookup produced a bare formula (not real SMILES); PubChem
    # resolves it by name instead of the row being dropped.
    raw_df = pd.DataFrame(
        [_pure_raw_row(**{"component 1 SMILES": IL_FORMULA})])
    monkeypatch.setattr(cleaning, "resolve_smiles_by_name",
                        lambda name, timeout=10.0: IL_SMILES)

    curated = cleaning.build_curated_dataset(
        raw_df, "Density", solvent_name=None)
    assert len(curated) == 1
    assert curated.loc[0, "IL_SMILES"] == IL_SMILES


def test_build_curated_dataset_rejects_pubchem_match_with_wrong_formula(monkeypatch):
    # Regression test: PubChem can return a single, unambiguous-looking match that is
    # nonetheless the wrong compound -- confirmed against a real IL salt name. The formula
    # cross-check must catch this even though there's no ambiguity signal to rely on.
    raw_df = pd.DataFrame(
        [_pure_raw_row(**{"component 1 SMILES": IL_FORMULA})])
    wrong_but_valid_smiles = "CCO"  # ethanol: valid SMILES, but not C8H15BF4N2
    monkeypatch.setattr(cleaning, "resolve_smiles_by_name",
                        lambda name, timeout=10.0: wrong_but_valid_smiles)

    curated = cleaning.build_curated_dataset(
        raw_df, "Density", solvent_name=None)
    # wrong-formula candidate rejected, falls back to the (unparseable) formula, row dropped
    assert curated.empty


def test_build_curated_dataset_drops_when_pubchem_fallback_disabled(monkeypatch):
    raw_df = pd.DataFrame(
        [_pure_raw_row(**{"component 1 SMILES": IL_FORMULA})])
    calls = []
    monkeypatch.setattr(cleaning, "resolve_smiles_by_name",
                        lambda name, timeout=10.0: calls.append(name) or IL_SMILES)

    curated = cleaning.build_curated_dataset(
        raw_df, "Density", solvent_name=None, use_pubchem_fallback=False)
    assert curated.empty
    assert calls == []  # fallback never even attempted


def test_build_curated_dataset_caches_repeated_name_lookups(monkeypatch):
    raw_df = pd.DataFrame(
        [
            _pure_raw_row(
                **{"component 1 SMILES": IL_FORMULA, "idset": "SET1"}),
            _pure_raw_row(**{"component 1 SMILES": IL_FORMULA,
                          "idset": "SET2", "Temperature, K": 300.0}),
        ]
    )
    calls = []
    monkeypatch.setattr(cleaning, "resolve_smiles_by_name",
                        lambda name, timeout=10.0: calls.append(name) or IL_SMILES)

    curated = cleaning.build_curated_dataset(
        raw_df, "Density", solvent_name=None)
    assert len(curated) == 2
    assert len(calls) == 1  # same compound name looked up only once


def test_fetch_curated_dataset_raises_clear_error_for_unmatched_property(tmp_path, monkeypatch):
    def fake_getIdsets(**kwargs):
        path = tmp_path / "idsets.json"
        path.write_text('{"res": [["id1", "auth", "Density", "Liquid"]]}')
        return path

    monkeypatch.setattr(cleaning.ionics_client, "getIdsets", fake_getIdsets)
    with pytest.raises(ValueError, match="not-a-real-property"):
        cleaning.fetch_curated_dataset(
            "not-a-real-property", None, data_root=tmp_path)
