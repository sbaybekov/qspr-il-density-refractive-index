"""Fetch + clean pipeline: turns raw ILThermo data into curated, training-ready CSVs.

Nothing in this repo previously implemented the "duplicate removal, consistency
checks, and standardization" the README describes — :mod:`qspr_il.data.ionics`
only fetches and flattens raw data. This module is that missing piece. It does
**not** do any model training/fitting — its output is a curated CSV for a human
to use however they choose (including, but not limited to, retraining models
outside this pipeline's scope).

This module is **not** limited to density and refractive index (the two
properties this project has trained models for). Any of the ~56 properties
ILThermo tracks (see ``qspr_il/data/ionics/keydata/property_idsets.csv``, or
call :func:`list_available_properties`) can be fetched and cleaned, over any
temperature/pressure range -- the curated output uses a generic
``Property``/``Property_value`` column pair rather than a property-specific
column name, so the same code path handles density, viscosity, refractive
index, surface tension, or anything else in that list.

Column semantics here were reverse-engineered from real, live ILThermo API
responses (not just the existing curated CSVs) since no raw sample was
committed to the repo. In particular:

* The bundled ``keydata/property_idsets.csv`` "prp" ids used by
  :func:`qspr_il.data.ionics.client.getIdsets` were found to return zero
  results against the *current* live API (verified empirically) -- ILThermo's
  internal ids appear to have drifted since that lookup table was built. This
  module works around that by searching broadly (unfiltered by ``prp``) and
  then filtering the returned idsets client-side by their exact property
  display name (e.g. ``"Density"``, ``"Refractive index"``, ``"Viscosity"``),
  resolved fuzzily (see :func:`resolve_property_display_name`) so callers can
  pass either that display name or the hyphenated short name from
  ``property_idsets.csv`` (e.g. ``"refractive-index"``).
* The bundled ``keydata/smiles.csv`` compound-id lookup table is similarly a
  point-in-time snapshot and may not cover every compound id ILThermo returns
  today; :func:`qspr_il.data.ionics.client.addSmiles` silently falls back to
  the raw molecular formula string when a compound id isn't found. Live
  spot-checks found this table's IL-compound coverage is sparse to
  nonexistent -- it appears to mostly cover common small molecules/solvents
  rather than the complex IL cations/anions this project models, and even
  solvent component *ids* drift from what's in the table (though the table's
  SMILES for that solvent *name* is still correct).
* **Fix for the above:** whenever the id-based lookup produces something that
  doesn't parse as a valid SMILES, :func:`build_curated_dataset` falls back to
  :func:`resolve_smiles_by_name`, which looks the compound's *name* up on
  PubChem instead of its (possibly stale) ILThermo id. ILThermo's compound
  names are usually well-formed enough for PubChem's exact-name search to
  resolve directly (confirmed against real IL cation/anion names), which
  recovers most rows that used to be silently dropped. Solvent SMILES still
  prefer the small, reliable :data:`SOLVENT_SMILES_BY_NAME` table first. Only
  a compound that fails *both* the id-based lookup and the PubChem name
  lookup is dropped -- pass ``use_pubchem_fallback=False`` to
  :func:`build_curated_dataset`/:func:`fetch_curated_dataset` to disable this
  and stay fully offline (at the cost of recovering fewer rows).
"""

from __future__ import annotations

import csv
import json
import re
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from rdkit.Chem import Descriptors, MolFromSmiles, rdMolDescriptors

from qspr_il.data.ionics import client as ionics_client
from qspr_il.models.engine import reorder_charged_species, standardize_molecule

PUBCHEM_SMILES_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/TXT"

# Elements whose presence marks an ionic liquid as metal-containing (alkali, alkaline-earth,
# transition, and post-transition metals commonly seen in IL literature).
METALS = {
    "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}

# Solvents this project's trained models cover explicitly; any other component name in a
# binary system is treated as the IL. Fetching data for other solvent systems still works
# (the component roles just can't be told apart automatically -- see build_curated_dataset).
KNOWN_SOLVENTS = {"water", "ethanol", "isopropanol", "2-propanol"}

# The live ILThermo API was found (empirically) to assign a component id for a given named
# solvent that does not always match the id addSmiles's bundled keydata/smiles.csv table used
# for that same compound name -- even though the table does contain the right SMILES under a
# different id. Since the solvents this project cares about are a small, fixed set, resolving
# them by name here is far more reliable than depending on id drift in the fetched data.
SOLVENT_SMILES_BY_NAME = {
    "water": "O",
    "ethanol": "CCO",
    "isopropanol": "CC(C)O",
    "2-propanol": "CC(C)O",
}

# Curated output schema. Deliberately property-agnostic: "Property" names what was measured
# (e.g. "Density", "Viscosity") and "Property_value" holds the reading, instead of a
# different hardcoded column per property.
GENERIC_MIXTURE_COLUMNS = [
    "setid", "Temperature (K)", "Pressure (kPa)", "Property", "Property_value", "reference", "phases",
    "Mole fraction", "Mole fraction of compound name", "Weight fraction", "Weight fraction of compound name",
    "IL_SMILES", "IL_name", "IL_ID", "Solvent_SMILES", "Solvent_name", "Solvent_ID",
    "IL_cation_SMILES", "IL_anion_SMILES", "IL_component_number", "Mole fraction of solvent",
    "Mole_fraction_IL", "IL_MW", "Solvent_MW", "Record_ID", "Contains_Metal", "Standardized_IL_SMILES",
    "Changes", "Data_quality_flag",
]
GENERIC_PURE_COLUMNS = [
    "setid", "Temperature (K)", "Pressure (kPa)", "Property", "Property_value", "reference", "phases",
    "IL_ID", "IL_SMILES", "IL_name", "IL_cation_SMILES", "IL_anion_SMILES", "IL_component_number",
    "Record_ID", "Contains_Metal", "Standardized_IL_SMILES", "Changes", "Data_quality_flag",
]


def list_available_properties() -> list[str]:
    """All ILThermo property names this pipeline can fetch (from the bundled keydata table).

    Returns the hyphenated short names as ILThermo/pyionics use them (e.g.
    ``"refractive-index"``, ``"viscosity"``). Pass any of these -- or the human-readable
    display name (e.g. ``"Refractive index"``) -- as ``property_query`` to
    :func:`fetch_curated_dataset`.
    """
    with open(ionics_client.DEFAULT_PROPERTY_IDSETS, newline="") as f:
        return [row["property"] for row in csv.DictReader(f) if row.get("property")]


def resolve_property_display_name(idsets_json: dict, requested: str) -> str | None:
    """Match a user-supplied property name/short-name against a live search result's
    actual property display names (e.g. resolve ``"refractive-index"`` or ``"refractive"``
    to whatever exact string ILThermo returned, such as ``"Refractive index"``).

    Matching is case-insensitive and tolerant of hyphens vs spaces, since the bundled
    ``property_idsets.csv`` names (e.g. ``"refractive-index"``) don't exactly match ILThermo's
    live display strings (e.g. ``"Refractive index"``). Returns ``None`` if nothing matches.
    """
    candidates = sorted({row[2]
                        for row in idsets_json.get("res", []) if row[2]})
    requested_norm = requested.strip().lower().replace("-", " ")
    for candidate in candidates:
        if candidate.lower() == requested_norm:
            return candidate
    for candidate in candidates:
        candidate_norm = candidate.lower()
        if requested_norm in candidate_norm or candidate_norm in requested_norm:
            return candidate
    return None


def compute_molecular_fields(smiles: str) -> dict:
    """Standardize a (possibly multi-component) IL SMILES and derive descriptive fields.

    Returns a dict with ``Standardized_IL_SMILES``, ``Changes``, ``IL_cation_SMILES``,
    ``IL_anion_SMILES``, ``IL_component_number``, ``IL_MW``, and ``Contains_Metal``. If the
    SMILES fails to parse, ``Standardized_IL_SMILES`` is the original input and ``IL_MW`` is NaN.
    """
    standardized, changes = standardize_molecule(smiles)
    if changes == "Invalid SMILES":
        return {
            "Standardized_IL_SMILES": standardized,
            "Changes": changes,
            "IL_cation_SMILES": "",
            "IL_anion_SMILES": "",
            "IL_component_number": 0,
            "IL_MW": np.nan,
            "Contains_Metal": False,
        }

    df = pd.DataFrame({"Standardized_IL_SMILES": [standardized]})
    reorder_charged_species(df, smiles_col="Standardized_IL_SMILES")
    standardized = df.loc[0, "Standardized_IL_SMILES"]

    parts = standardized.split(".")
    cations = [p for p in parts if re.search(r"\+[0-9]*", p)]
    anions = [p for p in parts if re.search(r"\-[0-9]*", p)]

    mol = MolFromSmiles(standardized)
    mw = Descriptors.MolWt(mol) if mol is not None else np.nan
    contains_metal = any(atom.GetSymbol(
    ) in METALS for atom in mol.GetAtoms()) if mol is not None else False

    return {
        "Standardized_IL_SMILES": standardized,
        "Changes": changes,
        "IL_cation_SMILES": ".".join(cations),
        "IL_anion_SMILES": ".".join(anions),
        "IL_component_number": len(parts),
        "IL_MW": mw,
        "Contains_Metal": contains_metal,
    }


def _name_variants(name: str) -> list[str]:
    """A couple of textual variants worth trying against PubChem's exact-name matcher.

    ILThermo sometimes wraps a substituent in square brackets (e.g.
    ``"bis[(trifluoromethyl)sulfonyl]imide"``) where PubChem's own naming uses plain
    parentheses instead (``"bis(trifluoromethylsulfonyl)imide"``) -- confirmed empirically
    to make the difference between a 404 and a hit for several real ILThermo compound names.
    """
    name = name.strip()
    variants = [name]
    bracket_variant = name.replace("[", "(").replace("]", ")")
    if bracket_variant != name:
        variants.append(bracket_variant)
    return variants


def resolve_smiles_by_name(name: str, timeout: float = 10.0) -> str | None:
    """Look up a compound's isomeric SMILES on PubChem by its chemical name.

    Used as a fallback when the bundled ``keydata/smiles.csv`` id-based lookup misses (see
    the module-level "Known limitation" note above) -- ILThermo's compound names are usually
    well-formed enough for PubChem's exact-name search to resolve directly, recovering rows
    that would otherwise be dropped as unparseable. Returns ``None`` if nothing resolves (no
    match, an ambiguous multi-compound match, or a network error) rather than raising, since
    this is a best-effort fallback -- returning a wrong structure would be worse than dropping
    the row.
    """
    if not name or not name.strip():
        return None
    for variant in _name_variants(name):
        url = PUBCHEM_SMILES_URL.format(name=quote(variant, safe=""))
        try:
            response = requests.get(url, timeout=timeout)
        except requests.RequestException:
            continue
        if response.status_code != 200 or not response.text.strip():
            continue
        lines = [line.strip()
                 for line in response.text.strip().splitlines() if line.strip()]
        if len(lines) == 1:
            return lines[0]
        # More than one match means PubChem resolved the name ambiguously (a fuzzy/synonym
        # match against several distinct compounds, not several representations of the same
        # one) -- confirmed empirically: a full IL salt name matched several different real
        # salts with different alkyl chain lengths, and blindly taking the first line silently
        # substituted the wrong structure. Reject rather than guess.
    return None


def formula_matches(smiles: str, expected_formula: str) -> bool:
    """Check whether ``smiles`` has the same molecular formula as ``expected_formula``.

    Used to cross-check a PubChem name lookup against ILThermo's own reported formula for
    that compound before trusting it: PubChem's name search can return exactly one,
    high-confidence-looking match that is nonetheless the *wrong* compound -- confirmed
    empirically for a real IL salt name, where "1-hexyl-3-methylimidazolium
    bis[(trifluoromethyl)sulfonyl]imide" resolved to a structurally similar but wrong
    salt (an octyl chain instead of hexyl, and the wrong anion regiochemistry) with no
    ambiguity signal in the response at all. An empty ``expected_formula`` (no ground truth
    to check against) is treated as a pass, since there's nothing to contradict.
    """
    if not expected_formula or not expected_formula.strip():
        return True
    mol = MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        actual_formula = rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return False

    def _strip_charge_suffix(formula: str) -> str:
        return re.sub(r"[+-]\d*$", "", formula.strip())

    return _strip_charge_suffix(actual_formula) == _strip_charge_suffix(expected_formula)


def flag_duplicates(
    df: pd.DataFrame,
    key_cols: list[str],
    value_col: str,
    conflict_tolerance_frac: float = 0.02,
) -> pd.Series:
    """Flag rows as ``unique`` / ``duplicate_dropped`` / ``conflicting_duplicate``.

    Rows sharing the same ``key_cols`` are grouped. Within a group, if every ``value_col``
    reading agrees within ``conflict_tolerance_frac`` (relative) of the group median, the
    first row is kept as ``unique`` and the rest flagged ``duplicate_dropped``. Otherwise
    every row in the group is flagged ``conflicting_duplicate`` -- conflicting measurements
    are surfaced for a human to resolve, not silently averaged away.
    """
    flags = pd.Series("unique", index=df.index, dtype=object)
    for _, group in df.groupby(key_cols, dropna=False):
        if len(group) == 1:
            continue
        values = group[value_col].astype(float)
        median = values.median()
        if median == 0 or median != median:  # zero or NaN median: fall back to absolute tolerance
            within_tolerance = (
                values - median).abs() <= conflict_tolerance_frac
        else:
            within_tolerance = ((values - median).abs() /
                                abs(median)) <= conflict_tolerance_frac
        if within_tolerance.all():
            idxs = list(group.index)
            flags.loc[idxs[1:]] = "duplicate_dropped"
        else:
            flags.loc[group.index] = "conflicting_duplicate"
    return flags


def filter_conditions(
    df: pd.DataFrame,
    pressure_col: str = "Pressure (kPa)",
    pressure_range: tuple[float, float] | None = (90.0, 110.0),
    allow_missing_pressure: bool = True,
    temp_col: str = "Temperature (K)",
    temp_range: tuple[float, float] = (253.0, 573.0),
) -> pd.DataFrame:
    """Keep only rows within the given pressure/temperature ranges.

    ``pressure_range`` defaults to the project's documented near-atmospheric scope
    (90-110 kPa); pass ``None`` to skip pressure filtering entirely (useful for properties
    where pressure isn't a relevant condition, or when you want every pressure available).
    Both ranges are parameters, not hardcoded constants, so callers can widen or narrow them
    per property/solvent as needed.
    """
    mask = df[temp_col].between(*temp_range)
    if pressure_range is not None and pressure_col in df.columns:
        pressure_mask = df[pressure_col].between(*pressure_range)
        if allow_missing_pressure:
            pressure_mask = pressure_mask | df[pressure_col].isna()
        mask = mask & pressure_mask
    return df[mask].copy()


def filter_property_range(df: pd.DataFrame, value_col: str, plausible_range: tuple[float, float]) -> pd.DataFrame:
    """Keep only rows whose ``value_col`` falls within a caller-supplied physical sanity range."""
    return df[df[value_col].between(*plausible_range)].copy()


def _parse_numeric(value) -> float:
    """Parse a raw ILThermo cell value as a float.

    :func:`qspr_il.data.ionics.client.flatten_idset` joins a ``[value, uncertainty]`` cell
    into a single ``"value;uncertainty"`` string (e.g. ``"914.7;1.8"``); this takes just the
    primary value. Returns NaN for anything that isn't parseable rather than raising, since a
    single malformed measurement shouldn't abort building the whole dataset.
    """
    if value is None:
        return np.nan
    primary = str(value).split(";", 1)[0].strip()
    try:
        return float(primary)
    except ValueError:
        return np.nan


def _find_column(columns, pattern: str) -> str | None:
    regex = re.compile(pattern, re.IGNORECASE)
    for col in columns:
        if regex.search(col):
            return col
    return None


def _component_indices(columns) -> list[int]:
    indices = set()
    for col in columns:
        m = re.match(r"component (\d+) (idout|name|SMILES)$", col)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def _is_property_value_column(col: str, exclude: set[str]) -> bool:
    """Whether ``col`` could be the raw measurement column for whatever property a dataset
    is about (found by elimination, since it's whatever remains after excluding known
    condition/provenance/component columns).

    ILThermo idsets are per-property: besides the measurement conditions (temperature,
    pressure, composition) and the component/provenance columns, the remaining data column(s)
    are the property reading itself. Identifying it this way (by elimination) rather than by
    name keeps this pipeline generic across all ~56 ILThermo properties instead of needing a
    hardcoded column-name pattern per property.
    """
    if col in exclude or col in ("idset", "author"):
        return False
    if re.match(r"^component \d+ (idout|name|SMILES)$", col):
        return False
    skip_patterns = [r"^Temperature", r"^Pressure",
                     r"^Mole fraction", r"^Weight fraction"]
    return not any(re.match(p, col, re.IGNORECASE) for p in skip_patterns)


def _resolve_component_smiles(name: str, id_based_smiles, cache: dict, use_pubchem_fallback: bool) -> str:
    """Return a usable SMILES for one component, falling back to a PubChem name lookup when
    the id-based ``addSmiles`` value is unparseable (e.g. it's still a bare molecular formula
    because the compound id wasn't in the bundled ``keydata/smiles.csv`` table).

    That fallback value -- ILThermo's own reported molecular formula for the compound -- is
    also the ground truth :func:`formula_matches` cross-checks the PubChem candidate against
    before trusting it (see that function's docstring for why this check matters: a single,
    unambiguous-looking PubChem match can still be the wrong compound).

    ``cache`` is a caller-supplied dict reused across a whole :func:`build_curated_dataset`
    call so each distinct compound name is only looked up once, no matter how many
    measurement rows reference it.
    """
    text = str(
        id_based_smiles) if id_based_smiles == id_based_smiles else ""  # NaN-safe
    if text and MolFromSmiles(text) is not None:
        return text
    if not use_pubchem_fallback:
        return text
    key = str(name).strip().lower()
    if key not in cache:
        candidate = resolve_smiles_by_name(str(name))
        if candidate and not formula_matches(candidate, text):
            candidate = None
        cache[key] = candidate
    return cache[key] or text


def build_curated_dataset(
    raw_df: pd.DataFrame,
    property_name: str,
    solvent_name: str | None,
    pressure_range: tuple[float, float] | None = (90.0, 110.0),
    temp_range: tuple[float, float] = (253.0, 573.0),
    property_range: tuple[float, float] | None = None,
    use_pubchem_fallback: bool = True,
    progress_callback=None,
) -> pd.DataFrame:
    """Turn a raw, flattened+SMILES-joined ILThermo CSV (concatenated across idsets) into a
    curated DataFrame with the generic :data:`GENERIC_MIXTURE_COLUMNS`/:data:`GENERIC_PURE_COLUMNS`
    schema.

    ``raw_df`` is the concatenation of one or more CSVs produced by
    ``ionics.convert2csv`` + ``ionics.addSmiles``, all for the same property. ``property_name``
    is stored verbatim in the output's ``Property`` column (typically the resolved display
    name from :func:`resolve_property_display_name`). ``solvent_name`` is one of
    :data:`KNOWN_SOLVENTS`, or ``None`` for a pure-IL dataset. ``use_pubchem_fallback`` controls
    whether unresolved compound names are looked up on PubChem (see
    :func:`resolve_smiles_by_name`); set to ``False`` to keep this fully offline (compounds that
    the bundled ``keydata/smiles.csv`` table misses are then dropped, as before).
    ``progress_callback``, if given, is called with a one-line status string at each cleaning
    stage (standardization, filtering, deduplication) -- used by the CLI and Streamlit app to
    show what the cleaning step is actually doing.
    """

    def _report(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    is_pure = solvent_name is None
    columns = raw_df.columns

    # NOTE: raw_df is typically the concatenation of several idsets, and ILThermo embeds the
    # specific compound name directly in condition column headers (e.g. "Mole fraction of
    # 1-butyl-3-methylimidazolium tetrafluoroborate"), so *different* idsets in the same batch
    # generally use *differently-named* columns for what is semantically the same field. After
    # concatenation, a row from one idset has real data in its own idset's columns and NaN in
    # every other idset's differently-named columns. So column selection must happen per row
    # (which of the several candidate columns is actually populated for *this* row), not once
    # globally -- picking one column name for the whole batch silently mixes up unrelated
    # compounds' data (confirmed empirically: rows ended up with another row's compound name
    # and an all-NaN mole fraction).
    temp_cols = [c for c in columns if re.match(
        r"^Temperature", c, re.IGNORECASE)]
    pressure_cols = [c for c in columns if re.match(
        r"^Pressure", c, re.IGNORECASE)]
    mole_fraction_cols = [c for c in columns if re.match(
        r"^Mole fraction of ", c, re.IGNORECASE)]
    component_idxs = _component_indices(columns)
    component_cols = {
        f"component {i} {suffix}" for i in component_idxs for suffix in ("idout", "name", "SMILES")
    }
    exclude = {"idset", "author"} | component_cols
    property_cols = [
        c for c in columns if _is_property_value_column(c, exclude)]
    if not temp_cols or not property_cols:
        raise ValueError(
            "raw_df is missing a recognizable Temperature or property-value column.")

    _report(
        f"Standardizing SMILES and resolving component structures for {len(raw_df)} raw rows...")
    rows = []
    smiles_cache: dict[str, str | None] = {}
    for _, row in raw_df.iterrows():
        temp_col = next((c for c in temp_cols if pd.notna(row.get(c))), None)
        pressure_col = next(
            (c for c in pressure_cols if pd.notna(row.get(c))), None)
        mole_fraction_col = next(
            (c for c in mole_fraction_cols if pd.notna(row.get(c))), None)
        property_col = next(
            (c for c in property_cols if pd.notna(row.get(c))), None)
        if temp_col is None or property_col is None:
            continue  # this row's own idset didn't provide a usable temperature/property reading

        components = []
        for idx in component_idxs:
            name_col, smiles_col, idout_col = (
                f"component {idx} name",
                f"component {idx} SMILES",
                f"component {idx} idout",
            )
            if name_col not in row or smiles_col not in row:
                continue
            components.append(
                {"idx": idx, "name": row[name_col], "smiles": row[smiles_col], "idout": row.get(
                    idout_col, "")}
            )

        if is_pure:
            if len(components) != 1:
                continue
            il_component = components[0]
            solvent_component = None
        else:
            solvent_component = next(
                (c for c in components if str(c["name"]).strip(
                ).lower() == solvent_name.lower()), None
            )
            il_component = next(
                (c for c in components if c is not solvent_component), None)
            if solvent_component is None or il_component is None:
                continue
            # Resolve the solvent's SMILES by name (see SOLVENT_SMILES_BY_NAME) rather than
            # trusting addSmiles's id-based lookup, which was found to miss known solvents
            # when ILThermo's live component id doesn't match the bundled keydata table's id
            # for that same compound. For solvents outside that small known set, fall back to
            # the same PubChem name resolution used for the IL side.
            known_solvent_smiles = SOLVENT_SMILES_BY_NAME.get(
                str(solvent_component["name"]).strip().lower())
            if known_solvent_smiles:
                solvent_component = {**solvent_component,
                                     "smiles": known_solvent_smiles}
            else:
                resolved_solvent_smiles = _resolve_component_smiles(
                    solvent_component["name"], solvent_component["smiles"], smiles_cache, use_pubchem_fallback
                )
                solvent_component = {**solvent_component,
                                     "smiles": resolved_solvent_smiles}

        il_smiles = _resolve_component_smiles(
            il_component["name"], il_component["smiles"], smiles_cache, use_pubchem_fallback
        )
        molecular_fields = compute_molecular_fields(il_smiles)
        if molecular_fields["Changes"] == "Invalid SMILES":
            continue  # unparseable SMILES: neither the bundled keydata table nor PubChem resolved it

        record = {
            "setid": row.get("idset", ""),
            "Temperature (K)": _parse_numeric(row.get(temp_col)),
            "Pressure (kPa)": _parse_numeric(row.get(pressure_col)) if pressure_col else np.nan,
            "Property": property_name,
            "Property_value": _parse_numeric(row.get(property_col)),
            "reference": row.get("author", ""),
            "phases": (property_col.split(" - ")[-1] if " - " in property_col else ""),
            "IL_ID": il_component["idout"],
            "IL_SMILES": il_smiles,
            "IL_name": il_component["name"],
            **molecular_fields,
        }

        if not is_pure:
            fraction_value = _parse_numeric(
                row.get(mole_fraction_col)) if mole_fraction_col else np.nan
            fraction_component_name = (
                re.sub(r"^Mole fraction of ", "", mole_fraction_col,
                       flags=re.IGNORECASE) if mole_fraction_col else ""
            )
            solvent_mw_mol = MolFromSmiles(solvent_component["smiles"])
            record.update(
                {
                    "Solvent_ID": solvent_component["idout"],
                    "Solvent_SMILES": solvent_component["smiles"],
                    "Solvent_name": solvent_component["name"],
                    "Solvent_MW": Descriptors.MolWt(solvent_mw_mol) if solvent_mw_mol is not None else np.nan,
                    "Mole fraction": fraction_value,
                    "Mole fraction of compound name": fraction_component_name,
                    "Weight fraction": np.nan,
                    "Weight fraction of compound name": np.nan,
                }
            )
            if fraction_component_name.strip().lower() == solvent_name.lower() and fraction_value == fraction_value:
                mole_fraction_of_solvent = fraction_value
            elif fraction_value == fraction_value:
                mole_fraction_of_solvent = 1.0 - fraction_value
            else:
                mole_fraction_of_solvent = np.nan
            record["Mole fraction of solvent"] = mole_fraction_of_solvent
            # The IL's own mole fraction -- what the trained models expect (matches this
            # project's established "Mole_fraction_IL" convention, e.g.
            # datasets/external_test_set.csv) -- regardless of which of the two components
            # ILThermo's raw "Mole fraction" column happened to be reported against.
            record["Mole_fraction_IL"] = (
                1.0 - mole_fraction_of_solvent if mole_fraction_of_solvent == mole_fraction_of_solvent else np.nan
            )

        rows.append(record)

    target_columns = GENERIC_PURE_COLUMNS if is_pure else GENERIC_MIXTURE_COLUMNS
    dropped_unresolvable = len(raw_df) - len(rows)
    _report(
        f"Resolved {len(rows)} of {len(raw_df)} raw rows"
        + (f" ({dropped_unresolvable} dropped: unresolvable SMILES)" if dropped_unresolvable else "")
        + "."
    )
    curated = pd.DataFrame(rows)
    if curated.empty:
        _report("No rows remained after standardization -- nothing to clean.")
        return pd.DataFrame(columns=target_columns)

    curated["Record_ID"] = [f"{r['setid']}_{i}" for i, r in enumerate(rows)]

    before_conditions = len(curated)
    curated = filter_conditions(
        curated, pressure_range=pressure_range, temp_range=temp_range)
    if property_range is not None:
        curated = filter_property_range(
            curated, "Property_value", property_range)
    _report(
        f"{len(curated)} of {before_conditions} rows remain after temperature/pressure/property-range filtering."
    )

    key_cols = ["IL_SMILES", "Temperature (K)"] if is_pure else [
        "IL_SMILES", "Solvent_name", "Temperature (K)"]
    curated["Data_quality_flag"] = flag_duplicates(
        curated, key_cols, "Property_value")
    duplicate_count = int(
        (curated["Data_quality_flag"] == "duplicate_dropped").sum())
    conflicting_count = int(
        (curated["Data_quality_flag"] == "conflicting_duplicate").sum())
    curated = curated[curated["Data_quality_flag"] != "duplicate_dropped"]
    _report(
        f"Deduplication: dropped {duplicate_count} exact duplicate(s), flagged {conflicting_count} "
        f"conflicting duplicate(s) for review. {len(curated)} curated rows remain."
    )

    for col in target_columns:
        if col not in curated.columns:
            curated[col] = np.nan
    return curated.reindex(columns=target_columns).reset_index(drop=True)


def fetch_curated_dataset(
    property_query: str,
    solvent_name: str | None,
    data_root=None,
    max_datasets: int | None = None,
    year: str = "",
    author: str = "",
    keyword: str = "",
    progress_callback=None,
    **filter_kwargs,
) -> pd.DataFrame:
    """Download, flatten, SMILES-join, and clean ILThermo data for one property/solvent combo.

    This is the single high-level function a user calls to (re)generate a curated dataset for
    *any* ILThermo property, e.g. ``fetch_curated_dataset("density", "ethanol")`` or
    ``fetch_curated_dataset("viscosity", None)`` for a pure-IL viscosity dataset.
    ``property_query`` can be either the hyphenated short name from
    :func:`list_available_properties` (e.g. ``"refractive-index"``) or ILThermo's own display
    name (e.g. ``"Refractive index"``) -- it's resolved against the live search results via
    :func:`resolve_property_display_name`. Raises :class:`ValueError` if nothing matches.
    Pass ``max_datasets`` to cap how many ILThermo datasets are downloaded (useful for quick,
    rate-limit-friendly runs). ``year``/``author``/``keyword`` are the same server-side
    ILThermo search filters ``pyionics`` exposes (forwarded to
    :func:`qspr_il.data.ionics.client.getIdsets` as ``year``/``auth``/``keyw``): a publication
    year, an author surname, and a free-text keyword, each narrowing the ``ilsearch`` query
    *before* the client-side property-name filtering happens. And ``pressure_range``/``temp_range``/``property_range``/
    ``use_pubchem_fallback`` (via ``filter_kwargs``, forwarded to :func:`build_curated_dataset`)
    to restrict conditions or disable the PubChem name-lookup fallback (see
    :func:`resolve_smiles_by_name`). ``progress_callback``, if given, is called with a one-line
    status string at each stage of the fetch (search, download, convert, resolve SMILES) and
    forwarded into :func:`build_curated_dataset` for the cleaning stages too -- used by the CLI
    and Streamlit app to show what's actually happening, not just a single opaque spinner.
    """

    def _report(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    ncmp = "1" if solvent_name is None else "2"
    data_root_path = ionics_client.get_data_dir(data_root)

    active_filters = [
        f"{label}={value}"
        for label, value in (("year", year), ("author", author), ("keyword", keyword))
        if value
    ]
    _report(
        f"Searching ILThermo for '{property_query}'"
        + (f" ({', '.join(active_filters)})" if active_filters else "")
        + "..."
    )
    idsets_path = ionics_client.getIdsets(
        prop=None, ncmp=ncmp, year=year, auth=author, keyw=keyword, data_root=data_root_path)
    idsets_json = json.load(open(idsets_path))
    display_name = resolve_property_display_name(idsets_json, property_query)
    if display_name is None:
        available = sorted({row[2]
                           for row in idsets_json.get("res", []) if row[2]})
        raise ValueError(
            f"No property matching {property_query!r} found in the current search results. "
            f"Available properties for this search: {available}"
        )
    matching = [row[0]
                for row in idsets_json.get("res", []) if row[2] == display_name]
    total_matching = len(matching)
    if max_datasets is not None:
        matching = matching[:max_datasets]
    _report(
        f"Resolved property: '{display_name}'. Found {total_matching} matching dataset(s)"
        + (f", downloading the first {len(matching)}." if max_datasets is not None else ".")
    )

    safe_property = re.sub(r"[^a-zA-Z0-9]+", "_",
                           display_name.strip().lower()).strip("_")
    folder_name = f"{safe_property}_{'pure' if solvent_name is None else solvent_name}_data"

    def _download_progress(i: int, total: int, setid: str) -> None:
        _report(f"Downloading dataset {i}/{total} ({setid})...")

    ionics_client.download_idsets(
        matching, output_dir=data_root_path / folder_name, progress_callback=_download_progress
    )
    _report("Converting downloaded data to CSV...")
    ionics_client.convert2csv(folder_name=folder_name,
                              data_root=data_root_path)
    _report("Resolving component SMILES (bundled lookup table, PubChem fallback for the rest)...")
    ionics_client.addSmiles(
        folder_name=f"csv_{folder_name}", data_root=data_root_path)

    smiles_dir = data_root_path / f"smiles_csv_{folder_name}"
    frames = [pd.read_csv(f) for f in smiles_dir.glob(
        "*.csv")] if smiles_dir.exists() else []
    if not frames:
        _report("No data files were produced by the download/convert step.")
        target_columns = GENERIC_PURE_COLUMNS if solvent_name is None else GENERIC_MIXTURE_COLUMNS
        return pd.DataFrame(columns=target_columns)

    raw_df = pd.concat(frames, ignore_index=True)
    return build_curated_dataset(raw_df, display_name, solvent_name, progress_callback=progress_callback, **filter_kwargs)
