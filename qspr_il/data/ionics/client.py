"""ILThermo (NIST) data-acquisition client — vendored from the ``pyionics`` package.

``pyionics`` is being deprecated as an external dependency; its functionality now
lives here permanently, with two fixes over the original:

1. The bundled ``keydata`` lookup tables (``property_idsets.csv``, ``smiles.csv``)
   resolve relative to this module's own location, not two independently-hardcoded
   ``os.path.dirname(__file__)`` calls.
2. The output directory (``data_root``) is an explicit, injectable parameter on every
   function instead of each function independently recomputing ``os.getcwd()/"data"``.
   The default (``data_root=None`` -> ``Path.cwd()/"data"``) preserves prior behavior.

This module only fetches and reshapes raw ILThermo data — it performs no
deduplication, consistency checking, or filtering. See :mod:`qspr_il.data.cleaning`
for that.
"""

from __future__ import annotations

import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm

try:  # urllib3 ships with requests; guard only against a very old layout
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    Retry = None

KEYDATA_DIR = Path(__file__).resolve().parent / "keydata"
DEFAULT_PROPERTY_IDSETS = KEYDATA_DIR / "property_idsets.csv"
DEFAULT_SMILES_TABLE = KEYDATA_DIR / "smiles.csv"

_NCMP_LABELS = {"1": "pure", "2": "binary", "3": "triple"}

# ILThermo serves every dataset from one host over HTTPS; a fresh connection (DNS +
# TCP + TLS) per request dominates the cost of downloading a search's worth of sets.
# A pooled session with keep-alive plus a bounded thread pool turns a serial run of
# hundreds of tiny requests into a handful of concurrent, connection-reused ones.
_REQUEST_TIMEOUT = 30
_MAX_DOWNLOAD_WORKERS = 8


@lru_cache(maxsize=1)
def _http_session() -> requests.Session:
    session = requests.Session()
    if Retry is not None:
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        adapter = HTTPAdapter(
            max_retries=retry, pool_connections=_MAX_DOWNLOAD_WORKERS, pool_maxsize=_MAX_DOWNLOAD_WORKERS)
    else:  # pragma: no cover
        adapter = HTTPAdapter(
            pool_connections=_MAX_DOWNLOAD_WORKERS, pool_maxsize=_MAX_DOWNLOAD_WORKERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_data_dir(data_root: str | Path | None = None) -> Path:
    """Root directory for downloaded/converted data. Defaults to ``Path.cwd()/"data"``."""
    return Path(data_root).resolve() if data_root else Path.cwd() / "data"


def _lookup_property_id(prop: str, data_path: str | Path) -> str:
    data_path = Path(data_path)
    if data_path.suffix.lower() == ".json":
        with open(data_path, "r") as f:
            for row in json.load(f):
                if row.get("short") == prop:
                    return row.get("id")
    else:
        with open(data_path, newline="") as csvfile:
            for row in csv.DictReader(csvfile):
                if row["short"] == prop:
                    return row["id"]
    raise ValueError(f"Short property code '{prop}' not found in data file.")


def getIdsets(
    prop=None,
    data_path=None,
    cmp="",
    ncmp="0",
    year="",
    auth="",
    keyw="",
    data_root=None,
) -> Path:
    """Search ILThermo for matching dataset ids and save the raw JSON response."""
    prop = prop or ""
    if not prop and not any([cmp, ncmp != "0", year, auth, keyw]):
        raise ValueError(
            "At least one parameter must be provided (prop or another search parameter).")

    prop_id = None
    if prop:
        prop_id = _lookup_property_id(
            prop, data_path or DEFAULT_PROPERTY_IDSETS)

    url = (
        "https://ilthermo.boulder.nist.gov/ILT2/ilsearch?"
        f"cmp={cmp}&ncmp={ncmp}&year={year}&auth={auth}&keyw={keyw}"
    )
    if prop_id:
        url += f"&prp={prop_id}"
    response = requests.get(url)
    response.raise_for_status()
    content = response.content

    params = []
    if cmp:
        params.append(f"cmp_{cmp}")
    if ncmp and ncmp != "0":
        params.append(f"ncmp_{_NCMP_LABELS.get(str(ncmp), ncmp)}")
    elif ncmp == "0":
        params.append("ncmp_all")
    if year:
        params.append(f"year_{year}")
    if auth:
        params.append(f"auth_{auth}")
    if keyw:
        params.append(f"keyw_{keyw}")
    suffix = "_" + "_".join(params) if params else ""

    data_dir = get_data_dir(data_root) / "idsets"
    data_dir.mkdir(parents=True, exist_ok=True)
    save_path = data_dir / f"{prop}_idsets{suffix}.json"
    try:
        data = response.json()
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except ValueError:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content.decode("utf-8", errors="replace"))

    return save_path


def getData(
    prop=None,
    data_path=None,
    cmp="",
    ncmp="0",
    year="",
    auth="",
    keyw="",
    data_root=None,
) -> Path:
    """Download every dataset matching the search parameters as one JSON file per set."""
    save_path = getIdsets(
        prop=prop, data_path=data_path, cmp=cmp, ncmp=ncmp, year=year, auth=auth, keyw=keyw, data_root=data_root
    )

    prop_label = prop if prop else "all"
    ncmp_label = _NCMP_LABELS.get(
        str(ncmp), "all") if ncmp and ncmp != "0" else "all"

    extra_labels = []
    if cmp:
        extra_labels.append(f"cmp_{cmp}")
    if year:
        extra_labels.append(f"year_{year}")
    if auth:
        extra_labels.append(f"auth_{auth}")
    if keyw:
        extra_labels.append(f"keyw_{keyw}")
    extra_suffix = "_" + "_".join(extra_labels) if extra_labels else ""

    idset_dir = get_data_dir(data_root) / \
        f"{prop_label}_{ncmp_label}_data{extra_suffix}"

    with open(save_path, "r", encoding="utf-8") as f:
        idsets_json = json.load(f)
    idsets = []
    if isinstance(idsets_json, dict) and "res" in idsets_json:
        idsets = [row[0] for row in idsets_json["res"]]

    return download_idsets(idsets, idset_dir)


def _download_one_idset(setid: str, output_dir: Path) -> str:
    """Fetch a single ILThermo dataset and write it as ``idset_<setid>.json``. Returns ``setid``."""
    url = f"https://ilthermo.boulder.nist.gov/ILT2/ilset?set={setid}"
    resp = _http_session().get(url, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    try:
        data = resp.json()
        file_base = None
        if isinstance(data, dict) and "idsets" in data:
            if isinstance(data["idsets"], list) and data["idsets"]:
                entry = data["idsets"][0]
                if isinstance(entry, dict):
                    file_base = entry.get("id") or entry.get("name")
        if not file_base:
            file_base = str(setid)
    except Exception:
        data = {"raw": resp.content.decode("utf-8", errors="replace")}
        file_base = str(setid)
    data["filename"] = file_base
    fname = output_dir / f"idset_{setid}.json"
    with open(fname, "w", encoding="utf-8") as outf:
        json.dump(data, outf, ensure_ascii=False, indent=2)
    return str(setid)


def download_idsets(setids: list[str], output_dir: str | Path, progress_callback=None) -> Path:
    """Download each ILThermo dataset id in ``setids`` and save it as one JSON file per set.

    Extracted out of :func:`getData` so callers (e.g. :mod:`qspr_il.data.cleaning`) can
    filter a broad search's results (by exact property name, since the ``prp`` id lookup
    used by :func:`getIdsets` can go stale against the live ILThermo API) before downloading,
    rather than always fetching every idset a search returns.

    Downloads run concurrently (up to :data:`_MAX_DOWNLOAD_WORKERS` at a time) over a pooled,
    keep-alive session. ``progress_callback``, if given, is called
    ``progress_callback(completed, total, setid)`` as each set finishes (``completed`` counts
    1..total in completion order) -- used by :mod:`qspr_il.data.cleaning` and the Streamlit
    app to show live download progress.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(setids)
    if total == 0:
        return output_dir

    workers = max(1, min(_MAX_DOWNLOAD_WORKERS, total))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download_one_idset, setid, output_dir)
                   for setid in setids]
        for completed, future in enumerate(
            tqdm(as_completed(futures), desc="Downloading idsets", total=total), start=1
        ):
            done_setid = future.result()
            if progress_callback:
                progress_callback(completed, total, done_setid)

    return output_dir


def flatten_idset(jdata: dict, idset: str) -> tuple[list[str], list[list[str]]]:
    """Flatten one downloaded per-set JSON payload into ``(header, rows)`` CSV-ready lists.

    Pure function extracted out of :func:`convert2csv` so the flattening rules
    (``<SUB>`` stripping, author derivation from ``ref``, per-component columns) can be
    tested without touching disk.
    """
    dhead = jdata.get("dhead")
    data_rows = jdata.get("data")
    if not dhead or not data_rows:
        return [], []

    ref = jdata.get("ref")
    if not ref and isinstance(data_rows, list) and data_rows:
        try:
            ref = data_rows[0][1]
        except Exception:
            ref = ""
    author = ""
    if ref:
        if isinstance(ref, str):
            author = f"{ref} et al."
        elif isinstance(ref, dict):
            author_val = ref.get("full") or ref.get("authors") or ""
            if isinstance(author_val, str):
                author = f"{author_val.split(',')[0].strip()} et al."
            elif isinstance(author_val, list) and author_val:
                author = f"{str(author_val[0]).split(',')[0].strip()} et al."

    header = ["idset", "author"]
    for col in dhead:
        col_names = [str(x) for x in col if x]
        header.append(" - ".join(col_names) if col_names else "")

    components = jdata.get("components", [])
    for i in range(len(components)):
        header += [f"component {i + 1} idout",
                   f"component {i + 1} name", f"component {i + 1} formula"]

    def _strip(s: str) -> str:
        return str(s).replace("<SUB>", "").replace("</SUB>", "")

    rows = []
    for row in data_rows:
        flat_row = [idset, author]
        for cell in row:
            cell_str = ";".join(str(x) for x in cell) if isinstance(
                cell, list) else str(cell)
            flat_row.append(_strip(cell_str))
        for comp in components:
            flat_row.extend([_strip(comp.get("idout", "")), _strip(
                comp.get("name", "")), _strip(comp.get("formula", ""))])
        rows.append(flat_row)

    return header, rows


def convert2csv(folder_name="", file_name="", data_root=None) -> None:
    """Convert downloaded per-set JSON files in ``data_root/folder_name`` into CSVs."""
    data_root_path = get_data_dir(data_root)
    folder = Path(folder_name) if folder_name and Path(
        folder_name).is_absolute() else data_root_path / folder_name
    if not folder_name:
        folder = data_root_path
    if not folder.exists():
        print(f"Folder '{folder}' does not exist.")
        return

    output_folder = data_root_path / f"csv_{folder.name}"
    output_folder.mkdir(parents=True, exist_ok=True)

    files = [file_name] if file_name else os.listdir(folder)
    for file in files:
        file_path = folder / file
        if not file_path.is_file():
            continue
        if not file.endswith(".json"):
            continue
        match = re.match(r"idset_(.+)\.json$", file)
        idset = match.group(1) if match else ""
        with open(file_path, "r", encoding="utf-8") as fin:
            try:
                jdata = json.load(fin)
            except Exception as e:
                print(f"Failed to load JSON from {file}: {e}")
                continue
        header, rows = flatten_idset(jdata, idset)
        if not header:
            continue
        output_file = output_folder / (Path(file).stem + ".csv")
        with open(output_file, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(header)
            writer.writerows(rows)


def convert2tsv(folder_name="", file_name="", data_root=None) -> None:
    """Convert downloaded per-set JSON files in ``data_root/folder_name`` into TSVs."""
    data_root_path = get_data_dir(data_root)
    folder = Path(folder_name) if folder_name and Path(
        folder_name).is_absolute() else data_root_path / folder_name
    if not folder_name:
        folder = data_root_path
    if not folder.exists():
        print(f"Folder '{folder}' does not exist.")
        return

    output_folder = data_root_path / f"tsv_{folder.name}"
    output_folder.mkdir(parents=True, exist_ok=True)

    files = [file_name] if file_name else os.listdir(folder)
    for file in files:
        file_path = folder / file
        if not file_path.is_file() or not file.endswith(".json"):
            continue
        with open(file_path, "r", encoding="utf-8") as fin:
            try:
                jdata = json.load(fin)
            except Exception as e:
                print(f"Failed to load JSON from {file}: {e}")
                continue
        # TSV output omits the idset/author columns that convert2csv adds.
        header, rows = flatten_idset(jdata, idset="")
        if not header:
            continue
        header = header[2:]
        rows = [row[2:] for row in rows]
        output_file = output_folder / (Path(file).stem + ".tsv")
        with open(output_file, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout, delimiter="\t")
            writer.writerow(header)
            writer.writerows(rows)


def mergeFiles(folder_name: str, data_root=None) -> Path:
    """Concatenate every CSV in ``data_root/folder_name`` into one merged CSV."""
    data_root_path = get_data_dir(data_root)
    folder_path = data_root_path / folder_name
    if not folder_path.exists():
        print(
            f"Folder '{folder_name}' does not exist in the data directory (expected at: {folder_path}).")
        return None

    merged_data = pd.DataFrame()
    for file in os.listdir(folder_path):
        file_path = folder_path / file
        if not file_path.is_file():
            continue
        try:
            merged_data = pd.concat(
                [merged_data, pd.read_csv(file_path)], ignore_index=True)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    output_file = data_root_path / f"merged_{folder_name}.csv"
    merged_data.to_csv(output_file, index=False)
    return output_file


def addSmiles(folder_name="", file_name="", data_root=None, smiles_path=DEFAULT_SMILES_TABLE) -> None:
    """Replace each ``component {n} formula`` value with a looked-up SMILES string."""
    data_root_path = get_data_dir(data_root)

    if folder_name and not file_name:
        folder = folder_name if Path(
            folder_name).is_absolute() else data_root_path / folder_name
        input_paths = [
            folder / f for f in os.listdir(folder) if f.endswith(".csv")]
        base_folder = Path(folder).name
    elif file_name and not folder_name:
        if not file_name.endswith(".csv"):
            file_name += ".csv"
        file_path = data_root_path / file_name
        if not file_path.exists():
            print(f"File {file_path} not found.")
            return
        input_paths = [file_path]
        base_folder = ""
    elif folder_name and file_name:
        folder = folder_name if Path(
            folder_name).is_absolute() else data_root_path / folder_name
        file_path = Path(folder) / file_name
        if not file_path.exists():
            print(f"File {file_path} not found.")
            return
        input_paths = [file_path]
        base_folder = Path(folder).name
    else:
        print("Please provide either folder_name or file_name.")
        return

    output_folder = data_root_path / \
        (f"smiles_{base_folder}" if base_folder else "smiles")
    output_folder.mkdir(parents=True, exist_ok=True)

    smiles_path = Path(smiles_path)
    if not smiles_path.exists():
        print(f"smiles.csv not found at {smiles_path}")
        return

    smiles_dict = {}
    with open(smiles_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        id_header = next(
            (h for h in reader.fieldnames if "compound id" in h), None)
        smiles_header = next(
            (h for h in reader.fieldnames if "smiles" in h.lower()), None)
        if not id_header or not smiles_header:
            return
        for row in reader:
            smiles_dict[row[id_header]] = row[smiles_header]

    for file_path in input_paths:
        with open(file_path, "r", encoding="utf-8") as fin:
            rows = list(csv.reader(fin))
        if not rows:
            continue
        header = rows[0]
        comp_idout_idxs = [i for i, col in enumerate(
            header) if col.startswith("component ") and col.endswith(" idout")]
        comp_formula_idxs = [
            i for i, col in enumerate(header) if col.startswith("component ") and col.endswith(" formula")
        ]
        for idx in comp_formula_idxs:
            header[idx] = header[idx].replace("formula", "SMILES")

        new_rows = [header]
        for row in rows[1:]:
            new_row = row[:]
            for idx_idout, idx_formula in zip(comp_idout_idxs, comp_formula_idxs):
                comp_id = row[idx_idout].strip()
                new_row[idx_formula] = smiles_dict.get(
                    comp_id, row[idx_formula])
            new_rows.append(new_row)

        output_file = output_folder / Path(file_path).name
        with open(output_file, "w", newline="", encoding="utf-8") as fout:
            csv.writer(fout).writerows(new_rows)
