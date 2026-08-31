import json

from qspr_il.data.ionics import client


def test_getidsets_builds_expected_url_and_saves_under_data_root(tmp_path, mock_ilthermo_responses):
    save_path = client.getIdsets(prop="dens", data_root=tmp_path)
    assert save_path.exists()
    assert str(tmp_path) in str(save_path)
    assert any(
        "ilsearch" in url and "prp=" in url for url in mock_ilthermo_responses)


def test_getidsets_unknown_property_raises(tmp_path, mock_ilthermo_responses):
    import pytest

    with pytest.raises(ValueError):
        client.getIdsets(prop="not_a_real_property", data_root=tmp_path)


def test_getdata_downloads_each_set_under_data_root(tmp_path, mock_ilthermo_responses):
    idset_dir = client.getData(prop="dens", ncmp="1", data_root=tmp_path)
    files = list(idset_dir.glob("idset_*.json"))
    assert len(files) == 1
    assert str(tmp_path) in str(idset_dir)


def test_download_idsets_reports_progress(tmp_path, mock_ilthermo_responses):
    calls = []
    client.download_idsets(["a", "b"], output_dir=tmp_path,
                           progress_callback=lambda i, total, setid: calls.append((i, total, setid)))
    # Downloads run concurrently, so completion order isn't fixed -- but every set is
    # reported exactly once, the running count goes 1..total, and total is constant.
    assert [i for i, _, _ in calls] == [1, 2]
    assert {total for _, total, _ in calls} == {2}
    assert {setid for _, _, setid in calls} == {"a", "b"}
    assert sorted(p.name for p in tmp_path.glob("idset_*.json")) == [
        "idset_a.json", "idset_b.json"]


def test_flatten_idset_strips_sub_tags_and_builds_component_columns():
    jdata = {
        "dhead": [["Temperature", "K"], ["Density", "kg/m3"]],
        "data": [[298.15, "10<SUB>3</SUB>"]],
        "components": [{"idout": "C001", "name": "Water<SUB>2</SUB>", "formula": "O"}],
        "ref": "Doe, J., 2021",
    }
    header, rows = client.flatten_idset(jdata, idset="42")
    assert header == [
        "idset",
        "author",
        "Temperature - K",
        "Density - kg/m3",
        "component 1 idout",
        "component 1 name",
        "component 1 formula",
    ]
    assert rows == [["42", "Doe, J., 2021 et al.",
                     "298.15", "103", "C001", "Water2", "O"]]


def test_flatten_idset_empty_when_missing_dhead_or_data():
    header, rows = client.flatten_idset({}, idset="1")
    assert header == []
    assert rows == []


def test_addsmiles_resolves_keydata_regardless_of_cwd(tmp_path, monkeypatch):
    # Regression test: keydata resolution must not depend on the caller's cwd.
    monkeypatch.chdir(tmp_path)

    data_root = tmp_path / "data"
    csv_dir = data_root / "csv_sample"
    csv_dir.mkdir(parents=True)
    sample_csv = csv_dir / "idset_1.csv"
    sample_csv.write_text(
        "idset,author,component 1 idout,component 1 name,component 1 formula\n"
        "1,Doe et al.,B3QX,Water,O\n"
    )

    client.addSmiles(folder_name="csv_sample", data_root=data_root)

    output_file = data_root / "smiles_csv_sample" / "idset_1.csv"
    assert output_file.exists()
    content = output_file.read_text()
    assert "component 1 SMILES" in content


def test_get_data_dir_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert client.get_data_dir() == tmp_path / "data"


def test_get_data_dir_honors_explicit_root(tmp_path):
    custom = tmp_path / "custom"
    assert client.get_data_dir(custom) == custom.resolve()
