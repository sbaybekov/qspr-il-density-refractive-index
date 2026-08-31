import pytest

from qspr_il.registry import REGISTRY, find, get, iter_specs


def test_all_eight_keys_present():
    assert set(REGISTRY) == {str(i) for i in range(1, 9)}


def test_is_pure_implies_no_mole_fraction_default():
    for spec in iter_specs():
        if spec.is_pure:
            assert spec.default_mole_fraction_col is None
        else:
            assert spec.default_mole_fraction_col is not None


def test_property_short_is_valid():
    for spec in iter_specs():
        assert spec.property_short in {"dens", "n"}


def test_get_unknown_key_raises():
    with pytest.raises(KeyError):
        get("99")


def test_find_by_property_and_solvent():
    spec = find("Density", "ethanol")
    assert spec.key == "5"
    assert not spec.is_pure


def test_find_unknown_raises():
    with pytest.raises(KeyError):
        find("Viscosity", "ethanol")


def test_model_dirs_are_distinct():
    dirs = [spec.model_dir for spec in iter_specs()]
    assert len(dirs) == len(set(dirs))
