# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import mathspec
import pytest

import pypsa
from pypsa.optimization.spec._render import _load, _merge, formulated, renamer
from pypsa.optimization.spec.build import SPEC
from pypsa.optimization.spec.sources import _SOURCES, _providers, source


def test_every_declared_trait_has_one_file():
    declared = {t for c in pypsa.Network().components.values() for t in c._formulation}
    stated = {path.name.removesuffix(".yaml") for path in (SPEC / "traits").iterdir()}
    assert declared == stated


@pytest.mark.parametrize(
    ("component", "name", "renamed"),
    [
        ("StorageUnit", "Unit_NOM_ext", "StorageUnit_p_nom_ext"),
        ("Store", "Unit_LEVEL_initial", "Store_e_initial"),
        ("Link", "Unit_output_UNIT", "Link_output_link"),
        ("Line", "Unit-ext-NOM-lower", "Line-ext-s_nom-lower"),
        ("Store", "Unit_has_unit_years", "Store_has_unit_years"),
        ("Generator", "snapshot_period", "snapshot_period"),
    ],
)
def test_a_placeholder_takes_the_value_of_the_component(component, name, renamed):
    assert renamer(pypsa.Network().components[component])(name) == renamed


def test_a_placeholder_without_value_fails():
    with pytest.raises(KeyError, match="P"):
        renamer(pypsa.Network().components["Line"])("Unit_P")


@pytest.mark.parametrize(
    ("trait", "component"),
    [(t, c.name) for c in formulated(pypsa.Network()) for t in c._formulation],
)
def test_a_trait_is_a_spec_of_its_own(trait, component):
    c = pypsa.Network().components[component]
    rename = renamer(c)
    spec = mathspec.rename(_load(SPEC / "traits" / f"{trait}.yaml", rename)[0], rename)
    assert c._unit_dim in spec.dimensions


SHARED = {"dimensions": {"bus": {"description": "nodes"}}, "parameters": {"p": {}}}


def test_a_merge_drops_given_and_keeps_a_shared_entry_stated_alike():
    spec = {k: dict(v) for k, v in SHARED.items()}
    _merge(spec, {"dimensions": {"bus": {}}, "given": {"parameters": {"q": {}}}}, "f")
    assert spec == SHARED


@pytest.mark.parametrize(
    ("fragment", "section"),
    [
        ({"dimensions": {"bus": {"dtype": "int"}}}, "dimensions"),
        ({"parameters": {"p": {}}}, "parameters"),
    ],
)
def test_a_merge_refuses_an_entry_stated_twice(fragment, section):
    with pytest.raises(ValueError, match=f"f declares \\['\\w+'\\] in `{section}`"):
        _merge({k: dict(v) for k, v in SHARED.items()}, fragment, "f")


@pytest.mark.parametrize("key", ["snapshot_period", "Unit_active"])
def test_a_source_key_takes_one_derivation(key):
    with pytest.raises(ValueError, match="already"):
        source(key)(lambda s, name: None)


def test_a_source_name_takes_one_derivation(monkeypatch):
    monkeypatch.setitem(_SOURCES, "Generator_active", lambda s, name: None)
    with pytest.raises(ValueError, match="Generator_active has two derivations"):
        _providers(formulated(pypsa.Network()))
