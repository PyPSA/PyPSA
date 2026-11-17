# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import mathspec
import pytest

import pypsa
from pypsa.optimization.spec._render import (
    PLACEHOLDERS,
    _load,
    _merge,
    formulated,
    renamer,
    render,
)
from pypsa.optimization.spec.build import SPEC, spec
from pypsa.optimization.spec.sources import (
    _SOURCES,
    NetworkSources,
    _providers,
    source,
)


def test_every_declared_trait_has_one_file():
    declared = {t for c in pypsa.Network().components.values() for t in c._formulation}
    stated = {path.name.removesuffix(".yaml") for path in (SPEC / "traits").iterdir()}
    assert declared == stated


@pytest.mark.parametrize(
    ("component", "name", "renamed"),
    [
        ("StorageUnit", "COMPONENT_NOM_ext", "StorageUnit_p_nom_ext"),
        ("Store", "COMPONENT_LEVEL_initial", "Store_e_initial"),
        ("Link", "COMPONENT_port_UNIT", "Link_port_Link"),
        ("Line", "COMPONENT-ext-NOM-lower", "Line-ext-s_nom-lower"),
        ("Store", "COMPONENT_has_unit_years", "Store_has_unit_years"),
        ("Generator", "snapshot_period", "snapshot_period"),
    ],
)
def test_a_placeholder_takes_the_value_of_the_component(component, name, renamed):
    assert renamer(pypsa.Network().components[component])(name) == renamed


def test_a_placeholder_without_value_fails():
    with pytest.raises(KeyError, match="POWER"):
        renamer(pypsa.Network().components["Line"])("COMPONENT_POWER")


def test_no_placeholder_is_a_mathspec_keyword():
    assert PLACEHOLDERS.isdisjoint({"AND", "OR", "NOT"})


def test_a_trait_description_takes_the_names_of_the_component(tmp_path):
    (tmp_path / "traits").mkdir()
    (tmp_path / "traits" / "multiport.yaml").write_text(
        "dimensions:\n  UNIT_port: {}\nparameters:\n  COMPONENT_COEF_x:\n"
        "    dims: [UNIT_port]\n    description: the `COEF` of a `UNIT` port\n"
    )
    spec = render(tmp_path)[0]
    assert spec.parameters["Link_efficiency_x"].description == (
        "the `efficiency` of a `Link` port"
    )
    assert spec.parameters["Process_rate_x"].description == (
        "the `rate` of a `Process` port"
    )


@pytest.mark.parametrize(
    ("trait", "component"),
    [(t, c.name) for c in formulated(pypsa.Network()) for t in c._formulation],
)
def test_a_trait_is_a_spec_of_its_own(trait, component):
    c = pypsa.Network().components[component]
    rename = renamer(c)
    spec = mathspec.rename(_load(SPEC / "traits" / f"{trait}.yaml", rename)[0], rename)
    assert c._unit_dim in spec.dimensions


@pytest.mark.parametrize(
    ("rule", "error"), [("none", None), ("largest", "rules \\['largest'\\]")]
)
def test_a_constraint_states_its_scaling_rule_apart_from_the_spec(
    tmp_path, rule, error
):
    path = tmp_path / "fragment.yaml"
    path.write_text(
        f"constraints:\n  COMPONENT_x:\n    scaling: {rule}\n    dims: []\n"
    )
    rename = renamer(pypsa.Network().components["Generator"])
    if error:
        with pytest.raises(ValueError, match=error):
            _load(path, rename)
        return
    data, _, scalings = _load(path, rename)
    assert scalings == {"Generator_x": rule}
    assert data["constraints"]["COMPONENT_x"] == {"dims": []}


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


@pytest.mark.parametrize("key", ["snapshot_period", "COMPONENT_active"])
def test_a_source_key_takes_one_derivation(key):
    with pytest.raises(ValueError, match="already"):
        source(key)(lambda s, name: None)


def test_a_source_name_takes_one_derivation(monkeypatch):
    monkeypatch.setitem(_SOURCES, "Generator_active", lambda s, name: None)
    with pytest.raises(ValueError, match="Generator_active has two derivations"):
        _providers(formulated(pypsa.Network()))


@pytest.mark.parametrize(
    ("name", "error", "match"),
    [
        ("Store_e_nom_maxx", NotImplementedError, "Store has no attribute e_nom_maxx"),
        ("Store_marginal_cost_storage", ValueError, "varies by snapshot"),
    ],
)
def test_a_parameter_of_a_component_without_units_is_checked(
    ac_dc_network, name, error, match
):
    n = ac_dc_network
    assert n.c.stores.static.empty
    s = mathspec.override(spec(), [{"parameters": {name: {"dims": ["Store"]}}}])
    with pytest.raises(error, match=match):
        NetworkSources(n, s, n.snapshots, n.snapshots)[name]


@pytest.mark.parametrize("runtime_verification", [False, True])
def test_a_parameter_without_the_dim_of_its_units_is_refused(
    ac_dc_network, runtime_verification
):
    n = ac_dc_network
    s = mathspec.override(spec(), [{"parameters": {"Generator_weight": {"dims": []}}}])
    sources = NetworkSources(n, s, n.snapshots, n.snapshots)
    verification = pypsa.option_context(
        "debug.runtime_verification", runtime_verification
    )
    with verification, pytest.raises(ValueError, match="are not declared"):
        sources["Generator_weight"]
