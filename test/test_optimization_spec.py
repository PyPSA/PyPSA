# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import re
from pathlib import Path

import mathspec
import pytest
import yaml

import pypsa
from pypsa.optimization.spec._docs import marker, pages
from pypsa.optimization.spec._render import (
    PLACEHOLDERS,
    WORD,
    _load,
    _merge,
    _snake,
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


@pytest.mark.parametrize(
    "page",
    [
        "index.md",
        *(
            f"traits/{path.name.removesuffix('.yaml')}.md"
            for path in (SPEC / "traits").iterdir()
        ),
        *(f"components/{_snake(c.name)}.md" for c in formulated(pypsa.Network())),
    ],
)
def test_every_trait_and_component_gets_a_page_without_placeholders(page):
    assert not PLACEHOLDERS & set(WORD.findall(pages()[page]))


def test_the_symbol_table_gives_each_dimension_its_own_index():
    table = yaml.safe_load((SPEC / "symbols.yaml").read_text())
    indices = [entry["index"] for entry in table["dimensions"].values()]
    assert len(indices) == len(set(indices))
    assert not set(indices) & set(table["names"].values())


GUIDE = Path(__file__).parents[1] / "docs" / "user-guide" / "optimization"
BACKGROUND = '!!! note "Background"'
MARKER = re.compile(r"<!-- md:(spec[\w-]*)(.*?) -->")


def _loose_math(text: str) -> list[str]:
    """Return the lines with display math outside a `Background` note."""
    loose: list[str] = []
    depth: int | None = None
    for line in text.splitlines():
        indent = len(line) - len(line.lstrip())
        if line.strip() and depth is not None and indent <= depth:
            depth = None
        if depth is None and line.strip() == BACKGROUND:
            depth = indent
        elif depth is None and "$$" in line:
            loose.append(line.strip())
    return loose


@pytest.mark.parametrize("page", sorted(p.name for p in GUIDE.glob("*.md")))
def test_a_guide_page_states_the_model_math_only_through_the_spec(page):
    text = (GUIDE / page).read_text()
    assert not _loose_math(text)
    for kind, args in MARKER.findall(text):
        assert marker(kind, args, f"user-guide/optimization/{page}")


@pytest.mark.parametrize(
    ("kind", "args"),
    [
        ("spec", " Generator_nonexistent"),
        ("spec-legend", ""),
        ("spec-tabel", " Generator_p"),
    ],
)
def test_a_marker_naming_no_entry_fails(kind, args):
    with pytest.raises(ValueError, match="spec has no|No marker"):
        marker(kind, args, "user-guide/optimization/index.md")


@pytest.mark.parametrize(
    "text",
    [
        "$$x$$",
        '!!! note "Background"\n\n    $$x$$\n\n$$y$$',
        '=== "A"\n\n    !!! note "Other"\n\n        $$x$$',
    ],
)
def test_display_math_outside_a_background_note_is_loose(text):
    assert _loose_math(text)


@pytest.mark.parametrize(
    "text",
    [
        '!!! note "Background"\n\n    $$x$$',
        '=== "A"\n\n    !!! note "Background"\n\n        $$x$$\n\n        text\n\n    after',
    ],
)
def test_display_math_inside_a_background_note_is_not_loose(text):
    assert not _loose_math(text)
