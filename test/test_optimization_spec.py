# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import re
from pathlib import Path

import mathspec
import pandas as pd
import pytest
import xarray as xr
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
from pypsa.optimization.spec.layers import SPEC, spec
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


def test_every_trait_serves_more_than_one_component():
    traits = (path.stem for path in (SPEC / "traits").iterdir())
    assert all(len(formulated(pypsa.Network(), t)) > 1 for t in traits)


def _generic(c: pypsa.components.Components) -> dict[tuple[str, str], str]:
    data = _load(SPEC / "components" / f"{_snake(c.name)}.yaml", lambda n: n)[0]
    return {
        (section, k.replace(c.name, "COMPONENT")): yaml.safe_dump(
            {f: v for f, v in entry.items() if f != "description"}
            if isinstance(entry, dict)
            else entry
        ).replace(c.name, "COMPONENT")
        for section, body in data.items()
        if section not in {"given", "dimensions", "relations", "macros"}
        for k, entry in body.items()
    }


def test_no_entry_is_stated_alike_in_two_component_files():
    stated: dict[tuple[str, str], list[str]] = {}
    for c in formulated(pypsa.Network()):
        for key, body in _generic(c).items():
            stated.setdefault((*key, body), []).append(c.name)
    alike = {key[:2]: names for key, names in stated.items() if len(names) > 1}
    assert not alike


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


@pytest.mark.parametrize("section", ["variables", "constraints"])
def test_a_key_is_its_linopy_name_with_underscores(section):
    named = render(SPEC)[1][section]
    mismatched = {}
    for key, name in named.items():
        names = [name] if isinstance(name, str) else name.values()
        spelled = {n.replace("-", "_") for n in names}
        capacity = {f"{n}_ext" for n in spelled if section == "variables"}
        if key not in spelled | capacity:
            mismatched[key] = name
    assert not mismatched


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


SHARED = {
    "dimensions": {"bus": {"description": "nodes"}},
    "parameters": {"p": {}},
    "macros": {"m": {"args": ["x"], "template": "2 * x"}},
}


def test_a_merge_drops_given_and_keeps_a_shared_entry_stated_alike():
    spec = {k: dict(v) for k, v in SHARED.items()}
    fragment = {
        "dimensions": {"bus": {}},
        "macros": {"m": {"args": ["x"], "template": "2 * x"}},
        "given": {"parameters": {"q": {}}},
    }
    _merge(spec, fragment, "f")
    assert spec == SHARED


@pytest.mark.parametrize(
    ("fragment", "section"),
    [
        ({"dimensions": {"bus": {"dtype": "int"}}}, "dimensions"),
        ({"parameters": {"p": {}}}, "parameters"),
        ({"macros": {"m": {"args": ["x"], "template": "3 * x"}}}, "macros"),
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


def test_a_subclass_extends_the_source_registry(ac_dc_network):
    class Extended(NetworkSources):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            extra = {"COMPONENT_tag": lambda s, name, c: s.filled(name, 2.0)}
            self._providers = _providers(self.units.values(), _SOURCES | extra)

    n = ac_dc_network
    s = mathspec.override(
        spec(), [{"parameters": {"Generator_tag": {"dims": ["Generator"]}}}]
    )
    assert (Extended(n, s, n.snapshots, n.snapshots)["Generator_tag"] == 2.0).all()
    with pytest.raises(NotImplementedError):
        NetworkSources(n, s, n.snapshots, n.snapshots)["Generator_tag"]


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


def test_ports_of_links_and_processes_share_one_dim():
    n = pypsa.Network(snapshots=range(2))
    n.add("Bus", ["a", "b", "c"])
    n.add("Generator", "g", bus="a", p_nom=100, marginal_cost=1)
    n.add("Load", "l", bus="b", p_set=10)
    curve = {0.0: 0.0, 1.0: 0.5}
    n.add("Link", "link", bus0="a", bus1="b", bus2="c", p_nom=50, efficiency2=curve)
    n.add("Process", "proc", bus0="a", bus1="b", rate0=-1, rate1=curve, p_nom=50)
    m = n.optimize.create_model()
    link, proc = (m.variables[f"{c}-p_piecewise"] for c in ("Link", "Process"))
    assert list(link.indexes["port"]) == list(proc.indexes["port"]) == [0, 1, 2]
    live = {
        c: (v.labels != -1).any(["scenario", "snapshot"]).to_series()
        for c, v in [("Link", link), ("Process", proc)]
    }
    assert live["Link"][live["Link"]].index.tolist() == [("link", 2)]
    assert live["Process"][live["Process"]].index.tolist() == [("proc", 1)]


@pytest.fixture
def ac_dc_two_periods(ac_dc_network):
    n = ac_dc_network
    n.snapshots = pd.MultiIndex.from_product([[2020, 2030], n.snapshots[:5]])
    n.investment_periods = [2020, 2030]
    return n


NETWORKS = ["ac_dc_network", "ac_dc_stochastic", "ac_dc_two_periods"]


def _solved(
    request: pytest.FixtureRequest, network: str, **kwargs: object
) -> pypsa.Network:
    n = request.getfixturevalue(network)
    multi_invest = n.has_investment_periods
    n.optimize(
        multi_investment_periods=multi_invest, include_expressions=True, **kwargs
    )
    return n


def _pandas(da: xr.DataArray, n: pypsa.Network) -> pd.Series | pd.DataFrame:
    if not n.has_scenarios:
        da = da.squeeze("scenario", drop=True)
    units = [d for d in ("scenario", "name") if d in da.dims]
    if "snapshot" not in da.dims:
        return da.transpose(*units).to_series()
    if len(units) > 1:
        da = da.stack(column=units)
    return da.transpose("snapshot", ...).to_pandas().set_axis(n.snapshots)


@pytest.mark.parametrize("network", NETWORKS)
@pytest.mark.parametrize(
    ("name", "read"),
    [
        ("Generator_p_nom_effective", lambda n: n.c.generators.static.p_nom_opt),
        ("Line_s_monitored", lambda n: n.c.lines.dynamic.p0),
    ],
)
def test_an_expression_reads_the_solution(request, network, name, read):
    n = _solved(request, network)
    expected = read(n)
    result = _pandas(n.model.expressions[name].solution, n)
    assert_equal = (
        pd.testing.assert_frame_equal
        if expected.ndim == 2
        else pd.testing.assert_series_equal
    )
    assert_equal(result, expected, check_names=False)


@pytest.mark.parametrize("network", NETWORKS)
@pytest.mark.parametrize("objective_scaling", [1.0, 1e-3])
def test_the_total_cost_is_the_objective(request, network, objective_scaling):
    n = _solved(request, network, objective_scaling=objective_scaling)
    cost = n.model.expressions["total_cost"].solution.item()
    assert cost == pytest.approx(n.objective + n.objective_constant)


@pytest.mark.parametrize("network", NETWORKS)
@pytest.mark.parametrize(
    ("name", "variable"),
    [("Line_s_monitored", "Line-s"), ("Generator_running_cost", "Generator-p")],
)
def test_an_expression_takes_the_dims_of_its_variable(request, network, name, variable):
    n = request.getfixturevalue(network)
    m = n.optimize.create_model(
        multi_investment_periods=n.has_investment_periods, include_expressions=True
    )
    expr, var = m.expressions[name], m.variables[variable]
    assert expr.coord_dims == var.dims
    xr.testing.assert_equal(expr.coords.to_dataset(), var.coords.to_dataset())


def test_an_override_adds_an_expression_with_a_variable_term(request):
    dims = ["scenario", "snapshot", "Generator"]
    added = {
        "Generator_twice_p": {"dims": dims, "expression": "2 * Generator_p"},
        "Generator_twice_p_max_pu": {
            "dims": dims,
            "expression": "2 * Generator_p_max_pu",
        },
        "Generator_price": {
            "dims": dims,
            "expression": "at(dual(Bus_nodal_balance), by=Generator_bus[bus])",
        },
    }
    n = _solved(request, "ac_dc_network", overrides=[{"expressions": added}])
    built = n.model.expressions
    assert {"Generator_twice_p_max_pu", "Generator_price"}.isdisjoint(built)
    result = _pandas(built["Generator_twice_p"].solution, n)
    pd.testing.assert_frame_equal(
        result, 2 * n.c.generators.dynamic.p, check_names=False
    )


def test_an_expression_over_two_components_keeps_their_dims(ac_dc_network):
    body = "Generator_p_nom_effective + Link_p_nom_effective"
    dims = ["scenario", "Generator", "Link"]
    added = {"Generator_Link_p_nom": {"dims": dims, "expression": body}}
    m = ac_dc_network.optimize.create_model(
        include_expressions=True, overrides=[{"expressions": added}]
    )
    expr = m.expressions["Generator_Link_p_nom"]
    assert expr.coord_dims == ("Generator", "Link", "scenario")


def test_an_expression_over_period_and_snapshot_keeps_its_periods(ac_dc_two_periods):
    dims = ["scenario", "snapshot", "period", "Generator"]
    body = "Generator_p * period_weighting_objective"
    added = {"Generator_p_by_period": {"dims": dims, "expression": body}}
    m = ac_dc_two_periods.optimize.create_model(
        multi_investment_periods=True,
        include_expressions=True,
        overrides=[{"expressions": added}],
    )
    period = m.expressions["Generator_p_by_period"].coords["period"]
    assert period.dims == ("period",)
    assert period.values.tolist() == [2020, 2030]


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
