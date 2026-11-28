# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import copy
import datetime
import inspect
import itertools

import numpy as np
import pandas as pd
import pytest
import yaml
from mathspec.errors import LanguageError

import pypsa
from pypsa.optimization.spec import build, spec

HALF = {
    "dims": ["Generator", "scenario", "snapshot"],
    "where": "Generator_p_nom_extendable AND not Generator_committable AND Generator_activity_mask",
    "expression": "Generator_p <= 0.5 * Generator_p_max_pu * Generator_p_nom_ext",
}
OVERRIDE = {"constraints": {"Generator_ext_p_upper": None, "Generator_p_half": HALF}}
NO_RAMPS = {
    "constraints": {
        "Generator_p_ramp_limit_up": None,
        "Generator_p_ramp_limit_down": None,
    }
}
FULL_LOAD_HOURS = """
parameters:
  Generator_full_load_hours_min:
    dims: [Generator]
constraints:
  Generator_full_load_hours:
    dims: [scenario, Generator]
    where: Generator_full_load_hours_min > 0 AND Generator_p_nom_extendable
    expression: >-
      sum(Generator_p * snapshot_weighting_generators, over=snapshot)
      >= Generator_full_load_hours_min * Generator_p_nom_ext
"""
CABLE = """
dimensions:
  cable: {description: "cables, each with a converter at both ends"}
relations:
  Link_cable_end0: {key: Link, values: cable, missing: absent}
  Link_cable_end1: {key: Link, values: cable, missing: absent}
constraints:
  Link_cable_rating:
    dims: [scenario, cable]
    expression: >-
      sum(Link_p_nom_effective, over=Link, by=Link_cable_end0[cable])
      == sum(Link_p_nom_effective, over=Link, by=Link_cable_end1[cable])
"""

GLOBAL_CONSTRAINT_TYPE = """
relations:
  Generator_{kind}: {{key: Generator, values: global_constraint, missing: absent}}
expressions:
  GlobalConstraint_total:
    cases:
      {kind}:
        when: GlobalConstraint_type == '{kind}'
        expression: sum(Generator_p_nom_ext, over=Generator, by=Generator_{kind}[global_constraint])
"""


def _as(form, tmp_path):
    if form == "mapping":
        return OVERRIDE
    if form == "text":
        return yaml.safe_dump(OVERRIDE)
    if form == "line":
        return yaml.safe_dump(OVERRIDE, default_flow_style=True, width=1000).strip()
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(OVERRIDE))
    return str(path) if form == "str" else path


@pytest.mark.parametrize("form", ["mapping", "text", "line", "str", "path"])
def test_an_override_removes_and_adds_constraints(ac_dc_network, tmp_path, form):
    m = ac_dc_network.optimize.create_model(overrides=[_as(form, tmp_path)])
    assert "Generator-ext-p-upper" not in m.constraints
    assert "Generator_p_half" in m.constraints


@pytest.mark.parametrize("override", ["noramp.yaml", "{constraints: [", "- a\n- b\n"])
def test_an_override_that_is_no_file_nor_yaml_text_fails(ac_dc_network, override):
    with pytest.raises(FileNotFoundError, match="not YAML text of an override"):
        ac_dc_network.optimize.create_model(overrides=[override])


def test_an_override_given_as_one_line_is_named_by_its_place(ac_dc_network):
    with pytest.raises(LanguageError, match="override #1 of `overrides`"):
        ac_dc_network.optimize.create_model(
            overrides=["{constraints: {Generator_nonexistent: null}}"]
        )


def test_an_override_removes_ramp_limits(ac_dc_network):
    n = ac_dc_network
    n.c.generators.static[["ramp_limit_up", "ramp_limit_down"]] = 0.1
    ramps = {f"Generator-p-ramp_limit_{d}" for d in ("up", "down")}
    assert ramps <= set(n.optimize.create_model().constraints)
    m = n.optimize.create_model(overrides=[NO_RAMPS])
    assert not ramps & set(m.constraints)


def test_an_override_changes_the_optimum(ac_dc_network):
    n = ac_dc_network
    n.optimize()
    base = n.objective
    n.optimize(overrides=[OVERRIDE])
    assert n.objective > base


def test_an_added_parameter_reads_a_static_column(ac_dc_network):
    n = ac_dc_network
    gas = n.c.generators.static.carrier == "gas"
    hours = n.snapshot_weightings.generators
    least = 0.6 * hours.sum()
    n.c.generators.static["full_load_hours_min"] = 0.0
    n.c.generators.static.loc[gas, "full_load_hours_min"] = least
    n.optimize(overrides=[FULL_LOAD_HOURS])
    energy = n.c.generators.dynamic.p.loc[:, gas].mul(hours, axis=0).sum()
    floor = least * n.c.generators.static.p_nom_opt[gas]
    assert (energy >= floor - 1e-6).all()
    assert (energy - floor).abs().min() < 1e-6


def test_overrides_add_types_of_global_constraints(ac_dc_network):
    n = ac_dc_network
    carrier = n.c.generators.static.carrier
    n.add("GlobalConstraint", "gas floor", type="floor", sense=">=", constant=1500.0)
    n.add("GlobalConstraint", "wind floor", type="minimum", sense=">=", constant=8000.0)
    n.add("GlobalConstraint", "no case", type="other", sense="<=", constant=-1.0)
    n.c.generators.static["floor"] = np.where(carrier == "gas", "gas floor", "")
    n.c.generators.static["minimum"] = np.where(carrier == "wind", "wind floor", "")
    kinds = [GLOBAL_CONSTRAINT_TYPE.format(kind=k) for k in ("floor", "minimum")]
    n.optimize(overrides=kinds)
    p_nom = n.c.generators.static.p_nom_opt.groupby(carrier).sum()
    assert p_nom["gas"] == pytest.approx(1500)
    assert p_nom["wind"] == pytest.approx(8000)
    assert (n.c.global_constraints.static.mu[["gas floor", "wind floor"]] > 0).all()
    assert "GlobalConstraint-no case" not in n.model.constraints


def test_an_override_of_an_undeclared_entry_fails(ac_dc_network):
    with pytest.raises(LanguageError, match="Generator_nonexistent"):
        ac_dc_network.optimize.create_model(
            overrides=[{"constraints": {"Generator_nonexistent": None}}]
        )


@pytest.mark.parametrize("override", [42, ["constraints"]])
def test_an_override_of_another_type_fails(ac_dc_network, override):
    with pytest.raises(TypeError, match="YAML path, YAML text or a mapping"):
        ac_dc_network.optimize.create_model(overrides=[override])


def test_an_override_is_kept_by_its_content():
    assert spec(overrides=[OVERRIDE]) is spec(overrides=[yaml.safe_dump(OVERRIDE)])
    assert spec(overrides=[OVERRIDE]) is not spec()


@pytest.mark.parametrize(
    ("blank", "spelling"),
    [
        (float("nan"), "key: Link, values: cable"),
        ("", "key: [Link], values: [cable]"),
        (" ", "key: Link, values: [cable]"),
    ],
)
def test_an_added_relation_reads_a_static_column(ac_dc_network, blank, spelling):
    n = ac_dc_network
    links = n.c.links.static
    links["cable_end0"] = links["cable_end1"] = pd.Series(blank, links.index, object)
    links.loc["Norwich Converter", "cable_end0"] = "north sea"
    links.loc["Bremen Converter", "cable_end1"] = "north sea"
    links.loc["Bremen Converter", ["p_nom", "p_nom_extendable"]] = [400.0, False]
    n.optimize(overrides=[CABLE.replace("key: Link, values: cable", spelling)])
    assert list(n.model.constraints["Link_cable_rating"].indexes["cable"]) == [
        "north sea"
    ]
    assert links.p_nom_opt["Norwich Converter"] == pytest.approx(400)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        (
            {"relations": {"Link_cable_end1": {"key": "Link", "values": "bus"}}},
            "has no column 'cable_end1'",
        ),
        (
            {
                "relations": {
                    "Link_ends": {"key": "Link", "values": {"a": "bus", "b": "bus"}}
                }
            },
            r"declare it as `\{key: Link, values: <dimension>\}`",
        ),
        (
            {
                "dimensions": {"cable": {}},
                "parameters": {"cable_length": {"dims": ["cable"]}},
            },
            "No relation maps components into it",
        ),
    ],
)
def test_added_data_without_a_source_fails(ac_dc_network, override, match):
    with pytest.raises(NotImplementedError, match=match):
        ac_dc_network.optimize.create_model(overrides=[override])


def test_an_added_relation_refuses_labels_that_differ_by_scenario(ac_dc_network):
    n = ac_dc_network
    n.c.links.static["cable_end0"] = n.c.links.static["cable_end1"] = ""
    n.set_scenarios({"low": 0.5, "high": 0.5})
    n.c.links.static.loc[("high", "Norwich Converter"), "cable_end0"] = "north sea"
    with pytest.raises(ValueError, match="differing values.*cable_end0"):
        n.optimize.create_model(overrides=[CABLE])


def test_a_stored_override_applies_to_every_build(ac_dc_network, tmp_path, caplog):
    n = ac_dc_network
    n.optimize.overrides.register(_as("path", tmp_path))
    n.optimize.overrides.register(NO_RAMPS, name="no_ramps")
    assert list(n.optimize.overrides) == ["override", "no_ramps"]
    assert n.optimize.overrides["no_ramps"] == NO_RAMPS
    with caplog.at_level("INFO", logger="pypsa.optimization.spec.build"):
        m = n.optimize.create_model()
    assert "['override', 'no_ramps']" in caplog.text
    assert "Generator_p_half" in m.constraints
    m = n.optimize.create_model(overrides=[{"constraints": {"Generator_p_half": None}}])
    assert "Generator_p_half" not in m.constraints
    n.optimize.overrides.unregister("override")
    assert list(n.optimize.overrides) == ["no_ramps"]
    assert "Generator_p_half" not in n.optimize.create_model().constraints


def test_a_loaded_network_solves_as_its_author_did(ac_dc_network, tmp_path):
    n = ac_dc_network
    n.optimize.overrides.register(OVERRIDE, name="half")
    n.optimize.overrides.register(FULL_LOAD_HOURS, name="full_load_hours")
    n.export_to_netcdf(tmp_path / "n.nc")
    other = pypsa.Network(tmp_path / "n.nc")
    n.c.generators.static["full_load_hours_min"] = 0.0
    other.c.generators.static["full_load_hours_min"] = 0.0
    n.optimize()
    other.optimize()
    assert other.objective == pytest.approx(n.objective)


@pytest.mark.parametrize(
    ("override", "name", "error", "match"),
    [
        (NO_RAMPS, None, ValueError, "Pass `name`"),
        (NO_RAMPS, "no-ramps", ValueError, "identifier"),
        ("# nothing\n", "empty", FileNotFoundError, "not YAML text"),
        ({"constraints": {"Generator_p_half": (1, 2)}}, "x", ValueError, "JSON"),
        (
            {"parameters": {"Generator_x": {"default": np.int64(1)}}},
            "x",
            ValueError,
            "JSON",
        ),
        ({"description": datetime.date(2030, 1, 1)}, "x", ValueError, "JSON"),
        ({"parameters": {2030: {"dims": ["Generator"]}}}, "x", ValueError, "JSON"),
        (
            {"constraints": {"Generator_nonexistent": None}},
            "typo",
            LanguageError,
            "the stored override 'typo' removes the constraint 'Generator_nonexistent'",
        ),
    ],
)
def test_a_stored_override_is_refused(ac_dc_network, override, name, error, match):
    n = ac_dc_network
    with pytest.raises(error, match=match):
        n.optimize.overrides.register(override, name=name)
    assert not n.optimize.overrides


@pytest.mark.parametrize("single", [NO_RAMPS, "no_ramps.yaml"])
def test_a_single_override_takes_a_list(ac_dc_network, single):
    with pytest.raises(TypeError, match="takes a list"):
        ac_dc_network.optimize.create_model(overrides=single)


def test_a_per_call_removal_a_stored_override_repeats_names_it(ac_dc_network):
    n = ac_dc_network
    n.optimize.overrides.register(NO_RAMPS, name="no_ramps")
    with pytest.raises(LanguageError, match="stored override 'no_ramps' removes"):
        n.optimize.create_model(overrides=[NO_RAMPS])


def test_a_file_name_that_is_no_identifier_asks_for_a_name(ac_dc_network, tmp_path):
    path = tmp_path / "2030-limits.yaml"
    path.write_text(yaml.safe_dump(NO_RAMPS))
    with pytest.raises(ValueError, match="Pass `name`"):
        ac_dc_network.optimize.overrides.register(path)
    ac_dc_network.optimize.overrides.register(path, name="limits")


def test_a_patch_is_stored_once_and_apart_from_its_source(ac_dc_network):
    overrides = ac_dc_network.optimize.overrides
    patch = copy.deepcopy(NO_RAMPS)
    overrides.register(patch, name="no_ramps")
    patch["constraints"].clear()
    assert overrides["no_ramps"] == NO_RAMPS
    with pytest.raises(ValueError, match="registered already as 'no_ramps'"):
        overrides.register(NO_RAMPS, name="again")
    overrides.register(NO_RAMPS, name="no_ramps", overwrite=True)
    assert list(overrides) == ["no_ramps"]


def test_overwrite_checks_an_override_against_the_spec_below_its_place(ac_dc_network):
    overrides = ac_dc_network.optimize.overrides
    tighter = HALF | {"expression": HALF["expression"].replace("0.5", "0.4")}
    overrides.register({"constraints": {"Generator_p_half": HALF}}, name="half")
    overrides.register({"constraints": {"Generator_p_half": None}}, name="none")
    overrides.register(
        {"constraints": {"Generator_p_half": tighter}}, name="half", overwrite=True
    )
    assert list(overrides) == ["half", "none"]
    assert "Generator_p_half" not in ac_dc_network.optimize.create_model().constraints
    overrides.unregister("none")
    assert "Generator_p_half" in ac_dc_network.optimize.create_model().constraints


def test_an_override_of_linearised_unit_commitment_stays_per_call(ac_dc_network):
    n = ac_dc_network
    luc = {"constraints": {"Generator_com_partly_start_up": None}}
    with pytest.raises(LanguageError, match="'luc' removes the constraint"):
        n.optimize.overrides.register(luc, name="luc")
    n.optimize.create_model(linearized_unit_commitment=True, overrides=[luc])


@pytest.mark.parametrize(
    ("stored", "match"),
    [
        (
            {
                "patch": {"constraints": {"Generator_old": HALF}},
                "edits": {"constraints.Generator_old": "0"},
                "creates": [],
            },
            r"'old' changes \['constraints.Generator_old'\], which no longer exist",
        ),
        (
            {
                "patch": {"constraints": {"Generator_ext_p_upper": HALF}},
                "edits": {},
                "creates": ["constraints.Generator_ext_p_upper"],
            },
            r"'old' adds \['constraints.Generator_ext_p_upper'\], which exist already",
        ),
        (
            {
                "patch": {"constraints": {"Generator_ext_p_upper": HALF}},
                "edits": {"constraints.Generator_ext_p_upper": "0"},
                "creates": [],
            },
            r"'old' changes \['constraints.Generator_ext_p_upper'\], which PyPSA now defines differently",
        ),
    ],
)
def test_a_stored_override_that_drifted_from_the_spec_fails(
    ac_dc_network, stored, match
):
    n = ac_dc_network
    n.optimize.overrides.register(NO_RAMPS, name="no_ramps")
    n._registries["overrides"] |= {"old": stored, "older": stored}
    with pytest.raises(ValueError, match=match):
        n.optimize.create_model()
    with pytest.raises(ValueError, match="register it again"):
        n.optimize.create_model(linearized_unit_commitment=True)
    for name in ("no_ramps", "old", "older"):
        n.optimize.overrides.unregister(name)
    n.optimize.create_model()


def test_an_entry_linearised_unit_commitment_rewrites_is_no_change(ac_dc_network):
    n = ac_dc_network
    status = {"variables": {"Generator_status": {"description": "on or off"}}}
    n.optimize.overrides.register(status, name="status")
    n.optimize.create_model(linearized_unit_commitment=True)


def test_a_stored_override_without_its_record_fails(ac_dc_network):
    ac_dc_network._registries = {"overrides": {"old": NO_RAMPS}}
    with pytest.raises(ValueError, match="'old' lacks `patch`"):
        ac_dc_network.optimize.create_model()


def test_a_stored_override_drifting_under_build_options_says_so(ac_dc_network):
    n = ac_dc_network
    n.optimize.overrides.register(
        {"constraints": {"Generator_com_p_before": HALF}}, name="before"
    )
    n.optimize.create_model()
    with pytest.raises(ValueError, match="Pass this override to"):
        n.optimize.create_model(linearized_unit_commitment=True)


def test_merge_refuses_overrides_that_do_not_build_together(ac_dc_network):
    n = ac_dc_network
    other = pypsa.Network()
    other.set_snapshots(n.snapshots)
    n.optimize.overrides.register(OVERRIDE, name="half")
    other.optimize.overrides.register(OVERRIDE, name="half_too")
    with pytest.raises(ValueError, match="do not work together"):
        n.merge(other, inplace=True)
    assert list(n.optimize.overrides) == ["half"]


@pytest.mark.parametrize(("missing", "built"), [(0, False), (1000, True)])
def test_an_added_parameter_without_a_column_takes_its_missing_value(
    ac_dc_network, missing, built
):
    override = FULL_LOAD_HOURS.replace(
        "    dims: [Generator]", f"    dims: [Generator]\n    missing: {missing}"
    )
    m = ac_dc_network.optimize.create_model(overrides=[override])
    assert ("Generator_full_load_hours" in m.constraints) is built


def test_an_added_bool_parameter_without_a_column_is_absent(ac_dc_network):
    override = """
parameters:
  Generator_capped: {dims: [Generator], dtype: bool, missing: absent}
constraints:
  Generator_cap:
    dims: [scenario, Generator, snapshot]
    where: Generator_capped
    expression: Generator_p <= 0
"""
    m = ac_dc_network.optimize.create_model(overrides=[override])
    assert "Generator_cap" not in m.constraints


SOLVES = [
    lambda n, **kw: n.optimize.optimize_security_constrained(
        branch_outages=n.c.lines.static.index[:2], **kw
    ),
    lambda n, **kw: n.optimize.optimize_mga(**kw),
    lambda n, **kw: n.optimize.optimize_mga_in_direction(
        {"wind": 1}, {"wind": {"Generator": {"p_nom": {"Manchester Wind": 1}}}}, **kw
    ),
]


@pytest.mark.parametrize("solve", SOLVES)
def test_other_solves_take_overrides(ac_dc_network, solve):
    n = ac_dc_network
    n.optimize()
    solve(n, overrides=[OVERRIDE])
    assert "Generator_p_half" in n.model.constraints


@pytest.mark.parametrize("solve", SOLVES)
def test_other_solves_take_expressions(ac_dc_network, solve):
    n = ac_dc_network
    n.set_snapshots(n.snapshots[:2])
    n.optimize()
    solve(n, include_expressions=True)
    assert "total_cost" in n.model.expressions


@pytest.mark.parametrize(
    "route",
    [
        "__call__",
        "create_model",
        "optimize_security_constrained",
        "optimize_mga",
        "optimize_mga_in_direction",
        "optimize_mga_in_multiple_directions",
    ],
)
@pytest.mark.parametrize("name", ["overrides", "include_expressions"])
def test_overrides_and_include_expressions_are_keyword_only(ac_dc_network, route, name):
    method = getattr(ac_dc_network.optimize, route)
    params = inspect.signature(method).parameters.values()
    before = [p.default for p in itertools.takewhile(lambda p: p.name != name, params)]
    with pytest.raises(TypeError, match="positional argument"):
        method(*before, [OVERRIDE])


PAIR = {
    "name": "pair",
    "components": {
        "Bus": {"dc": {}},
        "Link": {"link": {"bus0": "@dc", "bus1": "@dc"}},
    },
    "math": {
        "expressions": {
            "pair_flow": {
                "dims": ["scenario", "snapshot", "pair"],
                "expression": "at(Link_p, by=pair_members[link])",
            }
        }
    },
}
PIECEWISE = {
    "component": "Generator",
    "attribute": "marginal_cost",
    "sign": ">=",
    "method": "incremental",
}


@pytest.mark.parametrize(
    ("register", "kwargs"),
    [
        (None, {}),
        (None, {"overrides": [OVERRIDE]}),
        (lambda n: n.optimize.overrides.register(NO_RAMPS, name="no_ramps"), {}),
        (lambda n: n.composites.register(PAIR), {}),
        (None, {"linearized_unit_commitment": True}),
        (None, {"piecewise_options": [PIECEWISE]}),
    ],
    ids=["default", "per-call", "stored", "composite", "luc", "piecewise"],
)
def test_spec_is_the_one_the_build_uses(ac_dc_network, monkeypatch, register, kwargs):
    n = ac_dc_network
    if register:
        register(n)
    built = []
    sources = build.NetworkSources

    def spy(n, s, *args, **kw):
        built.append(s)
        return sources(n, s, *args, **kw)

    monkeypatch.setattr(build, "NetworkSources", spy)
    n.optimize.create_model(**kwargs)
    effective = n.optimize.spec(**kwargs).to_dict()
    assert effective == built[0].to_dict()
    assert (effective != spec().to_dict()) is bool(register or kwargs)
