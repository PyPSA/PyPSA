# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import copy
import datetime
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
import yaml
from mathspec.errors import LanguageError

import pypsa
from pypsa.optimization.spec.build import spec

HALF = {
    "dims": ["Generator", "scenario", "snapshot"],
    "where": "Generator_p_nom_extendable AND not Generator_committable AND Generator_active",
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


def _as(form, tmp_path):
    if form == "mapping":
        return OVERRIDE
    if form == "text":
        return yaml.safe_dump(OVERRIDE)
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(OVERRIDE))
    return str(path) if form == "str" else path


@pytest.mark.parametrize("form", ["mapping", "text", "str", "path"])
def test_an_override_removes_and_adds_constraints(ac_dc_network, tmp_path, form):
    m = ac_dc_network.optimize.create_model(overrides=[_as(form, tmp_path)])
    assert "Generator-ext-p-upper" not in m.constraints
    assert "Generator_p_half" in m.constraints


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
            "has no column cable_end1",
        ),
        (
            {
                "relations": {
                    "Link_ends": {"key": "Link", "values": {"a": "bus", "b": "bus"}}
                }
            },
            r"declare it \{key: Link, values: <dimension>\}",
        ),
        (
            {
                "dimensions": {"cable": {}},
                "parameters": {"cable_length": {"dims": ["cable"]}},
            },
            "no data prep for cable: no relation maps units into it",
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
    assert n.optimize.overrides.no_ramps == n.optimize.overrides["no_ramps"] == NO_RAMPS
    with caplog.at_level("INFO", logger="pypsa.optimization.spec.build"):
        m = n.optimize.create_model()
    assert "['override', 'no_ramps']" in caplog.text
    assert "Generator_p_half" in m.constraints
    m = n.optimize.create_model(overrides=[{"constraints": {"Generator_p_half": None}}])
    assert "Generator_p_half" not in m.constraints
    n.optimize.overrides.remove("override")
    assert list(n.optimize.overrides) == ["no_ramps"]
    assert "Generator_p_half" not in n.optimize.create_model().constraints


@pytest.mark.parametrize(
    "form",
    [
        "copy",
        "slice",
        "netcdf",
        "csv",
        pytest.param(
            "hdf5",
            marks=pytest.mark.skipif(
                find_spec("tables") is None, reason="PyTables not installed"
            ),
        ),
        pytest.param(
            "excel",
            marks=pytest.mark.skipif(
                find_spec("openpyxl") is None, reason="openpyxl not installed"
            ),
        ),
    ],
)
def test_stored_overrides_survive_copy_and_io(ac_dc_network, tmp_path, form):
    n = ac_dc_network
    n.optimize.overrides.register(OVERRIDE, name="half")
    n.optimize.overrides.register(FULL_LOAD_HOURS, name="full_load_hours")
    path = tmp_path / {"netcdf": "n.nc", "hdf5": "n.h5", "excel": "n.xlsx"}.get(
        form, "n"
    )
    if form in ("netcdf", "hdf5", "csv", "excel"):
        export = "csv_folder" if form == "csv" else form
        getattr(n, f"export_to_{export}")(path)
    other = {"copy": n.copy, "slice": lambda: n.slice_network(n.c.buses.names)}.get(
        form, lambda: pypsa.Network(path)
    )()
    assert other.optimize.overrides == n.optimize.overrides
    n.c.generators.static["full_load_hours_min"] = 0.0
    other.c.generators.static["full_load_hours_min"] = 0.0
    n.optimize()
    other.optimize()
    assert other.objective == pytest.approx(n.objective)


@pytest.mark.parametrize(
    ("override", "name", "error", "match"),
    [
        (NO_RAMPS, None, ValueError, "takes a `name`"),
        (NO_RAMPS, "no-ramps", ValueError, "identifier"),
        (NO_RAMPS, "register", ValueError, "identifier"),
        ("# nothing\n", "empty", TypeError, "states a mapping"),
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
    assert "_pypsa" not in n.meta


def test_a_name_holds_one_override(ac_dc_network):
    overrides = ac_dc_network.optimize.overrides
    patch = copy.deepcopy(NO_RAMPS)
    overrides.register(patch, name="no_ramps")
    patch["constraints"].clear()
    overrides.no_ramps["constraints"].clear()
    overrides.register(NO_RAMPS, name="no_ramps")
    with pytest.raises(ValueError, match="Another override 'no_ramps'"):
        overrides.register(OVERRIDE, name="no_ramps")
    with pytest.raises(ValueError, match="stored as 'no_ramps'"):
        overrides.register(NO_RAMPS, name="again")
    with pytest.raises(KeyError, match="No override 'again'"):
        overrides.remove("again")
    assert list(overrides) == ["no_ramps"]


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
                "edits": ["constraints.Generator_old"],
                "creates": [],
            },
            r"'old' edits \['constraints.Generator_old'\], which the spec below it no longer declares",
        ),
        (
            {
                "patch": {"constraints": {"Generator_ext_p_upper": HALF}},
                "edits": [],
                "creates": ["constraints.Generator_ext_p_upper"],
            },
            r"'old' creates \['constraints.Generator_ext_p_upper'\], which the spec below it now declares",
        ),
    ],
)
def test_a_stored_override_that_drifted_from_the_spec_fails(
    ac_dc_network, stored, match
):
    n = ac_dc_network
    n.meta["_pypsa"] = {"overrides": {"old": stored, "older": stored}}
    with pytest.raises(ValueError, match=match):
        n.optimize.create_model()
    n.optimize.overrides.remove("old")
    n.optimize.overrides.remove("older")
    n.optimize.create_model()


def test_stored_overrides_are_mappings(ac_dc_network):
    ac_dc_network.meta["_pypsa"] = {"overrides": {"old": "garbage"}}
    with pytest.raises(ValueError, match="maps each name to a mapping"):
        ac_dc_network.optimize.create_model()


def test_a_stored_override_drifting_under_build_options_says_so(ac_dc_network):
    n = ac_dc_network
    n.optimize.overrides.register(
        {"constraints": {"Generator_com_p_before": HALF}}, name="before"
    )
    n.optimize.create_model()
    with pytest.raises(ValueError, match="pass it to this build as a per-call"):
        n.optimize.create_model(linearized_unit_commitment=True)


def test_meta_keys_of_the_user_stay_theirs(ac_dc_network):
    n = ac_dc_network
    n.meta |= {"overrides": {"solver": {"threads": 4}}, "composites": "mine"}
    n.optimize.overrides.register(NO_RAMPS, name="no_ramps")
    n.optimize.create_model()
    assert n.equals(n.copy())
    assert n.meta["overrides"] == {"solver": {"threads": 4}}


def test_spatial_clustering_keeps_stored_overrides(ac_dc_network):
    n = ac_dc_network
    n.optimize.overrides.register(NO_RAMPS, name="no_ramps")
    busmap = n.c.buses.static.index.to_series()
    clustered = n.cluster.spatial.cluster_by_busmap(busmap)
    assert clustered.optimize.overrides == n.optimize.overrides


def test_an_override_others_build_on_is_kept(ac_dc_network):
    n = ac_dc_network
    parameter, constraint = ({k: v} for k, v in yaml.safe_load(FULL_LOAD_HOURS).items())
    n.optimize.overrides.register(parameter, name="parameter")
    n.optimize.overrides.register(constraint, name="constraint")
    with pytest.raises(ValueError, match=r"\['constraint'\] build on 'parameter'"):
        n.optimize.overrides.remove("parameter")
    assert list(n.optimize.overrides) == ["parameter", "constraint"]
    n.optimize.overrides.remove("constraint")
    n.optimize.overrides.remove("parameter")
    assert "_pypsa" not in n.meta


def test_merge_unites_stored_overrides(ac_dc_network):
    n = ac_dc_network
    other = pypsa.Network()
    other.set_snapshots(n.snapshots)
    n.optimize.overrides.register(NO_RAMPS, name="no_ramps")
    other.optimize.overrides.register(OVERRIDE, name="half")
    assert list(n.merge(other).optimize.overrides) == ["no_ramps", "half"]
    other.optimize.overrides.remove("half")
    stored = {"patch": OVERRIDE, "edits": [], "creates": []}
    other.meta["_pypsa"] = {"overrides": {"no_ramps": stored}}
    with pytest.raises(
        ValueError, match=r"different override under the names \['no_ramps'\]"
    ):
        n.merge(other)


def test_merge_refuses_overrides_that_do_not_build_together(ac_dc_network):
    n = ac_dc_network
    other = pypsa.Network()
    other.set_snapshots(n.snapshots)
    n.optimize.overrides.register(OVERRIDE, name="half")
    other.optimize.overrides.register(OVERRIDE, name="half_too")
    with pytest.raises(ValueError, match="do not build together"):
        n.merge(other, inplace=True)
    assert list(n.optimize.overrides) == ["half"]


def test_equals_follows_the_order_of_stored_overrides(ac_dc_network):
    n = ac_dc_network
    n.optimize.overrides.register(NO_RAMPS, name="no_ramps")
    n.optimize.overrides.register(OVERRIDE, name="half")
    other = n.copy()
    assert n.equals(other)
    stored = other.meta["_pypsa"]["overrides"]
    other.meta["_pypsa"]["overrides"] = dict(reversed(stored.items()))
    assert not n.equals(other)


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


@pytest.mark.parametrize(
    "solve",
    [
        lambda n: n.optimize.optimize_security_constrained(
            branch_outages=n.c.lines.static.index[:2], overrides=[OVERRIDE]
        ),
        lambda n: n.optimize.optimize_mga(overrides=[OVERRIDE]),
        lambda n: n.optimize.optimize_mga_in_direction(
            {"wind": 1},
            {"wind": {"Generator": {"p_nom": {"Manchester Wind": 1}}}},
            overrides=[OVERRIDE],
        ),
    ],
)
def test_other_solves_take_overrides(ac_dc_network, solve):
    n = ac_dc_network
    n.optimize()
    solve(n)
    assert "Generator_p_half" in n.model.constraints
