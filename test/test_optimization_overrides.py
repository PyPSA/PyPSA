# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest
import yaml
from mathspec.errors import LanguageError

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


@pytest.mark.parametrize("blank", [float("nan"), ""])
def test_an_added_relation_reads_a_static_column(ac_dc_network, blank):
    n = ac_dc_network
    links = n.c.links.static
    links["cable_end0"] = links["cable_end1"] = pd.Series(blank, links.index, object)
    links.loc["Norwich Converter", "cable_end0"] = "north sea"
    links.loc["Bremen Converter", "cable_end1"] = "north sea"
    links.loc["Bremen Converter", ["p_nom", "p_nom_extendable"]] = [400.0, False]
    n.optimize(overrides=[CABLE])
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
            "maps Link to one dimension",
        ),
        (
            {
                "dimensions": {"cable": {}},
                "parameters": {"cable_length": {"dims": ["cable"]}},
            },
            "no data prep for cable",
        ),
    ],
)
def test_added_data_without_a_source_fails(ac_dc_network, override, match):
    with pytest.raises(NotImplementedError, match=match):
        ac_dc_network.optimize.create_model(overrides=[override])
