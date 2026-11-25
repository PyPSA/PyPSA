# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

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
