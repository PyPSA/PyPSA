# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal as equal

import pypsa
from pypsa import option_context


def test_optimize_security_constrained(scipy_network):
    """Test security-constrained optimization functionality and dual variable assignment."""
    n = scipy_network

    # There are some infeasibilities without line extensions
    for line_name in ["316", "527", "602"]:
        n.c.lines.static.loc[line_name, "s_nom"] = 1200

    # Choose the contingencies
    branch_outages = n.c.lines.static.index[:2]

    # # Run security-constrained optimization with dual assignment
    # # Fight numerical instability using https://ergo-code.github.io/HiGHS/
    solver_options = {
        "primal_feasibility_tolerance": 1e-9,
        "dual_feasibility_tolerance": 1e-9,
        "time_limit": 300,
        "presolve": "on",
        "parallel": "off",
        "random_seed": 123,
    }

    n.optimize.optimize_security_constrained(
        n.snapshots[0],
        branch_outages=branch_outages,
        assign_all_duals=True,
        solver_options=solver_options,
    )

    # For the PF, set the P to the optimised P
    n.c.generators.dynamic.p_set = n.c.generators.dynamic.p.copy()
    n.c.storage_units.dynamic.p_set = n.c.storage_units.dynamic.p.copy()

    # Check no lines are overloaded with the linear contingency analysis
    p0_test = n.lpf_contingency(n.snapshots[0], branch_outages=branch_outages)

    # Check loading as per unit of s_nom in each contingency
    max_loading = (
        abs(p0_test.divide(n.passive_branches().s_nom, axis=0)).describe().loc["max"]
    )

    equal(max_loading, np.ones(len(max_loading)), decimal=4)
    equal(n.objective, 339758.4578, decimal=1)

    # === Dual variable assignment checks ===

    # Verify that marginal prices are assigned (nodal balance duals)
    assert hasattr(n.c.buses.dynamic, "marginal_price"), (
        "Marginal prices should be assigned"
    )
    assert not n.c.buses.dynamic.marginal_price.empty, (
        "Marginal prices should not be empty"
    )

    # Check that line constraint duals are assigned to n.c.lines.dynamic.mu_*
    line_dual_attrs = [
        attr for attr in n.c.lines.dynamic.keys() if attr.startswith("mu_")
    ]

    # Verify that standard line duals are assigned
    assert "mu_lower" in line_dual_attrs, "mu_lower should be assigned"
    assert "mu_upper" in line_dual_attrs, "mu_upper should be assigned"

    # Check for security constraint duals
    # TODO add again when dual is written to custom constraint
    # security_duals = [attr for attr in line_dual_attrs if "SubNetwork" in str(attr)]
    # assert len(security_duals) == 2, (
    #     "Should have two security constraint duals assigned"
    # )

    # Verify security constraint duals exist and can be converted
    security_constraints = [
        name for name in n.model.dual.data_vars if "security" in name
    ]
    assert len(security_constraints) == 4, "Should have four security constraint duals"


def test_optimize_security_constrained_multiindex_branch_outages():
    """See https://github.com/PyPSA/PyPSA/issues/1631."""
    n = pypsa.Network()
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")
    n.add("Line", "line1", bus0="bus1", bus1="bus2")
    # Add a component with an assigned cost, as required to create the objective function:
    n.add("Generator", "gen1", bus="bus1", marginal_cost=10)
    branch_outages = pd.MultiIndex.from_tuples([("Line", "line1")])

    status, _ = n.optimize.optimize_security_constrained(branch_outages=branch_outages)

    assert status == "ok"


def _multi_period_network() -> pypsa.Network:
    """See https://github.com/PyPSA/PyPSA/issues/1971."""
    n = pypsa.Network()
    n.set_snapshots(pd.MultiIndex.from_product([[2020, 2030], [0]]))
    n.investment_periods = [2020, 2030]
    n.add("Bus", ["a", "b", "c"])
    n.add("Line", ["ab", "bc"], bus0=["a", "b"], bus1=["b", "c"], x=1, s_nom=100)
    n.add(
        "Line",
        "ab_old",
        bus0="a",
        bus1="b",
        x=1,
        s_nom=100,
        build_year=2000,
        lifetime=25,
    )
    n.add("Line", "ca", bus0="c", bus1="a", x=1, s_nom=100, build_year=2030)
    n.add("Line", "future", bus0="b", bus1="c", x=1, s_nom=100, build_year=2040)
    n.add("Generator", "g", bus="a", p_nom=100, marginal_cost=1)
    n.add("Load", "l", bus="c", p_set=10)
    return n


def _security_shapes(n: pypsa.Network) -> dict[str, dict[str, int]]:
    return {
        name: dict(con.labels.sizes)
        for name, con in n.model.constraints.items()
        if "security" in name
    }


@pytest.mark.parametrize("snapshot_index", ["auto", "flat"])
@pytest.mark.parametrize("scenarios", [False, True])
def test_optimize_security_constrained_multi_period_bodf(scenarios, snapshot_index):
    n = _multi_period_network()
    if scenarios:
        n.set_scenarios({"s1": 0.5, "s2": 0.5})

    with option_context("optimization.model_snapshot_index", snapshot_index):
        status, _ = n.optimize.optimize_security_constrained(
            branch_outages=["ab", "ab_old", "ca"], multi_investment_periods=True
        )
    assert status == "ok"

    s = n.model.variables["Line-s"].labels
    if scenarios:
        s = s.sel(scenario="s1")
    coeffs = {}
    for period in n.investment_periods:
        name = f"Line-fix-s-upper-security-for-Line-outage-in-sub-network-0-period-{period}"
        con = n.model.constraints[name]
        if scenarios:
            con = con.sel(scenario="s1")
        position = n.snapshots.get_loc((period, 0))
        for out, aff in product(con.indexes["Line-outage"], con.indexes["name"]):
            if out == aff:
                continue
            lhs = con.lhs.isel(snapshot=0).sel(name=aff, **{"Line-outage": out})
            var = s.isel(snapshot=position).sel(name=out)
            coeff = lhs.coeffs.where(lhs.vars == var).sum()
            coeffs[period, out, aff] = round(float(coeff), 6)

    assert coeffs == {
        (2020, "ab", "bc"): 0,
        (2020, "ab", "ab_old"): 1,
        (2020, "ab_old", "ab"): 1,
        (2020, "ab_old", "bc"): 0,
        (2030, "ab", "bc"): -1,
        (2030, "ab", "ca"): -1,
        (2030, "ca", "ab"): -1,
        (2030, "ca", "bc"): -1,
    }


def test_optimize_security_constrained_multi_period_sub_network_renumbering():
    n = _multi_period_network()
    n.add("Bus", "z")
    n.c.buses.static = n.c.buses.static.loc[["z", "a", "b", "c"]]
    n.add("Line", "za", bus0="z", bus1="a", x=1, s_nom=100, build_year=2030)

    n.optimize.optimize_security_constrained(
        branch_outages=["ab", "ca"], multi_investment_periods=True
    )

    expected = {
        "1-period-2020": {"snapshot": 1, "name": 3, "Line-outage": 1},
        "0-period-2030": {"snapshot": 1, "name": 4, "Line-outage": 2},
    }
    assert _security_shapes(n) == {
        f"Line-fix-s-{bound}-security-for-Line-outage-in-sub-network-{suffix}": shape
        for bound in ("lower", "upper")
        for suffix, shape in expected.items()
    }


def test_optimize_security_constrained_single_period_names():
    n = _multi_period_network()
    n.optimize.optimize_security_constrained(branch_outages=["ab", "ca"])
    shapes = {"snapshot": 2, "name": 5, "Line-outage": 2}
    assert _security_shapes(n) == {
        "Line-fix-s-lower-security-for-Line-outage-in-sub-network-0": shapes,
        "Line-fix-s-upper-security-for-Line-outage-in-sub-network-0": shapes,
    }


@pytest.mark.parametrize(
    ("branch_outages", "match"),
    [
        (["ab", "future"], "not active in any optimized"),
        (pd.MultiIndex.from_tuples([("Line", "ab"), ("Line", "typo")]), "not in the"),
    ],
)
def test_optimize_security_constrained_invalid_outages_raise(branch_outages, match):
    n = _multi_period_network()
    with pytest.raises(ValueError, match=match):
        n.optimize.optimize_security_constrained(
            branch_outages=branch_outages, multi_investment_periods=True
        )
