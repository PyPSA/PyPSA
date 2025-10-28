# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

from numpy.testing import assert_almost_equal as equal


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

    # TODO see https://github.com/PyPSA/PyPSA/issues/1356

    # # Check no lines are overloaded with the linear contingency analysis
    # p0_test = n.lpf_contingency(n.snapshots[0], branch_outages=branch_outages)

    # # Check loading as per unit of s_nom in each contingency
    # max_loading = (
    #     abs(p0_test.divide(n.passive_branches().s_nom, axis=0)).describe().loc["max"]
    # )

    # arr_equal(max_loading, np.ones(len(max_loading)), decimal=4)
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
