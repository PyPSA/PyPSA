import numpy as np
from numpy.testing import assert_almost_equal as equal
from numpy.testing import assert_array_almost_equal as arr_equal


def test_optimize_security_constrained(scipy_network):
    n = scipy_network

    # There are some infeasibilities without line extensions
    for line_name in ["316", "527", "602"]:
        n.lines.loc[line_name, "s_nom"] = 1200

    # choose the contingencies
    branch_outages = n.lines.index[:2]

    n.optimize.optimize_security_constrained(
        n.snapshots[0],
        branch_outages=branch_outages,
    )
    # For the PF, set the P to the optimised P
    n.generators_t.p_set = n.generators_t.p.copy()
    n.storage_units_t.p_set = n.storage_units_t.p.copy()

    # Check no lines are overloaded with the linear contingency analysis

    p0_test = n.lpf_contingency(n.snapshots[0], branch_outages=branch_outages)

    # check loading as per unit of s_nom in each contingency

    max_loading = (
        abs(p0_test.divide(n.passive_branches().s_nom, axis=0)).describe().loc["max"]
    )

    arr_equal(max_loading, np.ones(len(max_loading)), decimal=4)

    equal(n.objective, 339758.4578, decimal=1)
