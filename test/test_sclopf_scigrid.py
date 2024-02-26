# -*- coding: utf-8 -*-

import numpy as np
import pytest
from conftest import SOLVER_NAME, SUPPORTED_APIS
from numpy.testing import assert_almost_equal as equal
from numpy.testing import assert_array_almost_equal as arr_equal


def sclopf(n, api, *args, **kwargs):
    if api == "linopy":
        return n.optimize.optimize_security_constrained(*args, **kwargs)
    elif api == "pyomo":
        return n.sclopf(pyomo=True, *args, **kwargs)
    elif api == "native":
        return n.sclopf(pyomo=False, *args, **kwargs)
    else:
        raise ValueError(f"api must be one of {SUPPORTED_APIS}")


@pytest.mark.parametrize("api", SUPPORTED_APIS)
def test_sclopf(scipy_network, api):
    n = scipy_network

    # There are some infeasibilities without line extensions
    for line_name in ["316", "527", "602"]:
        n.lines.loc[line_name, "s_nom"] = 1200

    # choose the contingencies
    branch_outages = n.lines.index[:2]

    sclopf(
        n,
        api,
        n.snapshots[0],
        branch_outages=branch_outages,
        solver_name=SOLVER_NAME,
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

    arr_equal(max_loading, np.ones((len(max_loading))), decimal=4)

    equal(n.objective, 339758.4578, decimal=1)
