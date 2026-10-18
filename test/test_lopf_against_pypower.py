# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Compare PyPSA's linear optimal power flow with PYPOWER's DC optimal power flow.

PYPOWER's ``rundcopf`` is also the engine behind pandapower's ``rundcopp``, so
this covers the request in GH #478 to test the OPF against pandapower without
depending on pandapower (which does not support pandas 3 yet).
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pypower.case30 import case30
from pypower.idx_brch import PF, RATE_A
from pypower.idx_bus import LAM_P
from pypower.idx_cost import COST, MODEL, NCOST, POLYNOMIAL
from pypower.idx_gen import PG, PMAX, PMIN
from pypower.ppoption import ppoption

import pypsa
from pypsa.constants import DEFAULT_TIMESTAMP

with warnings.catch_warnings():
    # PYPOWER 5.1.21 ships invalid escape sequences in qps_cplex.py and
    # qps_mosek.py, which Python 3.11 reports as a DeprecationWarning when the
    # modules are first compiled (an error under the test configuration).
    warnings.simplefilter("ignore", DeprecationWarning)
    warnings.simplefilter("ignore", SyntaxWarning)
    from pypower.rundcopf import rundcopf


@pytest.mark.parametrize(
    "quadratic", [False, True], ids=["linear-cost", "quadratic-cost"]
)
@pytest.mark.parametrize("rating_scale", [1.0, 0.75], ids=["rating-1", "rating-0.75"])
def test_lopf_against_pypower_dcopf(quadratic, rating_scale):
    ppc = case30()
    gencost = ppc["gencost"]
    assert (gencost[:, MODEL] == POLYNOMIAL).all()
    assert (gencost[:, NCOST] == 3).all()
    if not quadratic:
        # keep only c1 * p + c0
        gencost[:, COST] = 0.0
    # tighten the line ratings so that lines are congested and nodal prices differ
    ppc["branch"][:, RATE_A] *= rating_scale

    results = rundcopf(ppc, ppoption(VERBOSE=0, OUT_ALL=0))
    assert results["success"]

    n = pypsa.Network()
    n.import_from_pypower_ppc(ppc)
    # the importer fills the deprecated (and ignored) v_ang_min from ANGMIN
    n.c.lines.static["v_ang_min"] = -np.inf
    generators = n.c.generators.static
    # the importer carries the PYPOWER set-points, which optimize() would fix the
    # dispatch to, and it does not carry gencost (cost = c2 p^2 + c1 p + c0)
    generators["p_set"] = np.nan
    c2, c1, c0 = gencost[:, COST], gencost[:, COST + 1], gencost[:, COST + 2]
    generators["marginal_cost"] = c1
    generators["marginal_cost_quadratic"] = c2
    generators["p_min_pu"] = ppc["gen"][:, PMIN] / ppc["gen"][:, PMAX]

    status, condition = n.optimize(solver_name="highs")
    assert status == "ok"
    assert condition == "optimal"

    # the quadratic problem is solved by an interior point method
    atol = 1e-3 if quadratic else 1e-5

    # generator dispatch
    assert_allclose(
        n.c.generators.dynamic.p.loc[DEFAULT_TIMESTAMP].values,
        results["gen"][:, PG],
        atol=atol,
    )

    # objective; PYPOWER's includes the constant cost terms
    assert_allclose(n.objective + c0.sum(), results["f"], rtol=1e-6)

    # branch flows
    for item in ["lines", "transformers"]:
        static = getattr(n.c, item).static
        dynamic = getattr(n.c, item).dynamic
        branch_i = static.original_index.values.astype(int)
        assert_allclose(
            dynamic.p0.loc[DEFAULT_TIMESTAMP].values,
            results["branch"][branch_i, PF],
            atol=atol,
        )

    # nodal prices
    assert_allclose(
        n.c.buses.dynamic.marginal_price.loc[DEFAULT_TIMESTAMP].values,
        results["bus"][:, LAM_P],
        atol=atol,
    )

    # the comparison is only meaningful if the network limits bind somewhere
    loading = np.abs(results["branch"][:, PF]) / ppc["branch"][:, RATE_A]
    if rating_scale < 1 or not quadratic:
        assert (loading > 0.999).any()
        lam = results["bus"][:, LAM_P]
        assert lam.max() - lam.min() > 0.1
