# -*- coding: utf-8 -*-

import pytest

solver_name = "glpk"


@pytest.mark.parametrize("transmission_losses", [1, 2])
@pytest.mark.parametrize("pyomo", [True, False])
def test_lopf_losses(scipy_network, transmission_losses, pyomo):
    n = scipy_network
    n.lines.s_max_pu = 0.7
    n.lines.loc[["316", "527", "602"], "s_nom"] = 1715

    n.lopf(
        snapshots=n.snapshots[0],
        solver_name=solver_name,
        transmission_losses=transmission_losses,
        pyomo=pyomo,
    )

    gen = n.generators_t.p.iloc[0].sum() + n.storage_units_t.p.iloc[0].sum()
    dem = n.loads_t.p_set.iloc[0].sum()

    assert gen > 1.01 * dem, "For this example, losses should be greater than 1%"
