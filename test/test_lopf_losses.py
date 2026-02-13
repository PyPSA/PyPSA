# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest


@pytest.mark.parametrize("transmission_losses", [1, 2])
def test_optimize_losses(scipy_network, transmission_losses):
    n = scipy_network
    n.c.lines.static.s_max_pu = 0.7
    n.c.lines.static.loc[["316", "527", "602"], "s_nom"] = 1715

    with pytest.warns(FutureWarning, match="transmission_losses"):
        n.optimize(
            snapshots=n.snapshots[0],
            transmission_losses=transmission_losses,
        )

    gen = (
        n.c.generators.dynamic.p.iloc[0].sum()
        + n.c.storage_units.dynamic.p.iloc[0].sum()
    )
    dem = n.c.loads.dynamic.p_set.iloc[0].sum()

    assert gen > 1.01 * dem, "For this example, losses should be greater than 1%"
    assert gen < 1.05 * dem, "For this example, losses should be lower than 5%"
