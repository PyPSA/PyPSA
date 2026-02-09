# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT


import numpy as np

import pypsa


def test_transmission_losses_independent_of_s_nom_max():
    # This should fail for mode="tangents" as the tangents depend on s_nom_max
    n = pypsa.examples.ac_dc_meshed()
    n.lines["s_nom_max"] = 2000
    n.optimize(transmission_losses={"mode": "secants"})
    res = n.lines[["s_nom_opt"]]
    n.lines["s_nom_max"] = 3000
    n.optimize(transmission_losses={"mode": "secants"})
    res2 = n.lines[["s_nom_opt"]]
    assert res.equals(res2)


def test_secant_losses_larger():
    n = pypsa.examples.ac_dc_meshed()
    n.lines["s_nom_max"] = 2000
    n.optimize(transmission_losses={"mode": "secants"})
    true_losses = (n.lines_t.p0**2 * n.lines.r_pu_eff).values
    model_losses = n.model.solution["Line-loss"].sel(name=n.lines.index).values

    # Assert with tolerance
    np.testing.assert_array_less(true_losses - 1e-2, model_losses)


def test_tangent_losses_smaller():
    n = pypsa.examples.ac_dc_meshed()
    n.lines["s_nom_max"] = 2000
    n.optimize(transmission_losses={"mode": "tangents", "segments": 3})
    true_losses = (n.lines_t.p0**2 * n.lines.r_pu_eff).values
    model_losses = n.model.solution["Line-loss"].sel(name=n.lines.index).values

    # Assert with tolerance
    np.testing.assert_array_less(model_losses - 1e-2, true_losses)
