# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest

import pypsa


def test_transmission_losses_independent_of_s_nom_max():
    # This should fail for mode="tangents" as the tangents depend on s_nom_max
    n = pypsa.examples.ac_dc_meshed()
    n.lines["s_nom_max"] = 2000
    n.optimize(transmission_losses={"mode":"secants"})
    res = n.lines[["s_nom_opt"]]
    n.lines["s_nom_max"] = 3000
    n.optimize(transmission_losses={"mode":"secants"})
    res2 = n.lines[["s_nom_opt"]]
    assert res.equals(res2)

def test_secant_losses_larger():
    n = pypsa.examples.ac_dc_meshed()
    n.optimize(transmission_losses={"mode":"secants"})
    assert (n.lines_t.p0**2 * n.lines.r_pu_eff).values <= n.model.solution["Line-loss"]

def test_tangent_losses_smaller():
    n = pypsa.examples.ac_dc_meshed()
    n.optimize(transmission_losses={"mode":"tangents"})
    assert (n.lines_t.p0**2 * n.lines.r_pu_eff).values >= n.model.solution["Line-loss"]
