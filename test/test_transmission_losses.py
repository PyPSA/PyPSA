# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT


import numpy as np
import pytest

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


def test_secant_losses_ge_tangent_losses():
    n = pypsa.examples.ac_dc_meshed()
    n.lines["s_nom_max"] = 2000

    n.optimize(transmission_losses={"mode": "tangents", "segments": 3})
    tangent_losses = n.model.solution["Line-loss"].sel(name=n.lines.index).values.sum()

    n.optimize(transmission_losses={"mode": "secants"})
    secant_losses = n.model.solution["Line-loss"].sel(name=n.lines.index).values.sum()

    assert secant_losses >= tangent_losses - 1e-2


@pytest.mark.parametrize(
    "kwargs",
    [True, {"mode": "secants", "atol": 0.5, "rtol": 0.05}],
)
def test_secant_input_variants(kwargs):
    n = pypsa.examples.ac_dc_meshed()
    n.lines["s_nom_max"] = 2000
    status, _ = n.optimize(transmission_losses=kwargs)
    assert status == "ok"


def test_invalid_params_raise():
    n = pypsa.examples.ac_dc_meshed()
    n.lines["s_nom_max"] = 2000
    with pytest.raises(ValueError, match="segments"):
        n.optimize(transmission_losses={"mode": "tangents"})
    with pytest.raises(ValueError, match="Unknown"):
        n.optimize(transmission_losses={"mode": "foo"})
    with pytest.raises(ValueError, match="atol"):
        n.optimize(transmission_losses={"mode": "secants", "atol": -1})
    with pytest.raises(ValueError, match="rtol"):
        n.optimize(transmission_losses={"mode": "secants", "rtol": -1})
    with pytest.raises((ValueError, RuntimeError), match="max_segments"):
        n.optimize(transmission_losses={"mode": "secants", "max_segments": 0})
    with pytest.raises(ValueError, match="segments"):
        n.optimize(transmission_losses={"mode": "tangents", "segments": -1})
    n.lines["s_nom_max"] = np.inf
    with pytest.raises(ValueError, match="s_nom_max"):
        n.optimize(transmission_losses={"mode": "secants"})
