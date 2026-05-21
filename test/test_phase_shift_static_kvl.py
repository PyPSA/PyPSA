# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for static ``phase_shift`` in cycle-based KVL (issue #1220).

Prior to the fix, ``n.optimize()`` silently dropped the static ``phase_shift``
term from the cycle Kirchhoff Voltage Law constraint, so LOPF results diverged
from a subsequent ``n.lpf()`` / ``n.pf()`` verification run.

Physical setup
--------------

Two parallel branches between buses A and B (both at 380 kV):

* Line L1 with ``x = 1444 ohm`` → per-unit ``x_pu = x / v_nom**2 = 0.01``.
* Transformer T1 with ``x = 1.0``, ``s_nom = 100 MVA`` → ``x_pu = x / s_nom = 0.01``.

PyPSA uses different bases for Line vs Transformer per-unit reactance (see
``pypsa/network/power_flow.py``: ``x_pu_line = x / v_nom**2`` versus
``x_pu_trafo = x / s_nom``). Matching the *numerical* ``x_pu`` is what makes
the parallel branches split flow equally.

Around the single cycle (L1 forward, T1 reversed):

    x_pu * P_L - (x_pu * P_T + phi_rad) = 0

so the flow imbalance is ``P_L - P_T = phi_rad / x_pu``. With 50 MW total
flow A->B, the analytical split is::

    P_L = 25 + 0.5 * phi_rad / x_pu
    P_T = 25 - 0.5 * phi_rad / x_pu
"""

from __future__ import annotations

import math

import pytest

import pypsa


def _build(phase_shift_deg: float = 0.0) -> pypsa.Network:
    """Two-bus parallel Line + Transformer, matching x_pu = 0.01."""
    n = pypsa.Network()
    n.set_snapshots(["t"])
    n.add("Bus", "A", v_nom=380)
    n.add("Bus", "B", v_nom=380)
    n.add(
        "Line",
        "L1",
        bus0="A",
        bus1="B",
        x=0.01 * 380**2,  # ohm  -> x_pu = 0.01
        s_nom=200,
    )
    n.add(
        "Transformer",
        "T1",
        bus0="A",
        bus1="B",
        x=1.0,  # per-unit on s_nom -> x_pu = 1.0 / 100 = 0.01
        s_nom=100,
        phase_shift=phase_shift_deg,
    )
    n.add("Generator", "G", bus="A", p_nom=100, marginal_cost=1.0)
    n.add("Load", "D", bus="B", p_set=50)
    return n


def _flows(n: pypsa.Network) -> tuple[float, float]:
    p_line = float(n.lines_t.p0["L1"].iloc[0])
    p_trafo = float(n.transformers_t.p0["T1"].iloc[0])
    return p_line, p_trafo


def _solve(n: pypsa.Network) -> None:
    status, _ = n.optimize(solver_name="highs")
    assert status == "ok", f"solver returned {status!r}"


@pytest.mark.parametrize("phi_deg", [0.0])
def test_zero_phase_shift_splits_equally(phi_deg: float) -> None:
    n = _build(phase_shift_deg=phi_deg)
    _solve(n)
    p_line, p_trafo = _flows(n)
    assert p_line == pytest.approx(25.0, abs=1e-3)
    assert p_trafo == pytest.approx(25.0, abs=1e-3)


def test_positive_phase_shift_redistributes_flow() -> None:
    phi_deg = 10.0
    n = _build(phase_shift_deg=phi_deg)
    _solve(n)
    p_line, p_trafo = _flows(n)
    delta = math.radians(phi_deg) / 0.01
    assert p_line == pytest.approx(25.0 + 0.5 * delta, abs=1e-2)
    assert p_trafo == pytest.approx(25.0 - 0.5 * delta, abs=1e-2)


def test_negative_phase_shift_reverses_sign() -> None:
    n_pos = _build(phase_shift_deg=+10.0)
    n_neg = _build(phase_shift_deg=-10.0)
    _solve(n_pos)
    _solve(n_neg)
    p_line_pos, p_trafo_pos = _flows(n_pos)
    p_line_neg, p_trafo_neg = _flows(n_neg)
    assert (p_line_pos - 25.0) == pytest.approx(-(p_line_neg - 25.0), abs=1e-2)
    assert (p_trafo_pos - 25.0) == pytest.approx(-(p_trafo_neg - 25.0), abs=1e-2)


def test_lopf_matches_nonlinear_pf() -> None:
    """LOPF flow must match a subsequent n.pf() with the same phase shift."""
    phi_deg = 5.0
    n = _build(phase_shift_deg=phi_deg)
    _solve(n)
    p_line_lopf, p_trafo_lopf = _flows(n)

    n.lpf()
    p_line_lpf = float(n.lines_t.p0["L1"].iloc[0])
    p_trafo_lpf = float(n.transformers_t.p0["T1"].iloc[0])

    assert p_line_lopf == pytest.approx(p_line_lpf, abs=1e-2)
    assert p_trafo_lopf == pytest.approx(p_trafo_lpf, abs=1e-2)
