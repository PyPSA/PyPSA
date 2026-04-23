# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for Transformer ``phase_shift`` in LOPF.

Covers two related capabilities:

1. **Bug fix for issue #1220**: static ``phase_shift`` is now included in the
   cycle-based Kirchhoff Voltage Law constraint. Prior to this change, LOPF
   silently dropped the term (only ``network.lpf()`` / ``network.pf()``
   respected it).

2. **New feature (issue #456)**: ``phase_shift_extendable=True`` makes
   ``phase_shift`` a per-snapshot decision variable bounded by
   ``phase_shift_min`` / ``phase_shift_max`` (degrees). The optimised values
   are written to ``n.transformers_t["phase_shift_opt"]``.

Physical setup: two parallel branches between buses A and B.

* Line L1 with ``x_pu = x / v_nom² = 0.01``.
* Transformer T1 with ``x_pu = x / s_nom = 0.01``.

With equal per-unit reactance and no phase shift, flow splits 50/50.
With a phase shift ``φ`` on T1 (cycle traversed L1 forward, T1 backward):

    x_pu * P_L − (x_pu * P_T + φ_rad) = 0   →   P_L − P_T = φ_rad / x_pu

Generator only at A (cheap) with 50 MW load at B → total flow A→B = 50 MW.
"""

from __future__ import annotations

import numpy as np
import pytest

pypsa = pytest.importorskip("pypsa")


def _build(phase_shift: float = 0.0, extendable: bool = False,
           phase_shift_min: float = -20.0,
           phase_shift_max: float = 20.0) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots([0, 1])
    n.add("Carrier", "AC")
    n.add("Bus", "A", v_nom=1.0, carrier="AC")
    n.add("Bus", "B", v_nom=1.0, carrier="AC")
    n.add("Generator", "gen_A", bus="A", p_nom=100, marginal_cost=10.0,
          carrier="AC")
    n.add("Load", "load_B", bus="B", p_set=[50.0, 50.0])
    n.add("Line", "L1", bus0="A", bus1="B", x=0.01, r=1e-6, s_nom=100,
          carrier="AC")
    n.add(
        "Transformer",
        "T1",
        bus0="A",
        bus1="B",
        x=1.0,   # x_pu = x / s_nom = 0.01
        r=1e-6,
        s_nom=100,
        phase_shift=phase_shift,
        phase_shift_extendable=extendable,
        phase_shift_min=phase_shift_min,
        phase_shift_max=phase_shift_max,
    )
    return n


class TestStaticPhaseShiftInKVL:
    """Issue #1220 bug fix: static phase_shift respected by LOPF."""

    def test_baseline_equal_split(self):
        """No phase shift → 25/25 flow split (equal x_pu)."""
        n = _build(phase_shift=0.0)
        n.optimize(solver_name="highs")
        assert n.lines_t.p0["L1"].iloc[0] == pytest.approx(25.0, abs=0.1)
        assert n.transformers_t.p0["T1"].iloc[0] == pytest.approx(25.0, abs=0.1)

    def test_positive_phase_shift_shifts_flow(self):
        """+10 deg on T1 → analytically derived shift of 17.45 MW between paths."""
        n = _build(phase_shift=10.0)
        n.optimize(solver_name="highs")
        x_pu = 0.01
        expected_delta = (10.0 * np.pi / 180.0) / x_pu  # ≈ 17.453 MW
        L1 = n.lines_t.p0["L1"].iloc[0]
        T1 = n.transformers_t.p0["T1"].iloc[0]
        assert L1 + T1 == pytest.approx(50.0, abs=0.1)
        assert abs(L1 - T1) == pytest.approx(expected_delta, abs=0.1)

    def test_negative_phase_shift_reverses_sign(self):
        """-10 deg swaps which branch carries more flow vs +10 deg."""
        n_pos = _build(phase_shift=10.0)
        n_pos.optimize(solver_name="highs")
        n_neg = _build(phase_shift=-10.0)
        n_neg.optimize(solver_name="highs")
        # The dominant branch flips between +10 and -10 cases
        sign_pos = np.sign(n_pos.lines_t.p0["L1"].iloc[0]
                           - n_pos.transformers_t.p0["T1"].iloc[0])
        sign_neg = np.sign(n_neg.lines_t.p0["L1"].iloc[0]
                           - n_neg.transformers_t.p0["T1"].iloc[0])
        assert sign_pos * sign_neg == -1.0


class TestExtendablePhaseShift:
    """Issue #456 feature: phase_shift as an optimisation variable."""

    def test_extendable_registers_variable(self):
        """phase_shift_extendable=True creates a linopy variable."""
        n = _build(extendable=True)
        n.optimize(solver_name="highs")
        assert "Transformer-phase_shift" in n.model.variables

    def test_extendable_writes_phase_shift_opt(self):
        """Optimised per-snapshot values end up in n.transformers_t.phase_shift_opt."""
        n = _build(extendable=True)
        n.optimize(solver_name="highs")
        assert "T1" in n.transformers_t.phase_shift_opt.columns
        values = n.transformers_t.phase_shift_opt["T1"].values
        assert len(values) == len(n.snapshots)

    def test_extendable_respects_bounds(self):
        """Optimised phase_shift stays within [phase_shift_min, phase_shift_max]."""
        n = _build(extendable=True, phase_shift_min=-15.0, phase_shift_max=5.0)
        n.optimize(solver_name="highs")
        vals = n.transformers_t.phase_shift_opt["T1"].values
        assert (vals >= -15.0 - 1e-6).all()
        assert (vals <= 5.0 + 1e-6).all()

    def test_extendable_preserves_static_phase_shift(self):
        """The input attribute `phase_shift` is not overwritten by assign_solution."""
        n = _build(phase_shift=0.0, extendable=True)
        assert n.transformers.at["T1", "phase_shift"] == 0.0
        n.optimize(solver_name="highs")
        # Static input column must remain unchanged
        assert n.transformers.at["T1", "phase_shift"] == 0.0

    def test_static_only_no_variable_registered(self):
        """phase_shift_extendable=False (default) does not add a variable."""
        n = _build(phase_shift=5.0, extendable=False)
        n.optimize(solver_name="highs")
        assert "Transformer-phase_shift" not in n.model.variables


class TestMixedStaticAndExtendable:
    """One network with both a static-phase-shift transformer and an extendable one."""

    def test_mixed_solve(self):
        """Both kinds coexist in the same KVL constraint without error."""
        n = _build(phase_shift=0.0, extendable=False)
        # Add an extendable twin between the same buses
        n.add(
            "Transformer",
            "T2",
            bus0="A",
            bus1="B",
            x=1.0,
            r=1e-6,
            s_nom=100,
            phase_shift=0.0,
            phase_shift_extendable=True,
            phase_shift_min=-20.0,
            phase_shift_max=20.0,
        )
        # Give T1 a static shift so the KVL has a non-zero RHS
        n.transformers.at["T1", "phase_shift"] = 5.0
        n.optimize(solver_name="highs")
        # Both components should have carried flow summing to the load
        total = (
            n.lines_t.p0["L1"].iloc[0]
            + n.transformers_t.p0["T1"].iloc[0]
            + n.transformers_t.p0["T2"].iloc[0]
        )
        assert total == pytest.approx(50.0, abs=0.1)
        # Extendable var present, static input unchanged
        assert "Transformer-phase_shift" in n.model.variables
        assert n.transformers.at["T1", "phase_shift"] == 5.0
