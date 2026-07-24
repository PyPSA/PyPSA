# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for Transformer ``phase_shift`` in LOPF.

Covers two related capabilities:

1. **Bug fix for issue #1220**: a fixed ``phase_shift`` is now included in
   the cycle-based Kirchhoff Voltage Law constraint. Prior to this change, LOPF
   silently dropped the term (only ``network.lpf()`` / ``network.pf()``
   respected it).

2. **New feature (issue #456)**: when ``phase_shift_min < phase_shift_max`` the
   shift becomes a per-snapshot decision variable bounded by those two
   attributes (degrees). The optimised per-snapshot values are written to the
   dynamic output ``n.transformers_t["phase_shift_opt"]``, while the input
   ``phase_shift`` is preserved.

Physical setup: two parallel branches between buses A and B.

* Line L1 with ``x_pu = x / v_nom² = 0.01``.
* Transformer T1 with ``x_pu = x / s_nom = 0.01``.

With equal per-unit reactance and no phase shift, flow splits 50/50.
With a phase shift ``φ`` on T1 (cycle traversed L1 forward, T1 backward):

    x_pu * P_L − (x_pu * P_T + φ_rad) = 0   →   P_L − P_T = φ_rad / x_pu

Generator only at A (cheap) with 50 MW load at B → total flow A→B = 50 MW.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pypsa


def _build(
    phase_shift: float = 0.0,
    variable: bool = False,
    phase_shift_min: float = -20.0,
    phase_shift_max: float = 20.0,
) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots([0, 1])
    n.add("Carrier", "AC")
    n.add("Bus", "A", v_nom=1.0, carrier="AC")
    n.add("Bus", "B", v_nom=1.0, carrier="AC")
    n.add("Generator", "gen_A", bus="A", p_nom=100, marginal_cost=10.0, carrier="AC")
    n.add("Load", "load_B", bus="B", p_set=[50.0, 50.0])
    n.add("Line", "L1", bus0="A", bus1="B", x=0.01, r=1e-6, s_nom=100, carrier="AC")
    bounds: dict[str, Any] = (
        {"phase_shift_min": phase_shift_min, "phase_shift_max": phase_shift_max}
        if variable
        else {}
    )
    n.add(
        "Transformer",
        "T1",
        bus0="A",
        bus1="B",
        x=1.0,  # x_pu = x / s_nom = 0.01
        r=1e-6,
        s_nom=100,
        phase_shift=phase_shift,
        **bounds,
    )
    return n


class TestFixedPhaseShiftInKVL:
    """Issue #1220 bug fix: fixed phase_shift respected by LOPF."""

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
        # Absolute direction: +phi moves flow OFF the shifted transformer (T1)
        # onto the parallel line (L1), per PyPSA's P = (theta0 - theta1 - phi)/x
        # convention (power_flow.SubNetwork.calculate_B_H). This pins the sign,
        # which the abs()/sign-flip asserts above are invariant to (the #1220
        # failure mode).
        assert L1 > T1

    def test_negative_phase_shift_reverses_sign(self):
        """-10 deg swaps which branch carries more flow vs +10 deg."""
        n_pos = _build(phase_shift=10.0)
        n_pos.optimize(solver_name="highs")
        n_neg = _build(phase_shift=-10.0)
        n_neg.optimize(solver_name="highs")
        sign_pos = np.sign(
            n_pos.lines_t.p0["L1"].iloc[0] - n_pos.transformers_t.p0["T1"].iloc[0]
        )
        sign_neg = np.sign(
            n_neg.lines_t.p0["L1"].iloc[0] - n_neg.transformers_t.p0["T1"].iloc[0]
        )
        assert sign_pos * sign_neg == -1.0


class TestVariablePhaseShift:
    """Issue #456 feature: phase_shift as an optimisation variable (min < max)."""

    def test_variable_writes_dynamic_phase_shift(self):
        """Optimised per-snapshot values end up in n.transformers_t.phase_shift_opt."""
        n = _build(variable=True)
        n.optimize(solver_name="highs")
        assert "T1" in n.transformers_t.phase_shift_opt.columns
        values = n.transformers_t.phase_shift_opt["T1"].values
        assert len(values) == len(n.snapshots)

    def test_variable_respects_bounds(self):
        """Optimised phase_shift stays within [phase_shift_min, phase_shift_max]."""
        n = _build(variable=True, phase_shift_min=-15.0, phase_shift_max=5.0)
        n.optimize(solver_name="highs")
        vals = n.transformers_t.phase_shift_opt["T1"].values
        assert (vals >= -15.0 - 1e-6).all()
        assert (vals <= 5.0 + 1e-6).all()

    def test_variable_preserves_set_input(self):
        """The input attribute `phase_shift` is not overwritten by the solve."""
        n = _build(phase_shift=3.0, variable=True)
        assert n.transformers.at["T1", "phase_shift"] == 3.0
        n.optimize(solver_name="highs")
        assert n.transformers.at["T1", "phase_shift"] == 3.0

    def test_fixed_no_variable_registered(self):
        """A fixed transformer (min == max) does not add a variable."""
        n = _build(phase_shift=5.0, variable=False)
        n.optimize(solver_name="highs")
        assert "Transformer-phase_shift" not in n.model.variables

    def test_fixed_phase_shift_reports_setpoint(self):
        """Fixed transformers report their setpoint in the realised phase_shift_opt."""
        n = _build(phase_shift=5.0, variable=False)
        n.optimize(solver_name="highs")
        assert (n.transformers_t.phase_shift_opt["T1"].values == 5.0).all()
        assert n.transformers.at["T1", "phase_shift"] == 5.0

    def test_variable_rejects_unbounded(self):
        """An optimisable shift with a non-finite bound fails fast."""
        n = _build(variable=True, phase_shift_min=-np.inf, phase_shift_max=np.inf)
        with pytest.raises(ValueError, match="non-finite"):
            n.optimize(solver_name="highs")

    def test_variable_only_cycle_keeps_kvl_row(self):
        """Zero fixed RHS must not drop the cycle's KVL row (rhs.notnull mask)."""
        n = _build(variable=True)
        n.optimize(solver_name="highs")
        kvl = n.model.constraints["Kirchhoff-Voltage-Law"]
        assert (kvl.labels != -1).any()


class TestMixedFixedAndVariable:
    """One network with both a fixed and a variable phase-shift transformer."""

    def test_mixed_solve(self):
        """Both kinds coexist in the same KVL constraint without error."""
        n = _build(phase_shift=5.0, variable=False)
        # Add a variable twin between the same buses
        n.add(
            "Transformer",
            "T2",
            bus0="A",
            bus1="B",
            x=1.0,
            r=1e-6,
            s_nom=100,
            phase_shift_min=-20.0,
            phase_shift_max=20.0,
        )
        n.optimize(solver_name="highs")
        total = (
            n.lines_t.p0["L1"].iloc[0]
            + n.transformers_t.p0["T1"].iloc[0]
            + n.transformers_t.p0["T2"].iloc[0]
        )
        assert total == pytest.approx(50.0, abs=0.1)
        # Variable var present, fixed input unchanged
        assert "Transformer-phase_shift" in n.model.variables
        assert n.transformers.at["T1", "phase_shift"] == 5.0
        # Realised angles: fixed (T1) equals its setpoint, variable (T2) optimised
        assert (n.transformers_t.phase_shift_opt["T1"].values == 5.0).all()
        assert "T2" in n.transformers_t.phase_shift_opt.columns
        t2_opt = n.transformers_t.phase_shift_opt["T2"].values
        assert ((t2_opt >= -20.0 - 1e-6) & (t2_opt <= 20.0 + 1e-6)).all()


class TestMultiInvestmentPeriod:
    """Phase shift must not leak across investment periods."""

    def test_fixed_phase_shift_isolated_per_period(self):
        n = pypsa.Network()
        n.set_snapshots([0, 1])
        n.investment_periods = [2020, 2030]
        n.add("Carrier", "AC")
        n.add("Bus", "A", v_nom=1.0, carrier="AC")
        n.add("Bus", "B", v_nom=1.0, carrier="AC")
        n.add(
            "Generator", "gen_A", bus="A", p_nom=100, marginal_cost=10.0, carrier="AC"
        )
        n.add("Load", "load_B", bus="B", p_set=50.0)
        n.add("Line", "L1", bus0="A", bus1="B", x=0.01, r=1e-6, s_nom=100, carrier="AC")
        # T1 carries a fixed phase shift but retires before 2030.
        n.add(
            "Transformer",
            "T1",
            bus0="A",
            bus1="B",
            x=1.0,
            r=1e-6,
            s_nom=100,
            phase_shift=10.0,
            build_year=2020,
            lifetime=5,
            carrier="AC",
        )
        # L2 only exists in 2030 to keep a cycle there (x ratio L1:L2 = 1:3).
        n.add(
            "Line",
            "L2",
            bus0="A",
            bus1="B",
            x=0.03,
            r=1e-6,
            s_nom=100,
            build_year=2030,
            lifetime=100,
            carrier="AC",
        )
        n.optimize(multi_investment_periods=True, solver_name="highs")

        # 2030: T1 retired → pure L1||L2 split by inverse reactance, unaffected
        # by T1's phase shift. 3:1 inverse-x split of 50 MW.
        assert n.lines_t.p0.loc[(2030, 0), "L1"] == pytest.approx(37.5, abs=0.1)
        assert n.lines_t.p0.loc[(2030, 0), "L2"] == pytest.approx(12.5, abs=0.1)
        # 2020: phase shift active, equal x_pu=0.01 and phi=10deg give
        # |L1-T1| = phi_rad/x_pu = 17.453 MW around 50 MW.
        assert n.lines_t.p0.loc[(2020, 0), "L1"] == pytest.approx(33.727, abs=0.1)
        assert n.transformers_t.p0.loc[(2020, 0), "T1"] == pytest.approx(
            16.273, abs=0.1
        )

    def test_variable_phase_shift_multi_period(self):
        """Variable PST must build and solve under multi-investment.

        The per-snapshot variable is registered over the (period, snapshot)
        MultiIndex; flattening it broke the KVL constraint assembly.
        """
        n = pypsa.Network()
        n.set_snapshots([0, 1])
        n.investment_periods = [2020, 2030]
        n.add("Carrier", "AC")
        n.add("Bus", "A", v_nom=1.0, carrier="AC")
        n.add("Bus", "B", v_nom=1.0, carrier="AC")
        n.add(
            "Generator", "gen_A", bus="A", p_nom=100, marginal_cost=10.0, carrier="AC"
        )
        n.add("Load", "load_B", bus="B", p_set=50.0)
        n.add("Line", "L1", bus0="A", bus1="B", x=0.01, r=1e-6, s_nom=100, carrier="AC")
        n.add(
            "Transformer",
            "T1",
            bus0="A",
            bus1="B",
            x=1.0,
            r=1e-6,
            s_nom=100,
            phase_shift_min=-20.0,
            phase_shift_max=20.0,
            build_year=2015,
            lifetime=100,
            carrier="AC",
        )
        n.optimize(multi_investment_periods=True, solver_name="highs")

        opt = n.transformers_t.phase_shift_opt["T1"]
        assert opt.index.equals(n.snapshots)
        assert ((opt >= -20.0 - 1e-6) & (opt <= 20.0 + 1e-6)).all()


class TestLOPFMatchesLinearPowerFlow:
    """Issue #1220 regression: LOPF flows must match n.lpf().

    ``n.lpf()`` is the authoritative DC oracle and respected the phase shift
    even before this PR, so it is the cleanest sign/magnitude check for the
    cycle-KVL term. Pinning LOPF to it locks the convention
    ``P = (theta0 - theta1 - phi) / x`` and guards against a wrong-sign "fix"
    silently reopening #1220.
    """

    @pytest.mark.parametrize("phase_shift", [-15.0, -5.0, 0.0, 5.0, 15.0])
    def test_lopf_flows_match_lpf(self, phase_shift):
        n = _build(phase_shift=phase_shift)
        n.optimize(solver_name="highs")
        lopf_L1 = n.lines_t.p0["L1"].copy()
        lopf_T1 = n.transformers_t.p0["T1"].copy()
        ref = _build(phase_shift=phase_shift)
        ref.lpf()
        np.testing.assert_allclose(
            ref.lines_t.p0["L1"].values, lopf_L1.values, atol=1e-3
        )
        np.testing.assert_allclose(
            ref.transformers_t.p0["T1"].values, lopf_T1.values, atol=1e-3
        )

    def test_variable_result_consistent_with_lpf(self):
        """Lock the variable (LHS variable) path to the fixed (RHS) path.

        Solve with a variable PST, read back the optimised angle, then feed
        that angle as a fixed input to a fresh network and confirm n.lpf()
        reproduces the LOPF flow. Proves the variable term carries the same
        sign and scaling as the fixed term.
        """
        n = _build(variable=True, phase_shift_min=-20.0, phase_shift_max=20.0)
        n.optimize(solver_name="highs")
        opt_angle = float(n.transformers_t.phase_shift_opt["T1"].iloc[0])
        lopf_L1 = n.lines_t.p0["L1"].copy()
        check = _build(phase_shift=opt_angle)
        check.lpf()
        np.testing.assert_allclose(
            check.lines_t.p0["L1"].values, lopf_L1.values, atol=1e-2
        )
