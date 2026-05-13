# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the optional SMS++ accessor."""

import pytest

import pypsa

pytest.importorskip("pypsa2smspp")
pysmspp = pytest.importorskip("pysmspp")

SMSPP_IS_AVAILABLE = pysmspp.is_smspp_installed()


@pytest.mark.skipif(not SMSPP_IS_AVAILABLE, reason="SMS++ not installed")
def test_smspp_solving():
    nsms = pypsa.examples.ac_dc_meshed()

    # global constraints not yet supported
    nsms.remove("GlobalConstraint", nsms.global_constraints.index)

    # create base case on a fresh copy
    norig = nsms.copy()
    norig.optimize(solver_name="highs")

    nsms.optimize(solver_name="smspp")

    assert nsms.objective + nsms.objective_constant == pytest.approx(
        norig.objective + norig.objective_constant, rel=1e-3
    )
