# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the optional SMS++ accessor."""

import pypsa
import pytest


# Flag if SMSpp is available
try:
    import pypsa2smspp
    import pysmspp
    
    SMSPP_IS_AVAILABLE = pysmspp.pysmspp.is_smspp_installed()
except:
    SMSPP_IS_AVAILABLE = False


@pytest.mark.skipif(not SMSPP_IS_AVAILABLE, reason="SMS++ not installed")
def test_smspp_solving(monkeypatch):
    
    n = pypsa.examples.ac_dc_meshed()

    n.optimize(solver_name="smspp")
