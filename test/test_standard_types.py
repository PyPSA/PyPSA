# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for deriving branch impedances from standard types."""

import pytest

import pypsa

OVERRIDE_MSG = "overridden by the standard type"


@pytest.fixture(params=["Transformer", "Line"])
def typed_branch(request):
    """Network with a single typed branch named 'branch'."""
    component = request.param
    n = pypsa.Network()
    if component == "Transformer":
        n.add("Bus", ["b0", "b1"], v_nom=[20.0, 0.4])
        n.add("Transformer", "branch", bus0="b0", bus1="b1", type="0.25 MVA 20/0.4 kV")
    else:
        n.add("Bus", ["b0", "b1"], v_nom=380.0)
        type_name = "Al/St 240/40 4-bundle 380.0"
        n.add("Line", "branch", bus0="b0", bus1="b1", length=10.0, type=type_name)
    return n, n.components[component].static


def test_type_derives_impedance_silently(typed_branch, caplog):
    n, static = typed_branch
    n.calculate_dependent_values()
    assert static.loc["branch", "x"] > 0
    assert OVERRIDE_MSG not in caplog.text


def test_manual_impedance_override_warns(typed_branch, caplog):
    """See https://github.com/PyPSA/PyPSA/issues/1054."""
    n, static = typed_branch
    n.calculate_dependent_values()

    static.loc["branch", ["r", "x"]] = 0.01
    n.calculate_dependent_values()
    assert OVERRIDE_MSG in caplog.text
    assert static.loc["branch", "r"] != 0.01

    caplog.clear()
    n.calculate_dependent_values()
    assert OVERRIDE_MSG not in caplog.text
