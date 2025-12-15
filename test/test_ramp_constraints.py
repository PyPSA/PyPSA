# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

import pypsa


@pytest.mark.parametrize("direction", ["up", "down"])
def test_generator_ramp_constraints_mask_nan(direction):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=2, freq="h"))

    n.add("Bus", "bus")
    # Add load with sudden change to trigger ramping up/down
    p_set = [0.0, 50.0] if direction == "up" else [50.0, 10.0]
    n.add(
        "Load",
        "load",
        bus="bus",
        p_set=pd.Series(p_set, index=n.snapshots),
    )

    ramp_kw = {f"ramp_limit_{direction}": 0.5}

    # Generator with ramp limits
    n.add(
        "Generator",
        "gen_limited",
        bus="bus",
        p_nom=10,
        marginal_cost=1,
        **ramp_kw,
    )

    # Generator without ramp limits
    n.add(
        "Generator",
        "gen_unlimited",
        bus="bus",
        p_nom=1000,
        marginal_cost=10,
    )

    n.optimize(solver_name="highs")

    # Check labels to see which generators have active ramp constraints applied
    # Generator with undefined ramp limits should have label -1 (no constraint)
    key = f"Generator-fix-p-ramp_limit_{direction}"
    constraints = n.model.constraints[key]
    labels_gen_limited = constraints.data["labels"].sel(name="gen_limited").to_pandas()
    assert (labels_gen_limited != -1).all(), (
        f"ramp_{direction} constraint should be active for 'gen_limited'."
    )

    labels_gen_unlimited = (
        constraints.data["labels"].sel(name="gen_unlimited").to_pandas()
    )
    assert (labels_gen_unlimited == -1).all(), (
        f"ramp_{direction} constraint should be masked for 'gen_unlimited'."
    )


@pytest.mark.parametrize("direction", ["up", "down"])
def test_link_ramp_constraints_mask_nan(direction):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=2, freq="h"))

    n.add("Bus", "bus0")
    n.add("Bus", "bus1")

    # Add load with sudden change to trigger ramping up/down
    p_set = [0.0, 50.0] if direction == "up" else [50.0, 10.0]
    n.add(
        "Load",
        "load",
        bus="bus1",
        p_set=pd.Series(p_set, index=n.snapshots),
    )

    ramp_kw = {
        f"ramp_limit_{direction}": 0.5,
    }

    # Link with ramp limits
    n.add(
        "Link",
        "link_limited",
        bus0="bus0",
        bus1="bus1",
        p_nom=10,
        efficiency=1.0,
        marginal_cost=1,
        **ramp_kw,
    )

    # Link without ramp limits
    n.add(
        "Link",
        "link_unlimited",
        bus0="bus0",
        bus1="bus1",
        p_nom=1000,
        efficiency=1.0,
        marginal_cost=10,
    )

    n.add(
        "Generator",
        "generator",
        bus="bus0",
        p_nom=1000,
        marginal_cost=0.0,
    )

    n.optimize(solver_name="highs")

    # Check labels to see which links have active ramp constraints applied
    # Link with undefined ramp limits should have label -1 (no constraint)
    key = f"Link-fix-p-ramp_limit_{direction}"
    constraints = n.model.constraints[key]
    labels_link_limited = (
        constraints.data["labels"].sel(name="link_limited").to_pandas()
    )
    assert (labels_link_limited != -1).all(), (
        f"ramp_{direction} constraint should be active for 'link_limited'."
    )

    labels_link_unlimited = (
        constraints.data["labels"].sel(name="link_unlimited").to_pandas()
    )
    assert (labels_link_unlimited == -1).all(), (
        f"ramp_{direction} constraint should be masked for 'link_unlimited'."
    )
