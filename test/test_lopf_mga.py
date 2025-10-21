# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest
from numpy.testing import assert_almost_equal as almost_equal

import pypsa


def test_mga():
    n = pypsa.Network()

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        marginal_cost=20,
        capital_cost=200,
        p_nom_extendable=True,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        marginal_cost=40,
        capital_cost=230,
        p_nom_extendable=True,
    )

    n.add("Load", "load", bus="bus", p_set=100)

    # can only run MGA on solved networks
    with pytest.raises(ValueError):
        n.optimize.optimize_mga()

    n.optimize()

    opt_capacity = n.c.generators.static.p_nom_opt
    opt_cost = (n.statistics.capex() + n.statistics.opex()).sum()

    weights = {"Generator": {"p_nom": {"coal": 1}}}
    slack = 0.05
    n.optimize.optimize_mga(slack=0.05, weights=weights)

    mga_capacity = n.c.generators.static.p_nom_opt
    mga_cost = (n.statistics.capex() + n.statistics.opex()).sum()

    assert mga_capacity["coal"] <= opt_capacity["coal"]
    almost_equal(mga_cost / opt_cost, 1 + slack)


def test_mga_in_direction():
    n = pypsa.Network()
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "gen1",
        bus="bus",
        marginal_cost=10,
        capital_cost=99,
        p_nom_extendable=True,
    )
    n.add(
        "Generator",
        "gen2",
        bus="bus",
        marginal_cost=10,
        capital_cost=100,
        p_nom_extendable=True,
    )
    n.add("Load", "load", bus="bus", p_set=100)
    n.optimize()

    # Define dimensions for the MGA
    dimensions = {
        "cap1": {"Generator": {"p_nom": {"gen1": 1}}},
        "cap2": {"Generator": {"p_nom": {"gen2": 1}}},
    }

    # Test with a simple direction
    direction = {"cap1": -1, "cap2": 1}  # Minimize coal, maximize gas
    slack = 0.05
    status, condition, coords = n.optimize.optimize_mga_in_direction(
        direction=direction, dimensions=dimensions, slack=slack
    )

    assert status == "ok"
    assert "cap1" in coords
    assert "cap2" in coords
    assert n.meta["slack"] == slack
    assert n.meta["direction"] == direction
    assert n.meta["dimensions"] == {
        "cap1": {"Generator": {"p_nom": {"gen1": 1}}},
        "cap2": {"Generator": {"p_nom": {"gen2": 1}}},
    }

    # Assert that the capacity of gen1 is less than gen2
    assert (
        n.c.generators.static.p_nom_opt["gen1"]
        < n.c.generators.static.p_nom_opt["gen2"]
    )

    # Test error before solving network
    n_unsolved = pypsa.Network()
    n_unsolved.add("Bus", "bus")
    n_unsolved.add("Load", "load", bus="bus", p_set=100)
    n_unsolved.add(
        "Generator",
        "some_gen",
        bus="bus",
        marginal_cost=10,
        capital_cost=100,
        p_nom_extendable=True,
    )
    with pytest.raises(ValueError):
        n_unsolved.optimize.optimize_mga_in_direction(
            direction={"some_gen_cap": 1},
            dimensions={"some_gen_cap": {"Generator": {"p_nom": {"some_gen": 1}}}},
        )

    # Test inconsistent direction/dimensions keys
    with pytest.raises(
        ValueError, match="Keys of `direction` and `dimensions` arguments must match."
    ):
        n.optimize.optimize_mga_in_direction(
            direction={"cap1": -1},
            dimensions={"cap2": {"Generator": {"p_nom": {"gen2": 1}}}},
        )


def test_mga_in_multiple_directions():
    n = pypsa.Network()
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "coal",
        bus="bus",
        marginal_cost=20,
        capital_cost=200,
        p_nom_extendable=True,
    )
    n.add(
        "Generator",
        "gas",
        bus="bus",
        marginal_cost=40,
        capital_cost=230,
        p_nom_extendable=True,
    )
    n.add("Load", "load", bus="bus", p_set=100)
    n.optimize()

    # Define dimensions for the MGA
    dimensions = {
        "coal_cap": {"Generator": {"p_nom": {"coal": 1}}},
        "gas_cap": {"Generator": {"p_nom": {"gas": 1}}},
    }

    # Generate some example directions
    directions_list = [
        {"coal_cap": -1, "gas_cap": 1},
        {"coal_cap": 1, "gas_cap": -1},
    ]

    successful_directions, successful_coordinates = (
        n.optimize.optimize_mga_in_multiple_directions(
            directions=directions_list,
            dimensions=dimensions,
            max_parallel=1,  # use 1 for reliable testing
        )
    )

    assert not successful_directions.empty
    assert not successful_coordinates.empty
    assert len(successful_directions) <= len(directions_list)
    assert len(successful_coordinates) <= len(directions_list)
    assert "coal_cap" in successful_coordinates.columns
    assert "gas_cap" in successful_coordinates.columns


def test_generate_directions():
    keys = ["dim1", "dim2", "dim3"]
    n_directions = 5

    # Test generate_directions_random
    random_directions = pypsa.optimization.mga.generate_directions_random(
        keys, n_directions, seed=0
    )
    assert random_directions.shape == (n_directions, len(keys))
    for _, row in random_directions.iterrows():
        almost_equal(sum(val**2 for val in row), 1.0)  # Check unit vector normalization

    # Test generate_directions_evenly_spaced
    keys_2d = ["x", "y"]
    n_directions_2d = 4
    evenly_spaced_directions = pypsa.optimization.mga.generate_directions_evenly_spaced(
        keys_2d, n_directions_2d
    )
    assert evenly_spaced_directions.shape == (n_directions_2d, len(keys_2d))
    for _, row in evenly_spaced_directions.iterrows():
        almost_equal(sum(val**2 for val in row), 1.0)  # Check unit vector normalization
    with pytest.raises(ValueError, match="This function only supports two keys"):
        pypsa.optimization.mga.generate_directions_evenly_spaced(keys, n_directions)

    # Test generate_directions_halton
    halton_directions = pypsa.optimization.mga.generate_directions_halton(
        keys, n_directions, seed=0
    )
    assert halton_directions.shape == (n_directions, len(keys))
    for _, row in halton_directions.iterrows():
        almost_equal(sum(val**2 for val in row), 1.0)  # Check unit vector normalization
