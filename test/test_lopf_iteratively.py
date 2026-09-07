# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest

import pypsa


def test_optimize_post_discretization():
    n = pypsa.Network()

    n.add("Bus", ["a", "b", "c"], v_nom=380.0)
    n.add("Generator", "generator", bus="a", p_nom=900.0, marginal_cost=10.0)
    n.add("Load", "load", bus="c", p_set=900.0)
    n.add(
        "Line",
        "ab",
        bus0="a",
        bus1="b",
        x=0.0001,
        s_nom_extendable=True,
        capital_cost=1000,
    )
    n.add(
        "Link",
        "bc",
        bus0="b",
        bus1="c",
        p_nom_extendable=True,
        capital_cost=1000,
        carrier="HVDC",
    )

    line_unit_size = 500
    link_unit_size = {"HVDC": 600}

    status, _ = n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=1,
        line_unit_size=line_unit_size,
        link_unit_size=link_unit_size,
        link_threshold={"HVDC": 0.4},
    )

    assert status == "ok"
    assert all(
        n.c.lines.static.query("s_nom_extendable").s_nom_opt % line_unit_size == 0.0
    )
    assert all(
        n.c.links.static.query("p_nom_extendable and carrier == 'HVDC'").p_nom_opt
        % link_unit_size["HVDC"]
        == 0.0
    )


def test_post_discretization_objective_overnight_cost():
    def build(**cost):
        n = pypsa.Network()
        n.snapshot_weightings.loc[:, :] = 8760.0
        n.add("Bus", ["a", "b", "c"], v_nom=380.0)
        n.add("Generator", "generator", bus="a", p_nom=900.0, marginal_cost=10.0)
        n.add("Load", "load", bus="c", p_set=900.0)
        n.add("Line", "ab", bus0="a", bus1="b", x=0.0001, s_nom_extendable=True, **cost)
        n.add(
            "Link",
            "bc",
            bus0="b",
            bus1="c",
            p_nom_extendable=True,
            carrier="HVDC",
            **cost,
        )
        n.optimize.optimize_transmission_expansion_iteratively(
            max_iterations=1,
            line_unit_size=500,
            link_unit_size={"HVDC": 600},
            link_threshold={"HVDC": 0.4},
        )
        return n

    direct = build(capital_cost=100)
    overnight = build(overnight_cost=1000, discount_rate=0, lifetime=10)
    assert overnight.objective == pytest.approx(direct.objective)
