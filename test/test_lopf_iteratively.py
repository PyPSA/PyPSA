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
    assert all(n.lines.query("s_nom_extendable").s_nom_opt % line_unit_size == 0.0)
    assert all(
        n.links.query("p_nom_extendable and carrier == 'HVDC'").p_nom_opt
        % link_unit_size["HVDC"]
        == 0.0
    )
