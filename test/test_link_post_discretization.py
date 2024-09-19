import pytest

import pypsa

from pypsa.optimization.abstract import discretized_capacity

def build_network(unit_size=10):
    ''''
    Build a network with two buses and links that are have
    - p_nom_max = unit_size
    - p_nom_max = 2*unit_size
    - p_nom_max < unit_size
    - p_nom_max > unit_size and not a multiple of unit_size
    - p_nom_max = np.inf
    '''
    n = pypsa.Network()

    # add buses
    n.add("Bus",
          "Bus0")
    n.add("Bus",
          "Bus1")

    # add links
    n.add("Link",
          "Link0",
          bus0="Bus0",
          bus1="Bus1",
          p_nom=0,
          p_nom_opt=0.5*unit_size,
          p_nom_max=unit_size,
          p_nom_extendable=True,
          )

    n.add("Link",
          "Link1",
          bus0="Bus0",
          bus1="Bus1",
          p_nom=0,
          p_nom_opt=1.5*unit_size,
          p_nom_max=2*unit_size,
          p_nom_extendable=True,
          )
    n.add("Link",
          "Link2",
          bus0="Bus0",
          bus1="Bus1",
          p_nom=0,
          p_nom_opt=0.5*unit_size,
          p_nom_max=0.8*unit_size,
          p_nom_extendable=True,
          )
    n.add("Link",
          "Link3",
          bus0="Bus0",
          bus1="Bus1",
          p_nom=0,
          p_nom_opt=1.3*unit_size,
          p_nom_max=1.5*unit_size,
          p_nom_extendable=True,
          )
    n.add("Link",
          "Link4",
          bus0="Bus0",
          bus1="Bus1",
          p_nom=0,
          p_nom_opt=1.5*unit_size,
          p_nom_extendable=True,
          )
    return n
    
def test_post_discretization():
    """
    This test checks the post discretization function.
    If a Link has a p_nom_max that is not a multiple of the unit_size,
    depending on the variable fractional_last_unit_size the p_nom should
    either be the last full unit_size or the p_nom_max.
    """

    unit_size = 10
    threshold = 0.1

    n = build_network(unit_size=unit_size)

    n.links["p_nom"] = n.links.apply(
        lambda row: discretized_capacity(
            nom_opt=row["p_nom_opt"],
            nom_max=row["p_nom_max"],
            unit_size=unit_size,
            threshold=threshold,
            fractional_last_unit_size=True,
        ),
        axis=1,
    )

    assert (n.links.loc["Link0"].p_nom == unit_size)
    assert (n.links.loc["Link1"].p_nom == 2*unit_size)
    assert (n.links.loc["Link2"].p_nom == 0.8*unit_size)
    assert (n.links.loc["Link3"].p_nom == 1.5*unit_size)
    assert (n.links.loc["Link4"].p_nom == 2*unit_size)

    n = build_network(unit_size=unit_size)

    n.links["p_nom"] = n.links.apply(
        lambda row: discretized_capacity(
            nom_opt=row["p_nom_opt"],
            nom_max=row["p_nom_max"],
            unit_size=unit_size,
            threshold=threshold,
            fractional_last_unit_size=False,
        ),
        axis=1,
    )
    
    assert (n.links.loc["Link0"].p_nom == unit_size)
    assert (n.links.loc["Link1"].p_nom == 2*unit_size)
    assert (n.links.loc["Link2"].p_nom == 0)
    assert (n.links.loc["Link3"].p_nom == unit_size)
    assert (n.links.loc["Link4"].p_nom == 2*unit_size)