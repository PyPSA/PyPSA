# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

import pypsa


@pytest.mark.parametrize("carrier", [None, "unserved energy"])
def test_load_shedding_carrier(carrier):
    n = pypsa.Network()
    n.add("Bus", ["a", "b"])

    kwargs = {} if carrier is None else {"carrier": carrier}
    n.optimize.add_load_shedding(**kwargs)
    names = n.c.generators.static.index
    expected = "load shedding" if carrier is None else carrier

    assert names.equals(pd.Index(["a load shedding", "b load shedding"], name="name"))
    assert (n.c.generators.static.loc[names, "carrier"] == expected).all()
    assert n.c.carriers.static.index.tolist() == [expected]
    n.consistency_check()


def test_load_shedding_preserves_existing_carrier():
    n = pypsa.Network()
    n.add("Bus", "a")
    n.add("Carrier", "unserved energy", color="#123456")

    n.optimize.add_load_shedding(carrier="unserved energy")

    assert n.c.carriers.static.index.tolist() == ["unserved energy"]
    assert n.c.carriers.static.at["unserved energy", "color"] == "#123456"


def test_load_shedding_keeps_positional_arguments():
    n = pypsa.Network()
    n.add("Bus", ["a", "b"])

    n.optimize.add_load_shedding(" backup", pd.Index(["b"]), 1.0, 200.0, 50.0)
    names = n.c.generators.static.index

    assert names.tolist() == ["b backup"]
    generator = n.c.generators.static.loc["b backup"]
    assert generator["bus"] == "b"
    assert generator["sign"] == 1.0
    assert generator["marginal_cost"] == 200.0
    assert generator["p_nom"] == 50.0
