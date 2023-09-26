# -*- coding: utf-8 -*-
import numpy as np
import pytest
from conftest import SUPPORTED_APIS, optimize

import pypsa

COUPLING_COMPONENTS_APIS = ["linopy"]


@pytest.mark.parametrize("api", COUPLING_COMPONENTS_APIS)
def test_generator_p_coupling(api):
    n = pypsa.Network(snapshots=range(3))

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[1, 2, 3])
    n.add(
        "Generator",
        "ocgt",
        bus="bus",
        p_nom=1,
        p_nom_extendable=True,
        marginal_cost=1,
    )

    # add coupled generator
    coeff = 0.6  # 600 kg emissions per 1 MWel output from ocgt
    n.add("Bus", "co2", carrier="co2", unit="t")
    n.add(
        "Generator",
        "ocgt - co2",
        bus="co2",
        p_coupling="ocgt",
        p_coupling_coeff=coeff,
        p_nom=np.inf,
    )
    n.add("Store", "co2", bus="co2", e_nom=20)

    status, condition = optimize(n, api)

    assert status == "ok"
    assert condition == "optimal"
    assert n.generators_t.p["ocgt - co2"].div(n.generators_t.p["ocgt"]).eq(coeff).all()
    assert (n.stores_t.e["co2"] == [0.6, 1.8, 3.6]).all()
