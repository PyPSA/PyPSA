from packaging.version import Version, parse

try:
    # pypower is not maintained and with recent numpy verion it breaks
    from pypower.api import case118 as case
    from pypower.api import ppoption, runpf
    from pypower.ppver import ppver

    pypower_version = parse(ppver()["Version"])
except ImportError:
    pypower_version = Version("0.0.0")

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as equal

import pypsa


@pytest.mark.skipif(
    pypower_version <= Version("5.0.0"),
    reason="PyPOWER 5.0.0 is broken with recent numpy and unmaintained since Aug 2017.",
)
def test_pypower_case():
    # ppopt is a dictionary with the details of the optimization routine to run
    ppopt = ppoption(PF_ALG=2)

    # choose DC or AC
    ppopt["PF_DC"] = False

    # ppc is a dictionary with details about the network, including baseMVA, branches and generators
    ppc = case()

    results, success = runpf(ppc, ppopt)

    # branches
    columns = "bus0, bus1, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax, p0, q0, p1, q1".split(
        ", "
    )
    results_df = {"branch": pd.DataFrame(data=results["branch"], columns=columns)}
    # buses
    columns = [
        "bus",
        "type",
        "Pd",
        "Qd",
        "Gs",
        "Bs",
        "area",
        "v_mag_pu",
        "v_ang",
        "v_nom",
        "zone",
        "Vmax",
        "Vmin",
    ]
    results_df["bus"] = pd.DataFrame(
        data=results["bus"], columns=columns, index=results["bus"][:, 0]
    )

    # generators
    columns = "bus, p, q, q_max, q_min, Vg, mBase, status, p_max, p_min, Pc1, Pc2, Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf".split(
        ", "
    )
    results_df["gen"] = pd.DataFrame(data=results["gen"], columns=columns)

    # now compute in PyPSA

    n = pypsa.Network()
    n.import_from_pypower_ppc(ppc)

    # PYPOWER uses PI model for transformers, whereas PyPSA defaults to
    # T since version 0.8.0
    n.transformers.model = "pi"

    n.pf()

    # compare branch flows
    for c in n.iterate_components(n.passive_branch_components):
        for si in ["p0", "p1", "q0", "q1"]:
            si_pypsa = getattr(c.dynamic, si).loc["now"].values
            si_pypower = results_df["branch"][si][c.static.original_index].values
            equal(si_pypsa, si_pypower)

    # compare generator dispatch
    for s in ["p", "q"]:
        s_pypsa = getattr(n.generators_t, s).loc["now"].values
        s_pypower = results_df["gen"][s].values
        equal(s_pypsa, s_pypower)

    # compare voltages
    v_mag_pypsa = n.buses_t.v_mag_pu.loc["now"]
    v_mag_pypower = results_df["bus"]["v_mag_pu"]

    equal(v_mag_pypsa, v_mag_pypower)

    v_ang_pypsa = n.buses_t.v_ang.loc["now"]
    pypower_slack_angle = results_df["bus"]["v_ang"][
        results_df["bus"]["type"] == 3
    ].values[0]
    v_ang_pypower = (results_df["bus"]["v_ang"] - pypower_slack_angle) * np.pi / 180.0

    equal(v_ang_pypsa, v_ang_pypower)
