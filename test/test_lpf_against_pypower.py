from packaging.version import Version, parse

# NB: this test doesn't work for other cases because transformer tap
# ratio and phase angle not supported for lpf
try:
    # pypower is not maintained and with recent numpy verion it breaks
    from pypower.api import case30 as case
    from pypower.api import ppoption, runpf
    from pypower.ppver import ppver

    pypower_version = parse(ppver()["Version"])
except ImportError:
    pypower_version = Version("0.0.0")

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
    ppopt["PF_DC"] = True

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
        "v_mag_pu_set",
        "v_ang_set",
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
    n.lpf()

    # compare generator dispatch
    p_pypsa = n.generators_t.p.loc["now"].values
    p_pypower = results_df["gen"]["p"].values

    equal(p_pypsa, p_pypower)

    # compare branch flows
    for item in ["lines", "transformers"]:
        df = getattr(n, item)
        dynamic = getattr(n, item + "_t")

        for si in ["p0", "p1"]:
            si_pypsa = getattr(dynamic, si).loc["now"].values
            si_pypower = results_df["branch"][si][df.original_index].values
            equal(si_pypsa, si_pypower)
