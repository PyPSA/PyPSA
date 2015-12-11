

import pypsa

from pypower.api import ppoption, runpf, case30 as case


import pandas as pd

import numpy as np




def test_pypower_case():

    #ppopt is a dictionary with the details of the optimization routine to run
    ppopt = ppoption(PF_ALG=2)

    #choose DC or AC
    ppopt["PF_DC"] = True

    #ppc is a dictionary with details about the network, including baseMVA, branches and generators
    ppc = case()

    results,success = runpf(ppc, ppopt)

    #store results in a DataFrame for easy access
    results_df = {}

    #branches
    columns = 'bus0, bus1, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax, p0, q0, p1, q1'.split(", ")
    results_df['branch'] = pd.DataFrame(data=results["branch"],columns=columns)

    #buses
    columns = ["bus","type","Pd","Qd","Gs","Bs","area","v_mag_set","v_ang_set","v_nom","zone","Vmax","Vmin"]
    results_df['bus'] = pd.DataFrame(data=results["bus"],columns=columns)

    #generators
    columns = "bus, p, q, q_max, q_min, Vg, mBase, status, p_max, p_min, Pc1, Pc2, Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf".split(", ")
    results_df['gen'] = pd.DataFrame(data=results["gen"],columns=columns)



    #now compute in PyPSA

    network = pypsa.Network()
    network.import_from_pypower_ppc(ppc)
    network.lpf()



    #compare generator dispatch

    p_pypsa = np.array([gen.p[network.now] for gen in network.generators.itervalues()])
    p_pypower = results_df['gen']["p"].values

    np.testing.assert_array_almost_equal(p_pypsa,p_pypower)


    #compare line flows
    p0_pypsa = np.array([line.p0[network.now] for line in network.lines.itervalues()])
    p0_pypower = -results_df['branch']["p0"].values

    np.testing.assert_array_almost_equal(p0_pypsa,p0_pypower)
