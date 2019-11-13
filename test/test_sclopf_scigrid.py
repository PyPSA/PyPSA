import os
import numpy as np
import pypsa
import sys

solver_name = 'glpk' if sys.platform == 'win32' else 'cbc'


def test_sclopf():
    csv_folder_name = os.path.join(os.path.dirname(__file__), "..", "examples",
                                   "scigrid-de", "scigrid-with-load-gen-trafos")

    network = pypsa.Network(csv_folder_name)

    #test results were generated with GLPK and other solvers may differ

    #There are some infeasibilities without line extensions
    for line_name in ["316","527","602"]:
        network.lines.loc[line_name,"s_nom"] = 1200

    #choose the contingencies
    branch_outages = network.lines.index[:3]

    print("Performing security-constrained linear OPF:")

    network.sclopf(network.snapshots[0],branch_outages=branch_outages,solver_name=solver_name)

    #For the PF, set the P to the optimised P
    network.generators_t.p_set = network.generators_t.p.copy()
    network.generators.loc[:,'p_set_t'] = True
    network.storage_units_t.p_set = network.storage_units_t.p.copy()
    network.storage_units.loc[:,'p_set_t'] = True

    #Check no lines are overloaded with the linear contingency analysis

    p0_test = network.lpf_contingency(network.snapshots[0],branch_outages=branch_outages)

    #check loading as per unit of s_nom in each contingency

    max_loading = abs(p0_test.divide(network.passive_branches().s_nom,axis=0)).describe().loc["max"]


    np.testing.assert_array_almost_equal(max_loading,np.ones((len(max_loading))))


if __name__ == "__main__":
    test_sclopf()
