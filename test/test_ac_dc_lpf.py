import pypsa

import numpy as np

import os


def test_lpf():
    csv_folder_name = os.path.join(os.path.dirname(__file__), "..", "examples", "ac-dc-meshed", "ac-dc-data")

    network = pypsa.Network(csv_folder_name)

    results_folder_name = os.path.join(csv_folder_name, "results-lpf")

    network_r = pypsa.Network(results_folder_name)

    for snapshot in network.snapshots[:2]:
        network.lpf(snapshot)

    np.testing.assert_array_almost_equal(network.generators_t.p[network.generators.index].iloc[:2],network_r.generators_t.p[network.generators.index].iloc[:2])
    np.testing.assert_array_almost_equal(network.lines_t.p0[network.lines.index].iloc[:2],network_r.lines_t.p0[network.lines.index].iloc[:2])
    np.testing.assert_array_almost_equal(network.links_t.p0[network.links.index].iloc[:2],network_r.links_t.p0[network.links.index].iloc[:2])


    network.lpf(snapshots=network.snapshots)

    np.testing.assert_array_almost_equal(network.generators_t.p[network.generators.index],network_r.generators_t.p[network.generators.index])
    np.testing.assert_array_almost_equal(network.lines_t.p0[network.lines.index],network_r.lines_t.p0[network.lines.index])
    np.testing.assert_array_almost_equal(network.links_t.p0[network.links.index],network_r.links_t.p0[network.links.index])


if __name__ == "__main__":
    test_lpf()
