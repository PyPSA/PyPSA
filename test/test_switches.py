from __future__ import print_function, division
from __future__ import absolute_import

import logging
logger = logging.getLogger(__name__)

import pypsa
import pandas as pd
import numpy as np
import os


def sort_df(df):
    df = df.reindex(sorted(df.columns), axis=1)
    return df

def sort_series(df):
    df = df.reindex(sorted(df.index))
    return df

def test_pf_results(n1, n2):
    # compare bus angles
    np.testing.assert_array_almost_equal(sort_series(n1.buses_t.v_ang.iloc[0]),
                                         sort_series(n2.buses_t.v_ang.iloc[0]))
    # compare bus voltage magnitudes
    np.testing.assert_array_almost_equal(sort_series(n1.buses_t.v_mag_pu.iloc[0]),
                                         sort_series(n2.buses_t.v_mag_pu.iloc[0]))
    # compare bus active power
    np.testing.assert_array_almost_equal(sort_series(n1.buses_t.p.iloc[0]),
                                         sort_series(n2.buses_t.p.iloc[0]))
    # compare bus active power
    np.testing.assert_array_almost_equal(sort_series(n1.buses_t.q.iloc[0]),
                                         sort_series(n2.buses_t.q.iloc[0]))
    # compare branch flows
    np.testing.assert_array_almost_equal(sort_series(n1.lines_t.p0.iloc[0]),
                                         sort_series(n2.lines_t.p0.iloc[0]))
    np.testing.assert_array_almost_equal(sort_series(n1.lines_t.p1.iloc[0]),
                                         sort_series(n2.lines_t.p1.iloc[0]))
    np.testing.assert_array_almost_equal(sort_series(n1.lines_t.q0.iloc[0]),
                                         sort_series(n2.lines_t.q0.iloc[0]))
    np.testing.assert_array_almost_equal(sort_series(n1.lines_t.q1.iloc[0]),
                                         sort_series(n2.lines_t.q1.iloc[0]))

    np.testing.assert_array_almost_equal(sort_series(n1.transformers_t.p0.iloc[0]),
                                         sort_series(n2.transformers_t.p0.iloc[0]))
    np.testing.assert_array_almost_equal(sort_series(n1.transformers_t.p1.iloc[0]),
                                         sort_series(n2.transformers_t.p1.iloc[0]))
    np.testing.assert_array_almost_equal(sort_series(n1.transformers_t.q0.iloc[0]),
                                         sort_series(n2.transformers_t.q0.iloc[0]))
    np.testing.assert_array_almost_equal(sort_series(n1.transformers_t.q1.iloc[0]),
                                         sort_series(n2.transformers_t.q1.iloc[0]))


def test_switches():
    csv_name = os.path.join(os.path.dirname(__file__), "..", "examples",
                            "switches", "network_with_switches")
    """
    test 1: test the updated import function for networks with switches in case of not
    initialized switches (includes the new initialization of switches) and export it
    with the updated export function. Then reimport it and assert there's no difference
    """
    # read in network with switches that have not been initialized
    n_switches = pypsa.Network(csv_name)
    # export the read in network
    export_name = os.path.join(csv_name, "initialized")
    n_switches.export_to_csv_folder(export_name)
    # read in again
    n_switches_initialized = pypsa.Network(export_name)
    # compare networks:
    pd.testing.assert_frame_equal(sort_df(n_switches.buses),
                                  sort_df(n_switches_initialized.buses),
                                  check_dtype=True)
    pd.testing.assert_frame_equal(sort_df(n_switches.lines),
                                  sort_df(n_switches_initialized.lines),
                                  check_dtype=True)
    pd.testing.assert_frame_equal(sort_df(n_switches.transformers),
                                  sort_df(n_switches_initialized.transformers),
                                  check_dtype=False)  # int64 vs. int32?
    pd.testing.assert_frame_equal(sort_df(n_switches.generators),
                                  sort_df(n_switches_initialized.generators),
                                  check_dtype=False)  # int64 vs. int32?
    pd.testing.assert_frame_equal(sort_df(n_switches.loads),
                                  sort_df(n_switches_initialized.loads),
                                  check_dtype=False)  # int64 vs. int32?
    pd.testing.assert_frame_equal(sort_df(n_switches.switches),
                                  sort_df(n_switches_initialized.switches),
                                  check_dtype=True)
    """
    test 2: test pf against networks without switches
    """
    # this folder has been created by exporting and afterwards deleting all switch-related files
    csv_name_no_switches1 = os.path.join(os.path.dirname(__file__), "..", "examples",
                                         "switches", "network_without_switches1")
    n_no_switches1 = pypsa.Network(csv_name_no_switches1)
    n_switches.pf()
    n_no_switches1.pf()
    test_pf_results(n_switches, n_no_switches1)

    # this folder has been created by opening s2&s4 and closing s3&s5 and then exporting
    # and afterwards deleting all switch-related files:
    csv_name_no_switches2 = os.path.join(os.path.dirname(__file__), "..", "examples",
                                         "switches", "network_without_switches2")
    n_no_switches2 = pypsa.Network(csv_name_no_switches2)
    n_switches.open_switches(["s2", "s4"])
    n_switches.close_switches(["s3", "s5"])
    n_switches.pf()
    n_no_switches2.pf()
    test_pf_results(n_switches, n_no_switches2)

    # test if switching back does change anything
    n_switches.close_switches(["s2", "s4"])
    n_switches.open_switches(["s3", "s5"])
    n_switches.pf()
    test_pf_results(n_switches, n_no_switches1)

if __name__ == "__main__":
    test_switches()
