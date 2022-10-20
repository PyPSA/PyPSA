# -*- coding: utf-8 -*-
import os

import pandas as pd
import pytest
from conftest import SUPPORTED_APIS, optimize
from numpy.testing import assert_array_almost_equal as equal

import pypsa


@pytest.fixture
def target_gen_p():
    target_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "opf-storage-hvdc",
        "opf-storage-data",
        "results",
        "generators-p.csv",
    )
    return pd.read_csv(target_path, index_col=0, parse_dates=True)


@pytest.fixture
def network():
    csv_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "opf-storage-hvdc",
        "opf-storage-data",
    )
    return pypsa.Network(csv_folder)


@pytest.mark.parametrize("api", SUPPORTED_APIS)
def test_lopf(network, target_gen_p, api):
    optimize(network, api)
    equal(network.generators_t.p.reindex_like(target_gen_p), target_gen_p, decimal=2)
