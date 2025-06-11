import warnings

import pandas as pd
import pytest

import pypsa
from pypsa.common import expand_series
from pypsa.descriptors import (
    additional_linkports,
    get_extendable_i,
    get_non_extendable_i,
    get_switchable_as_dense,
    get_switchable_as_iter,
)
from pypsa.network.power_flow import allocate_series_dataframes


@pytest.fixture
def network():
    n = pypsa.Network()
    n.snapshots = pd.date_range("2019-01-01", "2019-01-02", freq="h")
    return n


def test_get_switchable_as_dense(network):
    n = network
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom=100)

    for attr, val in [("p_max_pu", 1.0), ("p_nom", 100)]:
        df = get_switchable_as_dense(n, "Generator", attr)
        assert isinstance(df, pd.DataFrame)
        assert df.index.equals(n.snapshots)
        assert df.columns.equals(pd.Index(["gen0"]))
        assert (df == val).all().all()


def test_get_switchable_as_iter(network):
    n = network
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom=100)

    iter_df = get_switchable_as_iter(n, "Generator", "p_max_pu", n.snapshots)
    df = pd.concat(iter_df, axis=1).T
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(n.snapshots)
    assert (df["gen0"] == 1.0).all()


def test_allocate_series_dataframes(network):
    n = network
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0")
    n.add("Load", "load0", bus="bus0")

    allocate_series_dataframes(n, {"Generator": ["p"], "Load": ["p"]})

    assert "p" in n.generators_t
    assert "p" in n.loads_t
    assert n.generators_t.p.shape == (len(n.snapshots), 1)
    assert n.loads_t.p.shape == (len(n.snapshots), 1)


def test_get_extendable_i(network):
    n = network
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom_extendable=True)
    n.add("Generator", "gen1", bus="bus0", p_nom_extendable=False)

    ext_i = get_extendable_i(n, "Generator")
    assert len(ext_i) == 1
    assert "gen0" in ext_i


def test_get_non_extendable_i(network):
    n = network
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom_extendable=True)
    n.add("Generator", "gen1", bus="bus0", p_nom_extendable=False)

    nonext_i = get_non_extendable_i(n, "Generator")
    assert len(nonext_i) == 1
    assert "gen1" in nonext_i


def test_expand_series():
    s = pd.Series([1, 2, 3])
    cols = ["a", "b", "c"]
    df = expand_series(s, cols)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == cols
    assert (df["a"] == df["b"]).all()
    assert (df["b"] == df["c"]).all()


def test_additional_linkports():
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")
    n.add("Link", "link0", bus0="bus0", bus1="bus1", bus2="bus2")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        ports = additional_linkports(n, n.links.columns)
    assert ports == ["2"]
    assert ports == n.c.links.additional_ports
