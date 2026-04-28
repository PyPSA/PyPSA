# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import warnings

import pandas as pd
import pytest

import pypsa
from pypsa.common import expand_series
from pypsa.descriptors import (
    get_bounds_pu,
    get_extendable_i,
    get_non_extendable_i,
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
        df = n.get_switchable_as_dense("Generator", attr)
        assert isinstance(df, pd.DataFrame)
        assert df.index.equals(n.snapshots)
        assert df.columns.equals(pd.Index(["gen0"]))
        assert (df == val).all().all()


def test_get_switchable_as_iter(network):
    n = network
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom=100)

    iter_df = n.get_switchable_as_iter("Generator", "p_max_pu", n.snapshots)
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

    assert "p" in n.c.generators.dynamic
    assert "p" in n.c.loads.dynamic
    assert n.c.generators.dynamic.p.shape == (len(n.snapshots), 1)
    assert n.c.loads.dynamic.p.shape == (len(n.snapshots), 1)


def test_get_extendable_i(network):
    n = network
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom_extendable=True)
    n.add("Generator", "gen1", bus="bus0", p_nom_extendable=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        ext_i = get_extendable_i(n, "Generator")
    assert len(ext_i) == 1
    assert "gen0" in ext_i


def test_get_non_extendable_i(network):
    n = network
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom_extendable=True)
    n.add("Generator", "gen1", bus="bus0", p_nom_extendable=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
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

    # Test index name preservation with MultiIndex (regression for #1580)
    mi = pd.MultiIndex.from_product(
        [[2020, 2030], [1, 2]], names=["period", "timestep"]
    )
    mi.name = "snapshot"
    s_named = pd.Series([1.0] * 4, index=mi)
    df_named = expand_series(s_named, ["x", "y"])
    assert df_named.index.name == "snapshot"


def test_additional_ports():
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")
    n.add("Link", "link0", bus0="bus0", bus1="bus1", bus2="bus2")

    ports = n.c.links.additional_ports
    assert ports == ["2"]


def test_additional_ports_process():
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")
    n.add("Bus", "bus3")
    n.add(
        "Process",
        "proc0",
        bus0="bus0",
        bus1="bus1",
        bus2="bus2",
        bus3="bus3",
        rate0=-1,
        rate1=0.5,
        rate2=0.3,
        rate3=0.1,
        p_nom=10,
    )

    ports = n.c.processes.additional_ports
    assert sorted(ports) == ["2", "3"]


def test_update_ports_component_attrs():
    n = pypsa.Network()

    process_defaults = n.components.processes["defaults"]
    assert "bus2" not in process_defaults.index
    assert "rate2" not in process_defaults.index
    assert "p2" not in process_defaults.index
    assert "delay2" not in process_defaults.index
    assert "cyclic_delay2" not in process_defaults.index

    link_defaults = n.components.links["defaults"]
    assert "bus2" not in link_defaults.index
    assert "efficiency2" not in link_defaults.index
    assert "p2" not in link_defaults.index
    assert "delay2" not in link_defaults.index
    assert "cyclic_delay2" not in link_defaults.index

    n.set_snapshots([0])
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")
    n.add(
        "Process",
        "proc0",
        bus0="bus0",
        bus1="bus1",
        bus2="bus2",
        rate0=-1,
        rate1=0.5,
        rate2=0.3,
        p_nom=10,
    )

    n.add(
        "Link",
        "proc0",
        bus0="bus0",
        bus1="bus1",
        bus2="bus2",
        efficiency=0.5,
        efficiency2=0.3,
        p_nom=10,
    )

    for c in n.components.filter(branch=True, controllable=True):
        c._update_port_attrs()

    process_defaults = n.components["Process"]["defaults"]
    assert "bus2" in process_defaults.index
    assert "rate2" in process_defaults.index
    assert "p2" in process_defaults.index
    assert "delay2" in process_defaults.index
    assert "cyclic_delay2" in process_defaults.index

    link_defaults = n.components["Link"]["defaults"]
    assert "bus2" in link_defaults.index
    assert "efficiency2" in link_defaults.index
    assert "p2" in link_defaults.index
    assert "delay2" in link_defaults.index
    assert "cyclic_delay2" in link_defaults.index


def test_update_ports_column_ordering():
    """Test that bus2, bus3, efficiency2, efficiency3 are inserted after bus1, efficiency."""
    n = pypsa.Network()
    n.set_snapshots([0])
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")
    n.add("Bus", "bus3")
    n.add(
        "Link",
        "link0",
        bus0="bus0",
        bus1="bus1",
        bus2="bus2",
        bus3="bus3",
        efficiency2=0.5,
        efficiency3=0.3,
        p_nom=10,
    )

    defaults = n.components["Link"]["defaults"]
    idx = list(defaults.index)

    # bus2 and bus3 should come right after bus1
    bus1_pos = idx.index("bus1")
    assert idx[bus1_pos + 1] == "bus2", (
        f"Expected bus2 after bus1, got {idx[bus1_pos + 1]}"
    )
    assert idx[bus1_pos + 2] == "bus3", (
        f"Expected bus3 after bus2, got {idx[bus1_pos + 2]}"
    )

    # efficiency2 and efficiency3 should come right after efficiency
    eff_pos = idx.index("efficiency")
    assert idx[eff_pos + 1] == "efficiency2", (
        f"Expected efficiency2 after efficiency, got {idx[eff_pos + 1]}"
    )
    assert idx[eff_pos + 2] == "efficiency3", (
        f"Expected efficiency3 after efficiency2, got {idx[eff_pos + 2]}"
    )

    # p2 and p3 should come right after p1
    p1_pos = idx.index("p1")
    assert idx[p1_pos + 1] == "p2", f"Expected p2 after p1, got {idx[p1_pos + 1]}"
    assert idx[p1_pos + 2] == "p3", f"Expected p3 after p2, got {idx[p1_pos + 2]}"


def test_update_ports_column_ordering_process():
    """Test that bus2, rate2 are inserted after bus1, rate1 for Process."""
    n = pypsa.Network()
    n.set_snapshots([0])
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")
    n.add(
        "Process",
        "proc0",
        bus0="bus0",
        bus1="bus1",
        bus2="bus2",
        rate0=-1,
        rate1=0.5,
        rate2=0.3,
        p_nom=10,
    )

    defaults = n.components["Process"]["defaults"]
    idx = list(defaults.index)

    # bus2 should come right after bus1
    bus1_pos = idx.index("bus1")
    assert idx[bus1_pos + 1] == "bus2", (
        f"Expected bus2 after bus1, got {idx[bus1_pos + 1]}"
    )

    # rate2 should come right after rate1
    rate1_pos = idx.index("rate1")
    assert idx[rate1_pos + 1] == "rate2", (
        f"Expected rate2 after rate1, got {idx[rate1_pos + 1]}"
    )

    # p2 should come right after p1
    p1_pos = idx.index("p1")
    assert idx[p1_pos + 1] == "p2", f"Expected p2 after p1, got {idx[p1_pos + 1]}"


def test_additional_linkports_isolated_between_networks():
    n1 = pypsa.Network()
    n1.add("Bus", "bus0")
    n1.add("Bus", "bus1")
    n1.add("Bus", "bus2")
    n1.add("Bus", "bus3")
    n1.add("Link", "link0", bus0="bus0", bus1="bus1", bus2="bus2", bus3="bus3")

    n2 = pypsa.Network()
    n2.add("Bus", "bus0")
    n2.add("Bus", "bus1")
    n2.add("Bus", "bus2")
    n2.add("Link", "link0", bus0="bus0", bus1="bus1", bus2="bus2")

    n3 = pypsa.Network()
    n3.add("Bus", "bus0")
    n3.add("Bus", "bus1")
    n3.add("Bus", "bus2")
    n3.add("Bus", "bus3")
    n3.add("Link", "link0", bus0="bus0", bus1="bus1", bus2="bus2", bus3="bus3")

    ports = n2.c.links.additional_ports

    assert ports == ["2"]
    assert "bus3" not in n2.c.links.static.columns


def test_get_bounds_pu():
    n = pypsa.Network()
    n.snapshots = pd.date_range("2019-01-01", "2019-01-02", freq="h")
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom=100, p_min_pu=0.2, p_max_pu=0.8)
    n.add("Generator", "gen1", bus="bus0", p_nom=200, p_min_pu=0.1, p_max_pu=0.9)

    # Test deprecated function vs component method consistency
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        deprecated_result = get_bounds_pu(n, "Generator", n.snapshots, attr="p")

    # Get bounds from component method directly
    component_bounds = n.components["Generator"].get_bounds_pu(attr="p")
    component_result = (
        component_bounds[0].sel(snapshot=n.snapshots).to_dataframe().unstack(level=0),
        component_bounds[1].sel(snapshot=n.snapshots).to_dataframe().unstack(level=0),
    )

    # Test that both methods return the same type and structure
    assert isinstance(deprecated_result, tuple)
    assert len(deprecated_result) == 2
    assert isinstance(deprecated_result[0], pd.DataFrame)
    assert isinstance(deprecated_result[1], pd.DataFrame)

    # Check that the values are the same - this is the key test
    pd.testing.assert_frame_equal(deprecated_result[0], component_result[0])
    pd.testing.assert_frame_equal(deprecated_result[1], component_result[1])

    # Basic sanity checks on data shape and values
    assert deprecated_result[0].shape == (2, 25)  # 2 generators, 25 snapshots
    assert deprecated_result[1].shape == (2, 25)

    # Check some values are correct (not all, just to verify functionality)
    assert 0.2 in deprecated_result[0].values  # gen0 min_pu
    assert 0.8 in deprecated_result[1].values  # gen0 max_pu
    assert 0.1 in deprecated_result[0].values  # gen1 min_pu
    assert 0.9 in deprecated_result[1].values  # gen1 max_pu
