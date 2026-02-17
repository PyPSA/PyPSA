# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.clustering.spatial import (
    aggregateoneport,
    busmap_by_hac,
    busmap_by_kmeans,
    get_clustering_from_busmap,
    normed_or_uniform,
)


def test_aggregate_generators(ac_dc_network):
    n = ac_dc_network
    busmap = pd.Series("all", n.c.buses.static.index)
    df, dynamic = aggregateoneport(n, busmap, "Generator")

    assert (
        df.loc["all gas", "p_nom"]
        == n.c.generators.static.query("carrier == 'gas'").p_nom.sum()
    )
    assert (
        df.loc["all wind", "p_nom"]
        == n.c.generators.static.query("carrier == 'wind'").p_nom.sum()
    )

    capacity_norm = normed_or_uniform(
        n.c.generators.static.query("carrier == 'wind'").p_nom
    )
    assert np.allclose(
        dynamic["p_max_pu"]["all wind"],
        (n.c.generators.dynamic.p_max_pu * capacity_norm).sum(axis=1),
    )
    assert np.allclose(
        df.loc["all wind", "marginal_cost"],
        (n.c.generators.static.marginal_cost * capacity_norm).sum(),
    )


def test_aggregate_generators_custom_strategies(ac_dc_network):
    n = ac_dc_network
    n.c.generators.static.loc["Frankfurt Wind", "p_nom_max"] = 100

    busmap = pd.Series("all", n.c.buses.static.index)

    strategies = {"p_max_pu": "max", "p_nom_max": "weighted_min"}
    df, dynamic = aggregateoneport(n, busmap, "Generator", custom_strategies=strategies)

    assert (
        df.loc["all gas", "p_nom"]
        == n.c.generators.static.query("carrier == 'gas'").p_nom.sum()
    )
    assert (
        df.loc["all wind", "p_nom"]
        == n.c.generators.static.query("carrier == 'wind'").p_nom.sum()
    )
    assert (
        df["p_nom_max"]["all wind"]
        == n.c.generators.static.loc["Frankfurt Wind", "p_nom_max"] * 3
    )
    assert np.allclose(
        dynamic["p_max_pu"]["all wind"], n.c.generators.dynamic.p_max_pu.max(axis=1)
    )


def test_aggregate_generators_consent_error(ac_dc_network):
    n = ac_dc_network
    n.add(
        "Generator",
        "Manchester Wind 2",
        bus="Manchester",
        carrier="wind",
        p_nom_extendable=False,
    )

    busmap = pd.Series("all", n.c.buses.static.index)

    with pytest.raises(ValueError):
        df, dynamic = aggregateoneport(n, busmap, "Generator")


def test_aggregate_storage_units(ac_dc_network):
    n = ac_dc_network

    n.add(
        "StorageUnit",
        "Frankfurt Storage",
        bus="Frankfurt",
        p_nom_extendable=True,
        p_nom_max=100,
        p_nom=100,
        marginal_cost=10,
        capital_cost=100,
    )
    n.add(
        "StorageUnit",
        "Manchester Storage",
        bus="Manchester",
        p_nom_extendable=True,
        p_nom_max=200,
        p_nom=200,
        marginal_cost=30,
        capital_cost=50,
    )

    busmap = pd.Series("all", n.c.buses.static.index)
    df, dynamic = aggregateoneport(n, busmap, "StorageUnit")
    capacity_norm = normed_or_uniform(n.c.storage_units.static.p_nom)

    assert df.loc["all", "p_nom"] == n.c.storage_units.static.p_nom.sum()
    assert (
        df.loc["all", "p_nom_extendable"]
        == n.c.storage_units.static.p_nom_extendable.all()
    )
    assert df.loc["all", "p_nom_min"] == n.c.storage_units.static.p_nom_min.sum()
    assert df.loc["all", "p_nom_max"] == n.c.storage_units.static.p_nom_max.sum()
    assert (
        df.loc["all", "marginal_cost"]
        == (n.c.storage_units.static.marginal_cost * capacity_norm).sum()
    )
    assert (
        df.loc["all", "capital_cost"]
        == (n.c.storage_units.static.capital_cost * capacity_norm).sum()
    )


def test_aggregate_storage_units_consent_error(ac_dc_network):
    n = ac_dc_network
    n.add("StorageUnit", "Bremen Storage", bus="Bremen", p_nom_extendable=False)


def prepare_network_for_aggregation(n):
    n.c.lines.static = n.c.lines.static.reindex(
        columns=n.components["Line"]["defaults"].index[1:]
    )
    n.c.lines.static["type"] = np.nan
    n.c.buses.static = n.c.buses.static.reindex(
        columns=n.components["Bus"]["defaults"].index[1:]
    )
    n.c.buses.static["frequency"] = 50


def test_default_clustering_k_means(scipy_network):
    n = scipy_network
    prepare_network_for_aggregation(n)
    weighting = pd.Series(1, n.c.buses.static.index)
    busmap = busmap_by_kmeans(n, bus_weightings=weighting, n_clusters=50)
    C = get_clustering_from_busmap(n, busmap)
    nc = C.n
    assert len(nc.buses) == 50


def test_default_clustering_hac(scipy_network):
    n = scipy_network
    prepare_network_for_aggregation(n)
    busmap = busmap_by_hac(n, n_clusters=50)
    C = get_clustering_from_busmap(n, busmap)
    nc = C.n
    assert len(nc.buses) == 50


def test_cluster_accessor(scipy_network):
    n = scipy_network
    prepare_network_for_aggregation(n)

    weighting = pd.Series(1, n.c.buses.static.index)
    busmap = n.cluster.spatial.busmap_by_kmeans(
        bus_weightings=weighting, n_clusters=50, random_state=42
    )
    buses = n.cluster.spatial.cluster_by_busmap(busmap).buses

    buses_direct = n.cluster.spatial.cluster_by_kmeans(
        bus_weightings=weighting, n_clusters=50, random_state=42
    ).buses
    assert buses.equals(buses_direct)


def test_cluster_accessor_deprecated(scipy_network):
    n = scipy_network
    prepare_network_for_aggregation(n)

    weighting = pd.Series(1, n.c.buses.static.index)
    with pytest.warns(DeprecationWarning, match="n.cluster.spatial"):
        busmap = n.cluster.busmap_by_kmeans(
            bus_weightings=weighting, n_clusters=50, random_state=42
        )
    with pytest.warns(DeprecationWarning, match="n.cluster.spatial"):
        n.cluster.cluster_by_busmap(busmap)
    with pytest.warns(DeprecationWarning, match="n.cluster.spatial"):
        n.cluster.cluster_spatially_by_kmeans(
            bus_weightings=weighting, n_clusters=50, random_state=42
        )


def test_custom_line_groupers(scipy_network):
    n = scipy_network
    random_build_years = [1900, 2000]
    rng = np.random.default_rng()
    n.c.lines.static.loc[:, "build_year"] = rng.choice(
        random_build_years, size=len(n.c.lines.static)
    )
    prepare_network_for_aggregation(n)
    weighting = pd.Series(1, n.c.buses.static.index)
    busmap = busmap_by_kmeans(n, bus_weightings=weighting, n_clusters=20)
    C = get_clustering_from_busmap(n, busmap, custom_line_groupers=["build_year"])
    linemap = C.linemap
    nc = C.n
    assert len(nc.buses) == 20
    assert (n.c.lines.static.groupby(linemap).build_year.nunique() == 1).all()


def test_clustering_multiport_links():
    """Test that clustering correctly remaps all bus ports in multi-port links.

    See https://github.com/PyPSA/PyPSA/issues/1439
    """

    # Create network with multilink
    n = pypsa.Network()
    n.add("Bus", "gas")
    n.add("Bus", "heat")
    n.add("Bus", "power A")
    n.add("Bus", "power B")
    n.add("Bus", "power C")
    n.add(
        "Link",
        "CHP",
        bus0="gas",
        bus1="heat",
        bus2="power A",
        efficiency=0.6,
        efficiency2=0.3,
    )

    # Create busmap that clusters power buses together
    busmap = pd.Series(
        ["gas", "heat", "power", "power", "power"],
        index=["gas", "heat", "power A", "power B", "power C"],
    )

    # Apply clustering
    C = get_clustering_from_busmap(n, busmap)
    n = C.n

    # Assert that bus2 is correctly remapped to "power"
    assert n.c.links.static.loc["CHP", "bus2"] == "power", (
        f"bus2 should be remapped to 'power', got {n.c.links.static.loc['CHP', 'bus2']}"
    )

    # Assert that the old bus "power A" no longer exists
    assert "power A" not in n.c.buses.static.index, "Old bus 'power A' should not exist"

    # Assert all buses referenced by the link exist in the clustered network
    for col in ["bus0", "bus1", "bus2"]:
        bus_name = n.c.links.static.loc["CHP", col]
        assert bus_name in n.c.buses.static.index, (
            f"{col}={bus_name} not found in buses"
        )

    # Assert total number of buses is correct
    assert len(n.c.buses.static) == 3, f"Expected 3 buses, got {len(n.c.buses.static)}"


def test_aggregate_one_ports_no_time_series():
    """Test that empty time series stay empty after clustering."""
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", ["a", "b"])
    n.add("StorageUnit", "su", bus="a", p_nom=100)

    assert n.storage_units_t.p.empty

    busmap = pd.Series("x", n.buses.index)
    C = get_clustering_from_busmap(n, busmap, aggregate_one_ports=["StorageUnit"])

    assert C.n.storage_units_t.p.empty
