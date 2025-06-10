"""
Created on Mon Jan 31 18:11:09 2022.

@author: fabian
"""

import numpy as np
import pandas as pd
import pytest

from pypsa.clustering.spatial import (
    aggregateoneport,
    busmap_by_hac,
    busmap_by_kmeans,
    get_clustering_from_busmap,
    normed_or_uniform,
)


def test_aggregate_generators(ac_dc_network):
    n = ac_dc_network
    busmap = pd.Series("all", n.buses.index)
    df, dynamic = aggregateoneport(n, busmap, "Generator")

    assert (
        df.loc["all gas", "p_nom"] == n.generators.query("carrier == 'gas'").p_nom.sum()
    )
    assert (
        df.loc["all wind", "p_nom"]
        == n.generators.query("carrier == 'wind'").p_nom.sum()
    )

    capacity_norm = normed_or_uniform(n.generators.query("carrier == 'wind'").p_nom)
    assert np.allclose(
        dynamic["p_max_pu"]["all wind"],
        (n.generators_t.p_max_pu * capacity_norm).sum(axis=1),
    )
    assert np.allclose(
        df.loc["all wind", "marginal_cost"],
        (n.generators.marginal_cost * capacity_norm).sum(),
    )


def test_aggregate_generators_custom_strategies(ac_dc_network):
    n = ac_dc_network
    n.generators.loc["Frankfurt Wind", "p_nom_max"] = 100

    busmap = pd.Series("all", n.buses.index)

    strategies = {"p_max_pu": "max", "p_nom_max": "weighted_min"}
    df, dynamic = aggregateoneport(n, busmap, "Generator", custom_strategies=strategies)

    assert (
        df.loc["all gas", "p_nom"] == n.generators.query("carrier == 'gas'").p_nom.sum()
    )
    assert (
        df.loc["all wind", "p_nom"]
        == n.generators.query("carrier == 'wind'").p_nom.sum()
    )
    assert (
        df["p_nom_max"]["all wind"]
        == n.generators.loc["Frankfurt Wind", "p_nom_max"] * 3
    )
    assert np.allclose(
        dynamic["p_max_pu"]["all wind"], n.generators_t.p_max_pu.max(axis=1)
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

    busmap = pd.Series("all", n.buses.index)

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

    busmap = pd.Series("all", n.buses.index)
    df, dynamic = aggregateoneport(n, busmap, "StorageUnit")
    capacity_norm = normed_or_uniform(n.storage_units.p_nom)

    assert df.loc["all", "p_nom"] == n.storage_units.p_nom.sum()
    assert df.loc["all", "p_nom_extendable"] == n.storage_units.p_nom_extendable.all()
    assert df.loc["all", "p_nom_min"] == n.storage_units.p_nom_min.sum()
    assert df.loc["all", "p_nom_max"] == n.storage_units.p_nom_max.sum()
    assert (
        df.loc["all", "marginal_cost"]
        == (n.storage_units.marginal_cost * capacity_norm).sum()
    )
    assert (
        df.loc["all", "capital_cost"]
        == (n.storage_units.capital_cost * capacity_norm).sum()
    )


def test_aggregate_storage_units_consent_error(ac_dc_network):
    n = ac_dc_network
    n.add("StorageUnit", "Bremen Storage", bus="Bremen", p_nom_extendable=False)


def prepare_network_for_aggregation(n):
    n.lines = n.lines.reindex(columns=n.components["Line"]["attrs"].index[1:])
    n.lines["type"] = np.nan
    n.buses = n.buses.reindex(columns=n.components["Bus"]["attrs"].index[1:])
    n.buses["frequency"] = 50


def test_default_clustering_k_means(scipy_network):
    n = scipy_network
    prepare_network_for_aggregation(n)
    weighting = pd.Series(1, n.buses.index)
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

    weighting = pd.Series(1, n.buses.index)
    busmap = n.cluster.busmap_by_kmeans(
        bus_weightings=weighting, n_clusters=50, random_state=42
    )
    buses = n.cluster.cluster_by_busmap(busmap).buses

    buses_direct = n.cluster.cluster_spatially_by_kmeans(
        bus_weightings=weighting, n_clusters=50, random_state=42
    ).buses
    assert buses.equals(buses_direct)


def test_custom_line_groupers(scipy_network):
    n = scipy_network
    random_build_years = [1900, 2000]
    rng = np.random.default_rng()
    n.lines.loc[:, "build_year"] = rng.choice(random_build_years, size=len(n.lines))
    prepare_network_for_aggregation(n)
    weighting = pd.Series(1, n.buses.index)
    busmap = busmap_by_kmeans(n, bus_weightings=weighting, n_clusters=20)
    C = get_clustering_from_busmap(n, busmap, custom_line_groupers=["build_year"])
    linemap = C.linemap
    nc = C.n
    assert len(nc.buses) == 20
    assert (n.lines.groupby(linemap).build_year.nunique() == 1).all()
