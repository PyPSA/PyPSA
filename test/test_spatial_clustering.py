#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 18:11:09 2022.

@author: fabian
"""

import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.clustering.spatial import (
    aggregategenerators,
    aggregateoneport,
    busmap_by_hac,
    busmap_by_kmeans,
    get_clustering_from_busmap,
    normed_or_uniform,
)


def test_aggregategenerators(ac_dc_network):
    n = ac_dc_network
    busmap = pd.Series("all", n.buses.index)
    df, pnl = aggregateoneport(n, busmap, "Generator")

    assert (
        df.loc["all gas", "p_nom"] == n.generators.query("carrier == 'gas'").p_nom.sum()
    )
    assert (
        df.loc["all wind", "p_nom"]
        == n.generators.query("carrier == 'wind'").p_nom.sum()
    )

    capacity_norm = normed_or_uniform(n.generators.query("carrier == 'wind'").p_nom)
    assert np.allclose(
        pnl["p_max_pu"]["all wind"],
        (n.generators_t.p_max_pu * capacity_norm).sum(axis=1),
    )
    assert np.allclose(
        df.loc["all wind", "marginal_cost"],
        (n.generators.marginal_cost * capacity_norm).sum(),
    )


def test_aggregategenerators_custom_strategies(ac_dc_network):
    n = ac_dc_network
    n.generators.loc["Frankfurt Wind", "p_nom_max"] = 100

    busmap = pd.Series("all", n.buses.index)

    strategies = {"p_max_pu": "max", "p_nom_max": "weighted_min"}
    df, pnl = aggregategenerators(n, busmap, custom_strategies=strategies)

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
    assert np.allclose(pnl["p_max_pu"]["all wind"], n.generators_t.p_max_pu.max(axis=1))


def test_aggregategenerators_consent_error(ac_dc_network):
    n = ac_dc_network
    n.add(
        "Generator",
        "Manchester Wind 2",
        bus="Manchester",
        carrier="wind",
        p_nom_extendable=False,
    )

    busmap = pd.Series("all", n.buses.index)

    with pytest.raises(AssertionError):
        df, pnl = aggregategenerators(n, busmap)


def test_aggregateoneport(ac_dc_network):
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
    df, pnl = aggregateoneport(n, busmap, "StorageUnit")
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


def test_aggregateoneport_consent_error(ac_dc_network):
    n = ac_dc_network
    n.add("StorageUnit", "Bremen Storage", bus="Bremen", p_nom_extendable=False)

    busmap = pd.Series("all", n.buses.index)
    with pytest.raises(AssertionError):
        df, pnl = aggregateoneport(n, busmap, "StorageUnit")


def test_default_clustering_k_means(scipy_network):
    n = scipy_network
    # delete the 'type' specifications to make this example easier
    n.lines["type"] = np.nan
    weighting = pd.Series(1, n.buses.index)
    busmap = busmap_by_kmeans(n, bus_weightings=weighting, n_clusters=50)
    C = get_clustering_from_busmap(n, busmap)
    nc = C.network
    assert len(nc.buses) == 50


def test_default_clustering_hac(scipy_network):
    n = scipy_network
    # delete the 'type' specifications to make this example easier
    n.lines["type"] = np.nan
    busmap = busmap_by_hac(n, n_clusters=50)
    C = get_clustering_from_busmap(n, busmap)
    nc = C.network
    assert len(nc.buses) == 50
