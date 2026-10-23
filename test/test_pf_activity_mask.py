# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest


@pytest.fixture
def n_full(scipy_network):
    return scipy_network.copy()


@pytest.fixture
def n_filtered(scipy_network):
    n = scipy_network.copy()
    n.c.lines.static.loc["2", "active"] = False
    return n


def test_different_shape_incidence_matrix(n_full, n_filtered):
    k_full = n_full.c.sub_networks.static.obj.iloc[0].incidence_matrix()
    k_filtered = n_filtered.c.sub_networks.static.obj.iloc[0].incidence_matrix()

    assert k_filtered.shape == (k_full.shape[0], k_full.shape[1] - 1)


def test_subnetwork_full_pf(n_full):
    n_full.c.sub_networks.static.obj.iloc[0].pf(n_full.snapshots[:3])


def test_subnetwork_filtered_pf(n_filtered):
    n_filtered.c.sub_networks.static.obj.iloc[0].pf(n_filtered.snapshots[:3])
    n = n_filtered
    assert n.c.lines.dynamic.p0.loc[:, ~n.c.lines.static.active].eq(0).all().all()
