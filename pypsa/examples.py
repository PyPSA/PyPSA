#!/usr/bin/env python3
"""
This module contains functions for retrieving/loading example networks provided
by the PyPSA project.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.error import HTTPError
from urllib.request import urlretrieve

from pypsa.io import _data_dir
from pypsa.networks import Network

if TYPE_CHECKING:
    from pypsa import Network
logger = logging.getLogger(__name__)


def _decrement_version(version: str) -> str:
    x, y, z = map(int, version.split("."))
    if z > 0:
        z -= 1
    elif y > 0:
        y -= 1
        z = 25  # TODO: This is a hack right now
    elif x > 0:
        x -= 1
        y = z = 25  # TODO: This is a hack right now
    return f"{x}.{y}.{z}"


def _repo_url(master: bool = False) -> str:
    url = "https://github.com/PyPSA/PyPSA/raw/"
    if master:
        return f"{url}master/"
    from pypsa import release_version  # avoid cyclic imports

    assert release_version is not None, "release_version is None"
    # If the release version is not found, use the latest version, since this is
    # because we are in a dev branch which has not been released yet.
    version_with_data = release_version
    while True:
        try:
            urlretrieve(f"{url}v{version_with_data}/".replace("raw", "releases/tag"))
            break
        except HTTPError:
            version_with_data = _decrement_version(version_with_data)

    return f"{url}v{version_with_data}/"


def _retrieve_if_not_local(
    name: str, repofile: str, update: bool = False, from_master: bool = False
) -> str:
    path = (_data_dir / name).with_suffix(".nc")

    if not path.exists() or update:
        url = _repo_url(from_master) + repofile
        logger.info(f"Retrieving network data from {url}")
        urlretrieve(url, path)

    return str(path)


def _sanitize_ac_dc_meshed(n: Network, remove_link_p_set: bool = True) -> None:
    # TODO: make this function obsolete by adjusting the input files
    n.buses["country"] = ["UK", "UK", "UK", "UK", "DE", "DE", "DE", "NO", "NO"]
    n.carriers["color"] = ["red", "blue", "green"]
    n.loads["carrier"] = "load"
    n.lines["carrier"] = "AC"
    n.links["carrier"] = "DC"
    n.add("Carrier", "load", color="black")
    n.add("Carrier", "AC", color="orange")
    n.add("Carrier", "DC", color="purple")
    if remove_link_p_set:
        n.links_t.p_set.drop(columns=n.links_t.p_set.columns, inplace=True)


def ac_dc_meshed(
    update: bool = False, from_master: bool = False, remove_link_p_set: bool = True
) -> Network:
    """
    Load the meshed AC-DC network example of pypsa stored in the PyPSA
    repository.

    Parameters
    ----------
    update : bool, optional
        Whether to update the locally stored network data. The default is False.
    from_master : bool, optional
        Whether to retrieve from the master branch of the pypsa repository.

    Returns
    -------
    pypsa.Network
    """
    name = "ac-dc-meshed"
    repofile = "examples/ac-dc-meshed/ac-dc-data.nc"
    path = _retrieve_if_not_local(
        name, repofile, update=update, from_master=from_master
    )
    n = Network(path)
    _sanitize_ac_dc_meshed(n, remove_link_p_set=remove_link_p_set)
    return n


def storage_hvdc(update: bool = False, from_master: bool = False) -> Network:
    """
    Load the storage network example of pypsa stored in the PyPSA repository.

    Parameters
    ----------
    update : bool, optional
        Whether to update the locally stored network data. The default is False.

    Returns
    -------
    pypsa.Network
    """
    name = "storage-hvdc"
    repofile = "examples/opf-storage-hvdc/storage-hvdc.nc"
    path = _retrieve_if_not_local(
        name, repofile, update=update, from_master=from_master
    )
    return Network(path)


def scigrid_de(update: bool = False, from_master: bool = False) -> Network:
    """
    Load the SciGrid network example of pypsa stored in the PyPSA repository.

    Parameters
    ----------
    update : bool, optional
        Whether to update the locally stored network data. The default is False.

    Returns
    -------
    pypsa.Network
    """
    name = "scigrid-de"
    repofile = "examples/scigrid-de/scigrid-with-load-gen-trafos.nc"
    path = _retrieve_if_not_local(
        name, repofile, update=update, from_master=from_master
    )
    n = Network(path)
    carriers = list(
        {
            carrier
            for c in n.iterate_components()
            if "carrier" in c.static
            for carrier in c.static.carrier.unique()
        }
    )
    n.add("Carrier", carriers)
    return n
