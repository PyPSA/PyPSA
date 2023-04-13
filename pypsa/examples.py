#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains functions for retrieving/loading example networks provided
by the PyPSA project.
"""

__author__ = (
    "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
)
__copyright__ = (
    "Copyright 2021-2023 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
    "MIT License"
)

import logging
from urllib.request import urlretrieve

import pandas as pd

from pypsa.components import Network
from pypsa.io import _data_dir

logger = logging.getLogger(__name__)


def _repo_url(master=False):
    url = "https://github.com/PyPSA/PyPSA/raw/"
    if master:
        return url + "master/"
    from pypsa import __version__  # avoid cyclic imports

    return url + f"v{__version__}/"


def _retrieve_if_not_local(name, repofile, update=False, from_master=False):
    path = (_data_dir / name).with_suffix(".nc")

    if not path.exists() or update:
        url = _repo_url(from_master) + repofile
        logger.info(f"Retrieving network data from {url}")
        urlretrieve(url, path)

    return str(path)


def ac_dc_meshed(update=False, from_master=False, remove_link_p_set=True):
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
    if remove_link_p_set:
        n.links_t.p_set = pd.DataFrame(index=n.snapshots)
    return n


def storage_hvdc(update=False, from_master=False):
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


def scigrid_de(update=False, from_master=False):
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
    return Network(path)
