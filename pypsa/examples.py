#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains functions for retrieving/loading example networks provided
by the PyPSA project.
"""

import logging
from urllib.request import urlretrieve as _retrieve
import os

from pathlib import Path
from .components import Network

logger = logging.getLogger(__name__)


# for the writable data directory follow the XDG guidelines
# https://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html
_writable_dir = os.path.join(os.path.expanduser("~"), ".local", "share")
_data_dir = os.path.join(
    os.environ.get("XDG_DATA_HOME", os.environ.get("APPDATA", _writable_dir)),
    "pypsa-examples",
)
_data_dir = Path(_data_dir)
try:
    _data_dir.mkdir(exist_ok=True)
except FileNotFoundError:
    os.makedirs(_data_dir)


def _repo_url(master=False):
    url = "https://github.com/PyPSA/PyPSA/raw/"
    if master:
        return url + "master/"
    else:
        from pypsa import __version__  # avoid cyclic imports

        return url + f"v{__version__}/"


def _retrieve_if_not_local(name, repofile, update=False, from_master=False):
    path = (_data_dir / name).with_suffix(".nc")

    if not path.exists() or update:
        url = _repo_url(from_master) + repofile
        logger.info(f"Retrieving network data from {url}")
        _retrieve(url, path)

    return str(path)


def ac_dc_meshed(update=False, from_master=False):
    """
    Load the meshed AC-DC network example of pypsa stored in the PyPSA repository.

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
    return Network(path)


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
