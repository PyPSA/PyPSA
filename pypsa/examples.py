#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:03:35 2021

@author: fabian
"""

import logging
import urllib
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
_data_dir.mkdir(exist_ok=True)


# TODO: replace ending by 'master/' as soon as merging into master
repo_url = "https://github.com/PyPSA/PyPSA/raw/example-module/"


def _retrieve_if_not_local(name, repofile, update=False):
    path = (_data_dir / name).with_suffix(".nc")

    if not path.exists() or update:
        url = f"{repo_url}/{repofile}"
        logger.info(f"Retrieving network data from {url}")
        urllib.request.urlretrieve(url, path)

    return str(path)


def ac_dc_meshed(update=False):
    name = "ac-dc-meshed"
    repofile = "examples/ac-dc-meshed/ac-dc-data.nc"
    path = _retrieve_if_not_local(name, repofile, update=update)
    return Network(path)


def storage_hvdc(update=False):
    name = "storage-hvdc"
    repofile = "examples/opf-storage-hvdc/opf-storage-hvdc-data.nc"
    path = _retrieve_if_not_local(name, repofile, update=update)
    return Network(path)


def scigrid_de(update=False):
    name = "scigrid-de"
    repofile = "examples/scigrid-de/scigrid-with-load-gen-trafos.nc"
    path = _retrieve_if_not_local(name, repofile, update=update)
    return Network(path)

