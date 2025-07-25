"""Retrieve PyPSA example networks."""

from __future__ import annotations

import logging
from pathlib import Path

from pypsa.networks import Network
from pypsa.version import __version_semver__, __version_semver_tuple__

logger = logging.getLogger(__name__)


def _repo_url(
    master: bool = False, url: str = "https://github.com/PyPSA/PyPSA/raw/"
) -> str:
    if master or __version_semver_tuple__ < (0, 35):  # Feature was added in 0.35.0
        return f"{url}master/"
    return f"{url}v{__version_semver__}/"


def _retrieve_if_not_local(path: str | Path) -> Network:
    if not (Path.cwd() / path).exists():
        path = _repo_url() + str(path)
        Path.cwd()

    return Network(path)


def ac_dc_meshed(
    update: bool = False, from_master: bool = False, remove_link_p_set: bool = False
) -> Network:
    """Load the meshed AC-DC example network.

    Returns
    -------
    pypsa.Network
        AC-DC meshed network.

    Examples
    --------
    >>> n = pypsa.examples.ac_dc_meshed()
    >>> n
    PyPSA Network 'AC-DC-Meshed'
    ----------------------------
    Components:
     - Bus: 9
     - Carrier: 6
     - Generator: 6
     - GlobalConstraint: 1
     - Line: 7
     - Link: 4
     - Load: 6
    Snapshots: 10

    """
    return _retrieve_if_not_local("examples/networks/ac-dc-meshed/ac-dc-meshed.nc")


def storage_hvdc(update: bool = False, from_master: bool = False) -> Network:
    """Load the storage network example of PyPSA.

    Returns
    -------
    pypsa.Network
        Storage network example network.

    Examples
    --------
    >>> n = pypsa.examples.storage_hvdc()
    >>> n
    PyPSA Network 'Storage-HVDC'
    ----------------------------
    Components:
     - Bus: 6
     - Carrier: 3
     - Generator: 12
     - GlobalConstraint: 1
     - Line: 6
     - Link: 2
     - Load: 6
     - StorageUnit: 6
    Snapshots: 12

    """
    return _retrieve_if_not_local("examples/networks/storage-hvdc/storage-hvdc.nc")


def scigrid_de(update: bool = False, from_master: bool = False) -> Network:
    """Load the SciGrid network example of PyPSA.

    Returns
    -------
    pypsa.Network
        SciGrid network example network.

    Examples
    --------
    >>> n = pypsa.examples.scigrid_de()
    >>> n
    PyPSA Network 'SciGrid-DE'
    --------------------------
    Components:
     - Bus: 585
     - Carrier: 16
     - Generator: 1423
     - Line: 852
     - Load: 489
     - StorageUnit: 38
     - Transformer: 96
    Snapshots: 24

    """
    return _retrieve_if_not_local("examples/networks/scigrid-de/scigrid-de.nc")


def model_energy(update: bool = False, from_master: bool = False) -> Network:
    """Load the single-node capacity expansion model in style of model.energy.

    Check out the [model.energy website](https://model.energy/) for more information.


    Returns
    -------
    pypsa.Network
        Single-node capacity expansion model in style of model.energy.

    Examples
    --------
    >>> n = pypsa.examples.model_energy()
    >>> n
    PyPSA Network 'Model-Energy'
    ----------------------------
    Components:
     - Bus: 2
     - Carrier: 9
     - Generator: 3
     - Link: 2
     - Load: 1
     - StorageUnit: 1
     - Store: 1
    Snapshots: 2920

    References
    ----------
    [^1]: See https://model.energy/

    """
    return _retrieve_if_not_local("examples/networks/model-energy/model-energy.nc")


def stochastic_network() -> Network:
    """Load the stochastic network example.

    For details check the example notebook. #TODO new-docs link.

    Returns
    -------
    pypsa.Network
        Stochastic network example network.

    Examples
    --------
    >>> n = pypsa.examples.stochastic_network()
    >>> n
    Stochastic PyPSA Network 'Stochastic-Network'
    ---------------------------------------------
    Components:
     - Bus: 3
     - Carrier: 1
     - Generator: 12
     - Load: 3
    Snapshots: 2920
    Scenarios: 3

    References
    ----------
    [^1]: See https://model.energy/

    """
    n = _retrieve_if_not_local(
        "examples/networks/stochastic-network/stochastic-network.nc"
    )

    n.add("Carrier", "AC", color="indianred")

    return n
