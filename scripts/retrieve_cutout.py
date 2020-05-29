# SPDX-FileCopyrightText: 2019-2020 Fabian Hofmann (FIAS)
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3517949.svg
   :target: https://doi.org/10.5281/zenodo.3517949

Cutouts are spatiotemporal subsets of the European weather data from the `ECMWF ERA5 <https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation>`_ reanalysis dataset and the `CMSAF SARAH-2 <https://wui.cmsaf.eu/safira/action/viewDoiDetails?acronym=SARAH_V002>`_ solar surface radiation dataset for the year 2013 (3.9 GB).
They have been prepared by and are for use with the `atlite <https://github.com/PyPSA/atlite>`_ tool. You can either generate them yourself using the ``build_cutouts`` rule or retrieve them directly from `zenodo <https://doi.org/10.5281/zenodo.3517949>`_ through the rule ``retrieve_cutout`` described here.

.. note::
    To download cutouts yourself from the `ECMWF ERA5 <https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation>`_ you need to `set up the CDS API <https://cds.climate.copernicus.eu/api-how-to>`_.

The :ref:`tutorial` uses smaller `cutouts <https://zenodo.org/record/3518020/files/pypsa-eur-tutorial-cutouts.tar.xz>`_ than required for the full model (19 MB)

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3518020.svg
    :target: https://doi.org/10.5281/zenodo.3518020


**Relevant Settings**

.. code:: yaml

    tutorial:
    enable:
        build_cutout:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`toplevel_cf`

**Outputs**

- ``cutouts/{cutout}``: weather data from either the `ERA5 <https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5>`_   reanalysis weather dataset or `SARAH-2 <https://wui.cmsaf.eu/safira/action/viewProduktSearch>`_ satellite-based historic weather data.

.. seealso::
    For details see :mod:`build_cutout` and read the `atlite documentation <https://atlite.readthedocs.io>`_.

"""

import logging
logger = logging.getLogger(__name__)

from pathlib import Path
import tarfile
from _helpers import progress_retrieve, configure_logging

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('retrieve_cutout')
        rootpath = '..'
    else:
        rootpath = '.'

    configure_logging(snakemake) # TODO Make logging compatible with progressbar (see PR #102)

    if snakemake.config['tutorial']:
        url = "https://zenodo.org/record/3518020/files/pypsa-eur-tutorial-cutouts.tar.xz"
    else:
        url = "https://zenodo.org/record/3517949/files/pypsa-eur-cutouts.tar.xz"

    # Save location
    tarball_fn = Path(f"{rootpath}/cutouts.tar.xz")

    logger.info(f"Downloading cutouts from '{url}'.")
    progress_retrieve(url, tarball_fn)

    logger.info(f"Extracting cutouts.")
    tarfile.open(tarball_fn).extractall(path=rootpath)

    tarball_fn.unlink()

    logger.info(f"Cutouts available in '{Path(tarball_fn.stem).stem}'.")
