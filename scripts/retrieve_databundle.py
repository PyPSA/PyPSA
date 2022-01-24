# Copyright 2019-2020 Fabian Hofmann (FIAS)
# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3517935.svg
   :target: https://doi.org/10.5281/zenodo.3517935

The data bundle (1.4 GB) contains common GIS datasets like NUTS3 shapes, EEZ shapes, CORINE Landcover, Natura 2000 and also electricity specific summary statistics like historic per country yearly totals of hydro generation, GDP and POP on NUTS3 levels and per-country load time-series.

This rule downloads the data bundle from `zenodo <https://doi.org/10.5281/zenodo.3517935>`_ and extracts it in the ``data`` sub-directory, such that all files of the bundle are stored in the ``data/bundle`` subdirectory.

The :ref:`tutorial` uses a smaller `data bundle <https://zenodo.org/record/3517921/files/pypsa-eur-tutorial-data-bundle.tar.xz>`_ than required for the full model (19 MB)

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3517921.svg
    :target: https://doi.org/10.5281/zenodo.3517921

**Relevant Settings**

.. code:: yaml

    tutorial:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`toplevel_cf`

**Outputs**

- ``cutouts/bundle``: input data collected from various sources

"""

import logging
from _helpers import progress_retrieve, configure_logging

import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('retrieve_databundle')
        rootpath = '..'
    else:
        rootpath = '.'
    configure_logging(snakemake) # TODO Make logging compatible with progressbar (see PR #102)

    if snakemake.config['tutorial']:
        url = "https://zenodo.org/record/3517921/files/pypsa-eur-tutorial-data-bundle.tar.xz"
    else:
        url = "https://zenodo.org/record/3517935/files/pypsa-eur-data-bundle.tar.xz"

    # Save locations
    tarball_fn = Path(f"{rootpath}/bundle.tar.xz")
    to_fn = Path(f"{rootpath}/data")

    logger.info(f"Downloading databundle from '{url}'.")
    progress_retrieve(url, tarball_fn)

    logger.info(f"Extracting databundle.")
    tarfile.open(tarball_fn).extractall(to_fn)

    tarball_fn.unlink()

    logger.info(f"Databundle available in '{to_fn}'.")
