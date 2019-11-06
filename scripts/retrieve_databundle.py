## Copyright 2019 Fabian Hofmann (FIAS)
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

import logging, os, tarfile
from _helpers import progress_retrieve

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    if snakemake.config['tutorial']:
        url = "https://zenodo.org/record/3517921/files/pypsa-eur-tutorial-data-bundle.tar.xz"
    else:
        url = "https://zenodo.org/record/3517935/files/pypsa-eur-data-bundle.tar.xz"

    tarball_fn = "./bundle.tar.xz"

    progress_retrieve(url, tarball_fn)

    tarfile.open(tarball_fn).extractall('./data')

    os.remove(tarball_fn)
