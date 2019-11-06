## Copyright 2019 Fabian Hofmann (FIAS)

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

import logging, os, tarfile
from _helpers import progress_retrieve

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    if snakemake.config['tutorial']:
        url = "https://zenodo.org/record/3518020/files/pypsa-eur-tutorial-cutouts.tar.xz"
    else:
        url = "https://zenodo.org/record/3517949/files/pypsa-eur-cutouts.tar.xz"

    tarball_fn = "./cutouts.tar.xz"

    progress_retrieve(url, tarball_fn)

    tarfile.open(tarball_fn).extractall()

    os.remove(tarball_fn)
