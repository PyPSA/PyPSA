# Copyright 2019-2020 Fabian Hofmann (FIAS)
# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3518215.svg
   :target: https://doi.org/10.5281/zenodo.3518215

This rule, as a substitute for :mod:`build_natura_raster`, downloads an already rasterized version (`natura.tiff <https://zenodo.org/record/3518215/files/natura.tiff>`_) of `Natura 2000 <https://en.wikipedia.org/wiki/Natura_2000>`_ natural protection areas to reduce computation times. The file is placed into the ``resources`` sub-directory.

**Relevant Settings**

.. code:: yaml

    enable:
        build_natura_raster:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`toplevel_cf`

**Outputs**

- ``resources/natura.tiff``: Rasterized version of `Natura 2000 <https://en.wikipedia.org/wiki/Natura_2000>`_ natural protection areas to reduce computation times.

.. seealso::
    For details see :mod:`build_natura_raster`.

"""

import logging
logger = logging.getLogger(__name__)

from _helpers import progress_retrieve, configure_logging

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('retrieve_natura_raster')
    configure_logging(snakemake) # TODO Make logging compatible with progressbar (see PR #102)

    url = "https://zenodo.org/record/3518215/files/natura.tiff"

    logger.info(f"Downloading natura raster from '{url}'.")
    progress_retrieve(url, snakemake.output[0])

    logger.info(f"Natura raster available as '{snakemake.output[0]}'.")
