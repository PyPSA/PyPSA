## Copyright 2019 Fabian Hofmann (FIAS)

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

from pathlib import Path
from _helpers import progress_retrieve, configure_logging

if __name__ == "__main__":

    configure_logging(snakemake) # TODO Make logging compatible with progressbar (see PR #102)

    # Save location, ensure folder existence
    to_fn = Path("resources/natura.tiff")
    to_fn.parent.mkdir(parents=True, exist_ok=True)

    url = "https://zenodo.org/record/3518215/files/natura.tiff"

    logger.info(f"Downloading natura raster from '{url}'.")
    progress_retrieve(url, to_fn)
    
    logger.info(f"Natura raster available as '{to_fn}'.")