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

import logging, os
from _helpers import progress_retrieve

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    d = './resources'
    if not os.path.exists(d):
        os.makedirs(d)

    progress_retrieve("https://zenodo.org/record/3518215/files/natura.tiff",
                    "resources/natura.tiff")
