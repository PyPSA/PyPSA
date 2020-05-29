# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Rasters the vector data of the `Natura 2000 <https://en.wikipedia.org/wiki/Natura_2000>`_ natural protection areas onto all cutout regions.

Relevant Settings
-----------------

.. code:: yaml

    renewable:
        {technology}:
            cutout:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`renewable_cf`

Inputs
------

- ``data/bundle/natura/Natura2000_end2015.shp``: `Natura 2000 <https://en.wikipedia.org/wiki/Natura_2000>`_ natural protection areas.

    .. image:: ../img/natura.png
        :scale: 33 %

Outputs
-------

- ``resources/natura.tiff``: Rasterized version of `Natura 2000 <https://en.wikipedia.org/wiki/Natura_2000>`_ natural protection areas to reduce computation times.

    .. image:: ../img/natura.png
        :scale: 33 %

Description
-----------

"""

import logging
from _helpers import configure_logging
import atlite
import geokit as gk
from pathlib import Path

logger = logging.getLogger(__name__)

def determine_cutout_xXyY(cutout_name):
    cutout = atlite.Cutout(cutout_name, cutout_dir=cutout_dir)
    x, X, y, Y = cutout.extent
    dx = (X - x) / (cutout.shape[1] - 1)
    dy = (Y - y) / (cutout.shape[0] - 1)
    return [x - dx/2., X + dx/2., y - dy/2., Y + dy/2.]


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_natura_raster') #has to be enabled
    configure_logging(snakemake)

    cutout_dir = Path(snakemake.input.cutouts[0]).parent.resolve()
    cutout_names = {res['cutout'] for res in snakemake.config['renewable'].values()}
    xs, Xs, ys, Ys = zip(*(determine_cutout_xXyY(cutout) for cutout in cutout_names))
    xXyY = min(xs), max(Xs), min(ys), max(Ys)

    natura = gk.vector.loadVector(snakemake.input.natura)
    extent = gk.Extent.from_xXyY(xXyY).castTo(3035).fit(100)
    extent.rasterize(natura, pixelWidth=100, pixelHeight=100, output=snakemake.output[0])
