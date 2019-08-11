"""
Creates Voronoi shapes for each bus representing both onshore and offshore regions.

Relevant Settings
-----------------

.. code:: yaml

    countries:


Inputs
------

- ``resources/country_shapes.geojson``: confer :ref:`shapes`
- ``resources/offshore_shapes.geojson``: confer :ref:`shapes`
- ``networks/base.nc``: confer :ref:`base`

Outputs
-------

- ``resources/regions_onshore.geojson``:

    .. image:: img/regions_onshore.png
        :scale: 33 %

- ``resources/regions_offshore.geojson``:

    .. image:: img/regions_offshore.png
        :scale: 33 %

Description
-----------

"""

import os
from operator import attrgetter

import pandas as pd
import geopandas as gpd

from vresutils.graph import voronoi_partition_pts

import pypsa
import logging

if __name__ == "__main__":
    logging.basicConfig(level=snakemake.config["logging_level"])

    countries = snakemake.config['countries']

    n = pypsa.Network(snakemake.input.base_network)

    country_shapes = gpd.read_file(snakemake.input.country_shapes).set_index('name')['geometry']
    offshore_shapes = gpd.read_file(snakemake.input.offshore_shapes).set_index('name')['geometry']

    onshore_regions = []
    offshore_regions = []

    for country in countries:
        c_b = n.buses.country == country

        onshore_shape = country_shapes[country]
        onshore_locs = n.buses.loc[c_b & n.buses.substation_lv, ["x", "y"]]
        onshore_regions.append(gpd.GeoDataFrame({
                'x': onshore_locs['x'],
                'y': onshore_locs['y'],
                'geometry': voronoi_partition_pts(onshore_locs.values, onshore_shape),
                'country': country
            }, index=onshore_locs.index))

        if country not in offshore_shapes.index: continue
        offshore_shape = offshore_shapes[country]
        offshore_locs = n.buses.loc[c_b & n.buses.substation_off, ["x", "y"]]
        offshore_regions_c = gpd.GeoDataFrame({
                'x': offshore_locs['x'],
                'y': offshore_locs['y'],
                'geometry': voronoi_partition_pts(offshore_locs.values, offshore_shape),
                'country': country
            }, index=offshore_locs.index)
        offshore_regions_c = offshore_regions_c.loc[offshore_regions_c.area > 1e-2]
        offshore_regions.append(offshore_regions_c)

    def save_to_geojson(s, fn):
        if os.path.exists(fn):
            os.unlink(fn)
        df = s.reset_index()
        schema = {**gpd.io.file.infer_schema(df), 'geometry': 'Unknown'}
        df.to_file(fn, driver='GeoJSON', schema=schema)

    save_to_geojson(pd.concat(onshore_regions), snakemake.output.regions_onshore)

    save_to_geojson(pd.concat(offshore_regions), snakemake.output.regions_offshore)
