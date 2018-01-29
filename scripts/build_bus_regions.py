import os
from operator import attrgetter

import pandas as pd
import geopandas as gpd

from vresutils import shapes as vshapes
from vresutils.graph import voronoi_partition_pts

import pypsa

countries = snakemake.config['countries']

n = pypsa.Network(snakemake.input.base_network)

country_shapes = vshapes.countries(subset=countries, add_KV_to_RS=True,
                                   tolerance=0.01, minarea=0.1)
offshore_shapes = vshapes.eez(subset=countries, tolerance=0.01)

onshore_regions = []
offshore_regions = []

for country in countries:
    c_b = n.buses.country == country

    onshore_shape = country_shapes[country]
    onshore_locs = n.buses.loc[c_b & n.buses.substation_lv, ["x", "y"]]
    onshore_regions.append(gpd.GeoDataFrame({
            'geometry': voronoi_partition_pts(onshore_locs.values, onshore_shape),
            'country': country
        }, index=onshore_locs.index))

    if country not in offshore_shapes: continue
    offshore_shape = offshore_shapes[country]
    offshore_locs = n.buses.loc[c_b & n.buses.substation_off, ["x", "y"]]
    offshore_regions_c = gpd.GeoDataFrame({
            'geometry': voronoi_partition_pts(offshore_locs.values, offshore_shape),
            'country': country
        }, index=offshore_locs.index)
    offshore_regions_c = offshore_regions_c.loc[offshore_regions_c.area > 1e-2]
    offshore_regions.append(offshore_regions_c)

def save_to_geojson(s, fn):
    if os.path.exists(fn):
        os.unlink(fn)
    s.reset_index().to_file(fn, driver='GeoJSON')

save_to_geojson(pd.concat(onshore_regions), snakemake.output.regions_onshore)

save_to_geojson(pd.concat(offshore_regions), snakemake.output.regions_offshore)
