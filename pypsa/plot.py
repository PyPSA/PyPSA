

## Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS)

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Functions for plotting networks.
"""


# make the code as Python 3 compatible as possible
from __future__ import division
from __future__ import absolute_import
from six import string_types

import pandas as pd
import numpy as np

import warnings
import logging
logger = logging.getLogger(__name__)


__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"


plt_present = True
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
    from matplotlib.collections import LineCollection, PatchCollection
except ImportError:
    plt_present = False

basemap_present = True
try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    basemap_present = False


cartopy_present = True
try:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.mpl.geoaxes
    import requests
except ImportError:
    cartopy_present = False

pltly_present = True
try:
    import plotly.offline as pltly
except ImportError:
    pltly_present = False


def plot(network, margin=0.05, ax=None, geomap=True, projection=None,
         bus_colors='b', line_colors={'Line':'g', 'Link':'cyan'}, bus_sizes=10,
         line_widths={'Line':2, 'Link':2},
         flow=None, title="", line_cmap=None, bus_cmap=None, boundaries=None,
         geometry=False, branch_components=['Line', 'Link'], jitter=None,
         basemap=None, basemap_parameters=None, color_geomap=None):
    """
    Plot the network buses and lines using matplotlib and Basemap.

    Parameters
    ----------
    margin : float
        Margin at the sides as proportion of distance between max/min x,y
    ax : matplotlib ax, defaults to plt.gca()
        Axis to which to plot the network
    geomap: bool/str, default True
        Switch to use Basemap or Cartopy (depends on what is installed).
        If string is passed, it will be used as a resolution argument.
        For Basemap users 'c' (crude), 'l' (low), 'i' (intermediate),
        'h' (high), 'f' (full) are valid resolutions options.
        For Cartopy users '10m', '50m', '110m' are valid resolutions options.
    projection: cartopy.crs.Projection, defaults to None
        Define the projection of your geomap, only valid if cartopy is
        installed. If None (default) is passed the projection for cartopy
        is set to cartopy.crs.PlateCarree
    bus_colors : dict/pandas.Series
        Colors for the buses, defaults to "b"
    bus_sizes : dict/pandas.Series
        Sizes of bus points, defaults to 10
    line_colors : dict/pandas.Series
        Colors for the lines, defaults to "g" for Lines and "cyan" for
        Links. Colors for branches other than Lines can be
        specified using a pandas Series with a MultiIndex.
    line_widths : dict/pandas.Series
        Widths of lines, defaults to 2. Widths for branches other
        than Lines can be specified using a pandas Series with a
        MultiIndex.
    flow : snapshot/pandas.Series/function/string
        Flow to be displayed in the plot, defaults to None. If an element of
        network.snapshots is given, the flow at this timestamp will be
        displayed. If an aggregation function is given, is will be applied
        to the total network flow via pandas.DataFrame.agg (accepts also
        function names). Otherwise flows can be specified by passing a pandas
        Series with MultiIndex including all necessary branch components.
        Use the line_widths argument to additionally adjust the size of the
        flow arrows.
    title : string
        Graph title
    line_cmap : plt.cm.ColorMap/str|dict
        If line_colors are floats, this color map will assign the colors.
        Use a dict to specify colormaps for more than one branch type.
    bus_cmap : plt.cm.ColorMap/str
        If bus_colors are floats, this color map will assign the colors
    boundaries : list of four floats
        Boundaries of the plot in format [x1,x2,y1,y2]
    branch_components : list of str
        Branch components to be plotted, defaults to Line and Link.
    jitter : None|float
        Amount of random noise to add to bus positions to distinguish
        overlapping buses
    basemap_parameters : dict
        Specify a dict with additional constructor parameters for the
        Basemap. Will disable Cartopy.
        Use this feature to set a custom projection.
        (e.g. `{'projection': 'tmerc', 'lon_0':10.0, 'lat_0':50.0}`)
    color_geomap : dict or bool
        Specify colors to paint land and sea areas in.
        If True, it defaults to `{'ocean': 'lightblue', 'land': 'whitesmoke'}`.
        If no dictionary is provided, colors are white.

    Returns
    -------
    bus_collection, branch_collection1, ... : tuple of Collections
        Collections for buses and branches.
    """
    defaults_for_branches = pd.Series({
        'Link': dict(color="cyan", width=2),
        'Line': dict(color="b", width=2),
        'Transformer': dict(color='green', width=2)
    }).rename_axis('component')

    if not plt_present:
        logger.error("Matplotlib is not present, so plotting won't work.")
        return

    if basemap is not None:
        logger.warning("argument `basemap` is deprecated, "
                       "use `geomap` instead.")
        geomap = basemap

    if geomap:
        if not (cartopy_present or basemap_present):
            # Not suggesting Basemap since it is being deprecated
            logger.warning("Cartopy needs to be installed to use `geomap=True`.")
            geomap = False

        # Use cartopy by default, fall back on basemap
        use_basemap = False
        use_cartopy = cartopy_present
        if not use_cartopy:
            use_basemap = basemap_present

        # If the user specifies basemap parameters, they prefer
        # basemap over cartopy.
        # (This means that you can force the use of basemap by
        # setting `basemap_parameters={}`)
        if basemap_present:
            if basemap_parameters is not None:
                logger.warning("Basemap is being deprecated, consider "
                               "switching to Cartopy.")
                use_basemap = True
                use_cartopy = False

        if use_cartopy:
            if projection is None:
                projection = get_projection_from_crs(network.srid)

            if ax is None:
                ax = plt.gca(projection=projection)
            else:
                assert isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot), (
                        'The passed axis is not a GeoAxesSubplot. You can '
                        'create one with: \nimport cartopy.crs as ccrs \n'
                        'fig, ax = plt.subplots('
                        'subplot_kw={"projection":ccrs.PlateCarree()})')
    elif ax is None:
        ax = plt.gca()

    x, y = network.buses["x"],  network.buses["y"]

    axis_transform = ax.transData

    if geomap:
        if use_cartopy:
            axis_transform = draw_map_cartopy(network, x, y, ax,
                    boundaries, margin, geomap, color_geomap)
            new_coords = pd.DataFrame(
                    ax.projection.transform_points(axis_transform,
                                                   x.values, y.values),
                       index=network.buses.index, columns=['x', 'y', 'z'])
            x, y = new_coords['x'], new_coords['y']
        elif use_basemap:
            basemap_transform = draw_map_basemap(network, x, y, ax,
                    boundaries, margin, geomap, basemap_parameters, color_geomap)

            # A non-standard projection might be used; the easiest way to
            # support this is to tranform the bus coordinates.
            x, y = basemap_transform(x.values, y.values)
            x = pd.Series(x, network.buses.index)
            y = pd.Series(y, network.buses.index)

    if jitter is not None:
        x = x + np.random.uniform(low=-jitter, high=jitter, size=len(x))
        y = y + np.random.uniform(low=-jitter, high=jitter, size=len(y))

    if isinstance(bus_sizes, pd.Series) and isinstance(bus_sizes.index, pd.MultiIndex):
        # We are drawing pies to show all the different shares
        assert len(bus_sizes.index.levels[0].difference(network.buses.index)) == 0, \
            "The first MultiIndex level of bus_sizes must contain buses"
        assert (isinstance(bus_colors, dict) and
                set(bus_colors).issuperset(bus_sizes.index.levels[1])), \
            "bus_colors must be a dictionary defining a color for each element " \
            "in the second MultiIndex level of bus_sizes"

        bus_sizes = bus_sizes.sort_index(level=0, sort_remaining=False)
        if geomap:
            bus_sizes *= projected_area_factor(ax, network.srid)**2

        patches = []
        for b_i in bus_sizes.index.levels[0]:
            s = bus_sizes.loc[b_i]
            radius = s.sum()**0.5
            if radius == 0.0:
                ratios = s
            else:
                ratios = s/s.sum()

            start = 0.25
            for i, ratio in ratios.iteritems():
                patches.append(Wedge((x.at[b_i], y.at[b_i]), radius,
                                     360*start, 360*(start+ratio),
                                     facecolor=bus_colors[i]))
                start += ratio
        bus_collection = PatchCollection(patches, match_original=True)
        ax.add_collection(bus_collection)
    else:
        c = pd.Series(bus_colors, index=network.buses.index)
        s = pd.Series(bus_sizes, index=network.buses.index, dtype="float").fillna(10)
        bus_collection = ax.scatter(x, y, c=c, s=s, cmap=bus_cmap, edgecolor='face')

    def as_branch_series(ser):
        # ensure that this function always return a multiindexed series
        if isinstance(ser, dict) and set(ser).issubset(branch_components):
            return pd.concat(
                    {c.name: pd.Series(s, index=c.df.index) for c, s in
                         zip(network.iterate_components(ser.keys()), ser.values())},
                    names=['component', 'name'])
        elif isinstance(ser, pd.Series) and isinstance(ser.index, pd.MultiIndex):
            return ser.rename_axis(index=['component', 'name'])
        else:
            ser =  pd.Series(ser, network.lines.index)
            return pd.concat([ser], axis=0, keys=['Line'],
                             names=['component', 'name']).fillna(0)

    line_colors = as_branch_series(line_colors)
    line_widths = as_branch_series(line_widths)

    if not isinstance(line_cmap, dict):
        line_cmap = {'Line': line_cmap}

    branch_collections = []

    if flow is not None:
        flow = (_flow_ds_from_arg(flow, network, branch_components)
                .pipe(as_branch_series)
                .div(sum(len(t.df) for t in
                         network.iterate_components(branch_components)) + 100))
        flow = flow.mul(line_widths[flow.index], fill_value=1)
        # update the line width, allows to set line widths separately from flows
        line_widths.update((5 * flow.abs()).pipe(np.sqrt))
        arrows = directed_flow(network, flow, x=x, y=y, ax=ax, geomap=geomap,
                               branch_colors=line_colors,
                               branch_comps=branch_components,
                               cmap=line_cmap['Line'])
        branch_collections.append(arrows)


    for c in network.iterate_components(branch_components):
        l_defaults = defaults_for_branches[c.name]
        l_widths = line_widths.get(c.name, l_defaults['width'])
        l_nums = None
        l_colors = line_colors.get(c.name, l_defaults['color'])

        if isinstance(l_colors, pd.Series):
            if issubclass(l_colors.dtype.type, np.number):
                l_nums = l_colors
                l_colors = None
            else:
                l_colors.fillna(l_defaults['color'], inplace=True)

        if not geometry:
            segments = (np.asarray(((c.df.bus0.map(x),
                                     c.df.bus0.map(y)),
                                    (c.df.bus1.map(x),
                                     c.df.bus1.map(y))))
                        .transpose(2, 0, 1))
        else:
            from shapely.wkt import loads
            from shapely.geometry import LineString
            linestrings = c.df.geometry[lambda ds: ds != ''].map(loads)
            assert all(isinstance(ls, LineString) for ls in linestrings), (
                "The WKT-encoded geometry in the 'geometry' column must be "
                "composed of LineStrings")
            segments = np.asarray(list(linestrings.map(np.asarray)))

        l_collection = LineCollection(segments,
                                      linewidths=l_widths,
                                      antialiaseds=(1,),
                                      colors=l_colors,
                                      transOffset=ax.transData)

        if l_nums is not None:
            l_collection.set_array(np.asarray(l_nums))
            l_collection.set_cmap(line_cmap.get(c.name, None))
            l_collection.autoscale()

        ax.add_collection(l_collection)
        l_collection.set_zorder(3)

        branch_collections.append(l_collection)

    bus_collection.set_zorder(4)

    ax.update_datalim(compute_bbox_with_margins(margin, x, y))
    ax.autoscale_view()

    if geomap:
        if use_cartopy:
            ax.outline_patch.set_visible(False)
        ax.axis('off')

    ax.set_title(title)

    return (bus_collection,) + tuple(branch_collections)


def get_projection_from_crs(crs):
    if crs == 4326:
        # if data is in latlon system, return default map with latlon system
        return ccrs.PlateCarree()
    try:
        return ccrs.epsg(crs)
    except requests.RequestException:
        logger.warning("A connection to http://epsg.io/ is "
                       "required for a projected coordinate reference system. "
                       "Falling back to latlong.")
    except ValueError:
        logger.warning("'{crs}' does not define a projected coordinate system. "
                       "Falling back to latlong.".format(crs=crs))
        return ccrs.PlateCarree()


def compute_bbox_with_margins(margin, x, y):
    'Helper function to compute bounding box for the plot'
    # set margins
    pos = np.asarray((x, y))
    minxy, maxxy = pos.min(axis=1), pos.max(axis=1)
    xy1 = minxy - margin*(maxxy - minxy)
    xy2 = maxxy + margin*(maxxy - minxy)
    return tuple(xy1), tuple(xy2)


def projected_area_factor(ax, original_crs=4326):
    """
    Helper function to get the area scale of the current projection in
    reference to the default projection. The default 'original crs' is assumed
    to be 4326, which translates to the cartopy default cartopy.crs.PlateCarree()
    """
    if not hasattr(ax, 'projection'):
        return 1
    if isinstance(ax.projection, ccrs.PlateCarree):
        return 1
    x1, x2, y1, y2 = ax.get_extent()
    pbounds = \
        get_projection_from_crs(original_crs).transform_points(ax.projection,
                    np.array([x1, x2]), np.array([y1, y2]))

    return np.sqrt(abs((x2 - x1) * (y2 - y1))
                   /abs((pbounds[0] - pbounds[1])[:2].prod()))



def draw_map_basemap(network, x, y, ax, boundaries=None, margin=0.05,
                     geomap=True, basemap_parameters=None, color_geomap=None):

    if boundaries is None:
        (x1, y1), (x2, y2) = compute_bbox_with_margins(margin, x, y)
    else:
        x1, x2, y1, y2 = boundaries

    if basemap_parameters is None:
        basemap_parameters = {}

    resolution = 'l' if isinstance(geomap, bool) else geomap
    gmap = Basemap(resolution=resolution,
                    llcrnrlat=y1, urcrnrlat=y2, llcrnrlon=x1,
                    urcrnrlon=x2, ax=ax, **basemap_parameters)
    gmap.drawcountries(linewidth=0.3, zorder=2)
    gmap.drawcoastlines(linewidth=0.4, zorder=2)

    if color_geomap is None:
        color_geomap = {'ocean': 'w', 'land': 'w'}
    elif color_geomap and not isinstance(color_geomap, dict):
        color_geomap = {'ocean': 'lightblue', 'land': 'whitesmoke'}

    gmap.drawlsmask(land_color=color_geomap['land'],
                    ocean_color=color_geomap['ocean'],
                    grid=1.25, ax=ax, zorder=1)

    # no transformation -> use the default
    basemap_projection = gmap

    # disable gmap transformation due to arbitrary conversion
    # x, y = gmap(x.values, y.values)

    return basemap_projection

def draw_map_cartopy(network, x, y, ax, boundaries=None, margin=0.05,
                     geomap=True, color_geomap=None):

    if boundaries is None:
        (x1, y1), (x2, y2) = compute_bbox_with_margins(margin, x, y)
    else:
        x1, x2, y1, y2 = boundaries

    resolution = '50m' if isinstance(geomap, bool) else geomap
    assert resolution in ['10m', '50m', '110m'], (
            "Resolution has to be one of '10m', '50m', '110m'")
    axis_transformation = get_projection_from_crs(network.srid)
    ax.set_extent([x1, x2, y1, y2], crs=axis_transformation)

    if color_geomap is None:
        color_geomap = {'ocean': 'w', 'land': 'w'}
    elif color_geomap and not isinstance(color_geomap, dict):
        color_geomap = {'ocean': 'lightblue', 'land': 'whitesmoke'}

    ax.add_feature(cartopy.feature.LAND.with_scale(resolution),
                    facecolor=color_geomap['land'])
    ax.add_feature(cartopy.feature.OCEAN.with_scale(resolution),
                    facecolor=color_geomap['ocean'])

    ax.coastlines(linewidth=0.4, zorder=2, resolution=resolution)
    border = cartopy.feature.BORDERS.with_scale(resolution)
    ax.add_feature(border, linewidth=0.3)

    return axis_transformation


def _flow_ds_from_arg(flow, n, branch_components):
    if isinstance(flow, pd.Series):
        return flow
    if flow in n.snapshots:
        return (pd.concat([n.pnl(c).p0.loc[flow]
                for c in branch_components],
                keys=branch_components, sort=True))
    elif isinstance(flow, str) or callable(flow):
        return (pd.concat([n.pnl(c).p0 for c in branch_components],
                axis=1, keys=branch_components, sort=True)
                .agg(flow, axis=0))


def directed_flow(n, flow, x=None, y=None, ax=None, geomap=True,
                  branch_colors='darkgreen', branch_comps=['Line', 'Link'],
                  cmap=None):
    """
    Helper function to generate arrows from flow data.
    """
    # this funtion is used for diplaying arrows representing the network flow
    from matplotlib.patches import FancyArrow
    if ax is None:
        ax = plt.gca()
    x = n.buses.x if x is None else x
    y = n.buses.y if y is None else y

    #set the scale of the arrowsizes
    fdata = pd.concat([pd.DataFrame(
                      {'x1': n.df(l).bus0.map(x),
                       'y1': n.df(l).bus0.map(y),
                       'x2': n.df(l).bus1.map(x),
                       'y2': n.df(l).bus1.map(y)})
                      for l in branch_comps], keys=branch_comps,
                    names=['component', 'name'])
    fdata['arrowsize'] = flow.abs().pipe(np.sqrt).clip(lower=1e-8)
    if geomap:
        fdata['arrowsize']= fdata['arrowsize'].mul(projected_area_factor(ax, n.srid))
    fdata['direction'] = np.sign(flow)
    fdata['linelength'] = (np.sqrt((fdata.x1 - fdata.x2)**2. +
                           (fdata.y1 - fdata.y2)**2))
    fdata['arrowtolarge'] = (1.5 * fdata.arrowsize >
                             fdata.loc[:, 'linelength'])
    # swap coords for negativ directions
    fdata.loc[fdata.direction == -1., ['x1', 'x2', 'y1', 'y2']] = \
        fdata.loc[fdata.direction == -1., ['x2', 'x1', 'y2', 'y1']].values
    if ((fdata.linelength > 0.) & (~fdata.arrowtolarge)).any():
        fdata['arrows'] = (
                fdata[(fdata.linelength > 0.) & (~fdata.arrowtolarge)]
                .apply(lambda ds:
                       FancyArrow(ds.x1, ds.y1,
                                  0.6*(ds.x2 - ds.x1) - ds.arrowsize
                                  * 0.75 * (ds.x2 - ds.x1) / ds.linelength,
                                  0.6 * (ds.y2 - ds.y1) - ds.arrowsize
                                  * 0.75 * (ds.y2 - ds.y1)/ds.linelength,
                                  head_width=ds.arrowsize), axis=1))
    fdata.loc[(fdata.linelength > 0.) & (fdata.arrowtolarge), 'arrows'] = \
        (fdata[(fdata.linelength > 0.) & (fdata.arrowtolarge)]
         .apply(lambda ds:
                FancyArrow(ds.x1, ds.y1,
                           0.001*(ds.x2 - ds.x1),
                           0.001*(ds.y2 - ds.y1),
                           head_width=ds.arrowsize), axis=1))
    if isinstance(branch_colors.index, (pd.MultiIndex, str)):
        # Catch the case that only multiindex with 'Line' in first level is passed
        fdata = fdata.assign(color=branch_colors.reindex_like(fdata)
                                                .fillna('darkgreen'))
    else:
        fdata = fdata.join(branch_colors.rename('color'))
    fdata = fdata.dropna(subset=['arrows'])
    arrowcol = PatchCollection(fdata.arrows,
                               color=fdata.color,
                               edgecolors='k',
                               linewidths=0.,
                               zorder=3, alpha=1)
    ax.add_collection(arrowcol)
    return arrowcol



#This function was borne out of a breakout group at the October 2017
#Munich Open Energy Modelling Initiative Workshop to hack together a
#working example of plotly for networks, see:
#https://forum.openmod-initiative.org/t/breakout-group-on-visualising-networks-with-plotly/384/7

#We thank Bryn Pickering for holding the tutorial on plotly which
#inspired the breakout group and for contributing ideas to the iplot
#function below.

def iplot(network, fig=None, bus_colors='blue',
          bus_colorscale=None, bus_colorbar=None, bus_sizes=10, bus_text=None,
          line_colors='green', line_widths=2, line_text=None, title="",
          branch_components=['Line', 'Link'], iplot=True, jitter=None):
    """
    Plot the network buses and lines interactively using plotly.

    Parameters
    ----------
    fig : dict, default None
        If not None, figure is built upon this fig.
    bus_colors : dict/pandas.Series
        Colors for the buses, defaults to "b"
    bus_colorscale : string
        Name of colorscale if bus_colors are floats, e.g. 'Jet', 'Viridis'
    bus_colorbar : dict
        Plotly colorbar, e.g. {'title' : 'my colorbar'}
    bus_sizes : dict/pandas.Series
        Sizes of bus points, defaults to 10
    bus_text : dict/pandas.Series
        Text for each bus, defaults to bus names
    line_colors : dict/pandas.Series
        Colors for the lines, defaults to "g" for Lines and "cyan" for
        Links. Colors for branches other than Lines can be
        specified using a pandas Series with a MultiIndex.
    line_widths : dict/pandas.Series
        Widths of lines, defaults to 2. Widths for branches other
        than Lines can be specified using a pandas Series with a
        MultiIndex.
    line_text : dict/pandas.Series
        Text for lines, defaults to line names. Text for branches other
        than Lines can be specified using a pandas Series with a
        MultiIndex.
    title : string
        Graph title
    branch_components : list of str
        Branch components to be plotted, defaults to Line and Link.
    iplot : bool, default True
        Automatically do an interactive plot of the figure.
    jitter : None|float
        Amount of random noise to add to bus positions to distinguish
        overlapping buses

    Returns
    -------
    fig: dictionary for plotly figure
    """

    defaults_for_branches = {
        'Link': dict(color="cyan", width=2),
        'Line': dict(color="blue", width=2),
        'Transformer': dict(color='green', width=2)
    }

    if fig is None:
        fig = dict(data=[],layout={})

    if bus_text is None:
        bus_text = 'Bus ' + network.buses.index

    x = network.buses.x
    y = network.buses.y

    if jitter is not None:
        x = x + np.random.uniform(low=-jitter, high=jitter, size=len(x))
        y = y + np.random.uniform(low=-jitter, high=jitter, size=len(y))

    bus_trace = dict(x=x, y=y,
                     text=bus_text,
                     type="scatter",
                     mode="markers",
                     hoverinfo="text",
                     marker=dict(color=bus_colors,
                                 size=bus_sizes),
                     )

    if bus_colorscale is not None:
        bus_trace['marker']['colorscale'] = bus_colorscale

    if bus_colorbar is not None:
        bus_trace['marker']['colorbar'] = bus_colorbar


    def as_branch_series(ser):
        if isinstance(ser, dict) and set(ser).issubset(branch_components):
            return pd.Series(ser)
        elif isinstance(ser, pd.Series):
            if isinstance(ser.index, pd.MultiIndex):
                return ser
            index = ser.index
            ser = ser.values
        else:
            index = network.lines.index
        return pd.Series(ser,
                         index=pd.MultiIndex(levels=(["Line"], index),
                                             labels=(np.zeros(len(index)),
                                                     np.arange(len(index)))))

    line_colors = as_branch_series(line_colors)
    line_widths = as_branch_series(line_widths)

    if line_text is not None:
        line_text = as_branch_series(line_text)

    shapes = []

    shape_traces = []

    for c in network.iterate_components(branch_components):
        l_defaults = defaults_for_branches[c.name]
        l_widths = line_widths.get(c.name, l_defaults['width'])
        l_colors = line_colors.get(c.name, l_defaults['color'])
        l_nums = None

        if line_text is None:
            l_text = c.name + ' ' + c.df.index
        else:
            l_text = line_text.get(c.name)

        if isinstance(l_colors, pd.Series):
            if issubclass(l_colors.dtype.type, np.number):
                l_nums = l_colors
                l_colors = None
            else:
                l_colors.fillna(l_defaults['color'], inplace=True)

        x0 = c.df.bus0.map(x)
        x1 = c.df.bus1.map(x)

        y0 = c.df.bus0.map(y)
        y1 = c.df.bus1.map(y)

        for line in c.df.index:
            color = l_colors if isinstance(l_colors, string_types) else l_colors[line]
            width = l_widths if isinstance(l_widths, (int, float)) else l_widths[line]

            shapes.append(dict(type='line',
                               x0=x0[line],
                               y0=y0[line],
                               x1=x1[line],
                               y1=y1[line],
                               opacity=0.7,
                               line=dict(color=color, width=width)))

        shape_traces.append(dict(x=0.5*(x0+x1),
                                 y=0.5*(y0+y1),
                                 text=l_text,
                                 type="scatter",
                                 mode="markers",
                                 hoverinfo="text",
                                 marker=dict(opacity=0.)))

    fig['data'].extend([bus_trace]+shape_traces)

    fig['layout'].update(dict(shapes=shapes,
                              title=title,
                              hovermode='closest',
                              showlegend=False))
                              #xaxis=dict(range=[6,14]),
                              #yaxis=dict(range=[47,55])


    if iplot:
        if not pltly_present:
            logger.warning("Plotly is not present, so interactive plotting won't work.")
        else:
            pltly.iplot(fig)

    return fig
