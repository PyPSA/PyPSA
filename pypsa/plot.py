

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
import six
from six import iteritems

import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)


__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"


plt_present = True
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
    from matplotlib.collections import LineCollection, PatchCollection
except:
    plt_present = False

basemap_present = True
try:
    from mpl_toolkits.basemap import Basemap
except:
    basemap_present = False


pltly_present = True
try:
        import plotly.offline as pltly
except:
        pltly_present = False


def plot(network, margin=0.05, ax=None, basemap=True, bus_colors='b',
         line_colors='g', bus_sizes=10, line_widths=2, title="",
         line_cmap=None, bus_cmap=None, boundaries=None,
         geometry=False, branch_components=['Line', 'Link'], jitter=None):
    """
    Plot the network buses and lines using matplotlib and Basemap.

    Parameters
    ----------
    margin : float
        Margin at the sides as proportion of distance between max/min x,y
    ax : matplotlib ax, defaults to plt.gca()
        Axis to which to plot the network
    basemap : bool, default True
        Switch to use Basemap
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

    Returns
    -------
    bus_collection, branch_collection1, ... : tuple of Collections
        Collections for buses and branches.
    """

    defaults_for_branches = {
        'Link': dict(color="cyan", width=2),
        'Line': dict(color="b", width=2),
        'Transformer': dict(color='green', width=2)
    }

    if not plt_present:
        logger.error("Matplotlib is not present, so plotting won't work.")
        return

    if ax is None:
        ax = plt.gca()

    def compute_bbox_with_margins(margin, x, y):
        #set margins
        pos = np.asarray((x, y))
        minxy, maxxy = pos.min(axis=1), pos.max(axis=1)
        xy1 = minxy - margin*(maxxy - minxy)
        xy2 = maxxy + margin*(maxxy - minxy)
        return tuple(xy1), tuple(xy2)

    x = network.buses["x"]
    y = network.buses["y"]

    if jitter is not None:
        x = x + np.random.uniform(low=-jitter, high=jitter, size=len(x))
        y = y + np.random.uniform(low=-jitter, high=jitter, size=len(y))

    if basemap and basemap_present:
        resolution = 'l' if isinstance(basemap, bool) else basemap

        if boundaries is None:
            (x1, y1), (x2, y2) = compute_bbox_with_margins(margin, x, y)
        else:
            x1, x2, y1, y2 = boundaries
        bmap = Basemap(resolution=resolution, epsg=network.srid,
                       llcrnrlat=y1, urcrnrlat=y2, llcrnrlon=x1,
                       urcrnrlon=x2, ax=ax)
        bmap.drawcountries()
        bmap.drawcoastlines()

        x, y = bmap(x.values, y.values)
        x = pd.Series(x, network.buses.index)
        y = pd.Series(y, network.buses.index)

    if isinstance(bus_sizes, pd.Series) and isinstance(bus_sizes.index, pd.MultiIndex):
        # We are drawing pies to show all the different shares
        assert len(bus_sizes.index.levels[0].difference(network.buses.index)) == 0, \
            "The first MultiIndex level of bus_sizes must contain buses"
        assert isinstance(bus_colors, dict) and set(bus_colors).issuperset(bus_sizes.index.levels[1]), \
            "bus_colors must be a dictionary defining a color for each element " \
            "in the second MultiIndex level of bus_sizes"

        bus_sizes = bus_sizes.sort_index(level=0, sort_remaining=False)

        patches = []
        for b_i in bus_sizes.index.levels[0]:
            s = bus_sizes.loc[b_i]
            radius = s.sum()**0.5
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
        if c.dtype == np.dtype('O'):
            c.fillna("b", inplace=True)
            c = list(c.values)
        s = pd.Series(bus_sizes, index=network.buses.index, dtype="float").fillna(10)
        bus_collection = ax.scatter(x, y, c=c, s=s, cmap=bus_cmap)

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
    if not isinstance(line_cmap, dict):
        line_cmap = {'Line': line_cmap}

    branch_collections = []
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
            linestrings = c.df.geometry.map(loads)
            assert all(isinstance(ls, LineString) for ls in linestrings), \
                "The WKT-encoded geometry in the 'geometry' column must be composed of LineStrings"
            segments = np.asarray(list(linestrings.map(np.asarray)))
            if basemap and basemap_present:
                segments = np.transpose(bmap(*np.transpose(segments, (2, 0, 1))), (1, 2, 0))

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
        l_collection.set_zorder(1)

        branch_collections.append(l_collection)

    bus_collection.set_zorder(2)

    ax.update_datalim(compute_bbox_with_margins(margin, x, y))
    ax.autoscale_view()

    ax.set_title(title)

    return (bus_collection,) + tuple(branch_collections)


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
          branch_components=['Line', 'Link'], iplot=True):
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

    bus_trace = dict(x=network.buses.x,
                     y=network.buses.y,
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
        l_nums = None
        l_colors = line_colors.get(c.name, l_defaults['color'])

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

        x0 = c.df.bus0.map(network.buses.x)
        x1 = c.df.bus1.map(network.buses.x)

        y0 = c.df.bus0.map(network.buses.y)
        y1 = c.df.bus1.map(network.buses.y)

        for line in c.df.index:
            shapes.append(dict(type='line',
                          x0=x0[line],
                          y0=y0[line],
                          x1=x1[line],
                          y1=y1[line],
                          opacity=0.7,
                          line=dict(color=l_colors[line],
                                    width=l_widths[line])
                          ))

        shape_traces.append(dict(x=0.5*(x0+x1),
                                 y=0.5*(y0+y1),
                                 text=l_text,
                                 type="scatter",
                                 mode="markers",
                                 hoverinfo="text",
                                 marker=dict(opacity=0.))
                            )

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
