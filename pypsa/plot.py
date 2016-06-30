

## Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS)

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
from __future__ import print_function, division
from __future__ import absolute_import
from six import iteritems

import pandas as pd
import numpy as np

__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"


plt_present = True
try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
except:
    plt_present = False

basemap_present = True
try:
    from mpl_toolkits.basemap import Basemap
except:
    basemap_present = False



def plot(network, margin=0.05, ax=None, basemap=True, bus_colors='b',
         line_colors='g', bus_sizes=10, line_widths=2, title="",
         line_cmap=None, bus_cmap=None, boundaries=None):
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
        Colors for the lines, defaults to "g"
    line_widths : dict/pandas.Series
        Widths of lines, defaults to 2
    title : string
        Graph title
    line_cmap : plt.cm.ColorMap
        If line_colors are floats, this color map will assign the colors
    bus_cmap : plt.cm.ColorMap
        If bus_colors are floats, this color map will assign the colors
    boundaries : list of four floats
        Boundaries of the plot in format [x1,x2,y1,y2]

    Returns
    -------
    ax : matplotlib axis
    """

    if not plt_present:
        print("Matplotlib is not present, so plotting won't work.")
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

    if basemap and basemap_present:
        if boundaries is None:
            (x1, y1), (x2, y2) = compute_bbox_with_margins(margin, x, y)
        else:
            x1, x2, y1, y2 = boundaries
        bmap = Basemap(resolution='l', epsg=network.srid,
                       llcrnrlat=y1, urcrnrlat=y2, llcrnrlon=x1,
                       urcrnrlon=x2, ax=ax)
        bmap.drawcountries()
        bmap.drawcoastlines()

        x, y = bmap(x.values, y.values)
        x = pd.Series(x, network.buses.index)
        y = pd.Series(y, network.buses.index)

    c = pd.Series(bus_colors, index=network.buses.index)
    if c.dtype == np.dtype('O'):
        c.fillna("b", inplace=True)
    s = pd.Series(bus_sizes, index=network.buses.index, dtype="float").fillna(10)
    bus_collection = ax.scatter(x, y, c=c, s=s, cmap=bus_cmap)

    if line_cmap is not None:
        line_nums = pd.Series(line_colors, index=network.lines.index)
        line_colors = None

    line_widths = pd.Series(line_widths, index=network.lines.index)

    segments = (np.asarray(((network.lines.bus0.map(x),
                             network.lines.bus0.map(y)),
                            (network.lines.bus1.map(x),
                             network.lines.bus1.map(y))))
                .transpose(2, 0, 1))

    line_collection = LineCollection(segments,
                                     linewidths=line_widths,
                                     antialiaseds=(1,),
                                     colors=line_colors,
                                     transOffset=ax.transData)

    if line_colors is None:
        line_collection.set_array(np.asarray(line_nums))
        line_collection.set_cmap(line_cmap)
        line_collection.autoscale()

    ax.add_collection(line_collection)

    bus_collection.set_zorder(2)
    line_collection.set_zorder(1)

    ax.update_datalim(compute_bbox_with_margins(margin, x, y))
    ax.autoscale_view()

    ax.set_title(title)

    return bus_collection, line_collection
