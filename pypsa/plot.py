

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


__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"


plt_present = True
try:
    import matplotlib.pyplot as plt
except:
    plt_present = False

basemap_present = True
try:
    from mpl_toolkits.basemap import Basemap
except:
    basemap_present = False



def plot(network,margin=0.05,ax=None,basemap=True,bus_colors={},
         line_colors={},bus_sizes={},line_widths={},title="",
         line_cmap=None,bus_cmap=None):
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

    Returns
    -------
    ax : matplotlib axis
    """

    if not plt_present:
        print("Matplotlib is not present, so plotting won't work.")
        return

    if ax is None:
        ax = plt.gca()

    #set margins

    mn = network.buses["x"].min()
    mx = network.buses["x"].max()

    x1 = mn - margin*(mx-mn)
    x2 = mx + margin*(mx-mn)

    mn = network.buses["y"].min()
    mx = network.buses["y"].max()

    y1 = mn - margin*(mx-mn)
    y2 = mx + margin*(mx-mn)

    x = network.buses["x"].values
    y = network.buses["y"].values

    if basemap and basemap_present:
        bmap = Basemap(resolution='l',epsg=network.srid,llcrnrlat=y1,urcrnrlat=y2,llcrnrlon=x1,urcrnrlon=x2,ax=ax)
        bmap.drawcountries()
        bmap.drawcoastlines()
        x,y = bmap(network.buses["x"].values,network.buses["y"].values)

    c = pd.Series(bus_colors, index=network.buses.index).fillna("b")

    s = pd.Series(bus_sizes, index=network.buses.index, dtype="float").fillna(10)

    ax.scatter(x, y, c=c, s=s, cmap=bus_cmap)

    if line_cmap is not None:
        line_nums = pd.Series(line_colors,index=network.lines.index)
        line_colors = pd.DataFrame(line_cmap(line_nums.values),index=network.lines.index)

    #should probably use LineCollection here instead - cf. networkx implementation
    for line in network.lines.obj:
        bus0 = network.buses.obj[line.bus0]
        bus1 = network.buses.obj[line.bus1]

        x,y = ([bus0.x,bus1.x],[bus0.y,bus1.y])

        if basemap and basemap_present:
            x,y = bmap(x,y)
        if isinstance(line_colors, pd.DataFrame):
            color = line_colors.loc[line.name]
        else:
            color = line_colors.get(line.name,"g")
        alpha = 0.7
        width = line_widths.get(line.name,2.)

        ax.plot(x,y,color=color,alpha=alpha,linewidth=width)

    ax.set_title(title)

    return ax
