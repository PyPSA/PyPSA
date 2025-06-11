######################
 Plotting Networks
######################

PyPSA has several functions available for plotting networks with
different colors/widths/labels on buses and branches and
geographic features in the background.


Static plotting with matplotlib
===============================

Static plots of networks can be created that use the library
`matplotlib <https://matplotlib.org/>`_.

To plot a network with ``matplotlib``, run
``n.plot()``, see :py:meth:`pypsa.Network.plot` for details.

See also the `SciGRID matplotlib example
<https://pypsa.readthedocs.io/en/latest/examples/scigrid-lopf-then-pf.html>`_ and the `Flow plotting matplotlib example
<https://pypsa.readthedocs.io/en/latest/examples/flow-plot.html>`_.


Interactive plotting with plotly
================================

Interactive plots of networks can be created using `plotly
<https://plot.ly/python/>`_.
To plot a network with ``plotly``, run
``n.iplot()``, see :py:meth:`pypsa.Network.iplot` for details.


# Flow Plot Example

Here, we are going to import a network and plot the electricity flow

```python
import warnings

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
from shapely.errors import ShapelyDeprecationWarning

import pypsa

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
plt.rc("figure", figsize=(10, 8))
```

#### Import and optimize a network

```python
n = pypsa.examples.ac_dc_meshed(from_master=True)
n.optimize()
```

Get mean generator power by bus and carrier:

```python
gen = n.generators.assign(g=n.generators_t.p.mean()).groupby(["bus", "carrier"]).g.sum()
```

Plot the electricity flows:

```python
# links are not displayed for prettier output ('link_widths=0')
n.plot(
    bus_sizes=gen / 5e3,
    bus_colors={"gas": "indianred", "wind": "midnightblue"},
    margin=0.5,
    line_widths=0.1,
    line_flow="mean",
    link_widths=0,
)
plt.show()
```

Plot the electricity flows with a different projection and a colored map:

```python
# links are not displayed for prettier output ('link_widths=0')
n.plot(
    bus_sizes=gen / 5e3,
    bus_colors={"gas": "indianred", "wind": "midnightblue"},
    margin=0.5,
    line_widths=0.1,
    line_flow="mean",
    link_widths=0,
    projection=ccrs.EqualEarth(),
    color_geomap=True,
)
plt.show()
```

Set arbitrary values as flow argument using a specific level from the `n.branches()` MultiIndex:

```python
line_flow = pd.Series(10, index=n.branches().loc["Line"].index)
link_flow = pd.Series(10, index=n.branches().loc["Link"].index)
```

```python
line_flow
```

```python
# links are not displayed for prettier output ('link_widths=0')
n.plot(
    bus_sizes=gen / 5e3,
    bus_colors={"gas": "indianred", "wind": "midnightblue"},
    margin=0.5,
    line_flow=line_flow,
    link_flow=link_flow,
    line_widths=2.7,
    link_widths=0,
    projection=ccrs.EqualEarth(),
    color_geomap=True,
)
plt.show()
```

Adjust link colors according to their mean load:

```python
# Pandas series with MultiIndex
# links are not displayed for prettier output ('link_widths=0')
collections = n.plot(
    bus_sizes=gen / 5e3,
    bus_colors={"gas": "indianred", "wind": "midnightblue"},
    margin=0.5,
    line_flow=line_flow,
    line_widths=2.7,
    link_widths=0,
    projection=ccrs.EqualEarth(),
    color_geomap=True,
    line_colors=n.lines_t.p0.mean().abs(),
)

plt.colorbar(
    collections["branches"]["Line"], fraction=0.04, pad=0.004, label="Flow in MW"
)
plt.show()
``` 