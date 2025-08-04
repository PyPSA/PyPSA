PyPSA offers a variety of functions for plotting networks. These include the ability to plot different colours and widths on buses and branches, as well as labels on geographical features. Line, bar and area charts can be plotted against any network metric, and plots can be created as either static or dynamic.


## Static Plots
Various key metrics that can be calculated on any PyPSA network are described in the [:material-bookshelf: Statistics](statistics.md) section. The plotting module lets you create plots of any of these metrics.

!!! info
    The examples below are based on two networks: `n_simple` is based on a minimal three-node network, which is available via [pypsa.examples.ac_dc_meshed][], and `n` is the more complex hybrid scenario network from a recently published paper on carbon management. This network is also available in PyPSA via [pypsa.examples.carbon_management][].
    
    Load the networks:
    
    === "Simple Network"
        ```python
        # Simple three-node network
        >>> n_simple = pypsa.examples.ac_dc_meshed()
        >>> n_simple
        PyPSA Network 'AC-DC-Meshed'
        ----------------------------
        Components:
         - Bus: 9
         - Carrier: 6
         - Generator: 6
         - GlobalConstraint: 1
         - Line: 7
         - Link: 4
         - Load: 6
        Snapshots: 10
        ```
    
    === "Complex Network"
        ```python
        # Complex sector coupled network
        >>> n = pypsa.examples.carbon_management()
        >>> n
        PyPSA Network 'Hybrid Scenario from https://www.nature.com/articles/s41560-025-01752-6'
        ---------------------------------------------------------------------------------------
        Components:
            - Bus: 2164
            - Carrier: 89
            - Generator: 1489
            - GlobalConstraint: 4
            - Line: 157
            - Link: 6830
            - Load: 1357
            - StorageUnit: 106
            - Store: 1263
        Snapshots: 168
        ```


### Basic Plots

Many statistics are already available via the [`n.statistics`][pypsa.Network.statistics] accessor:

```python
>>> n_simple.statistics.energy_balance()
component  carrier  bus_carrier
Generator  gas      AC              1465.27439
           wind     AC             31082.35370
Load       load     AC            -32547.62808
dtype: float64
```

Any of these metrics can also be used to create plots straight away:

```python
>>> n_simple.statistics.energy_balance.plot()
```
<figure markdown="span">
  ![Buses](../assets/images/ac_dc_meshed-energy_balance-area_plot.png){ width="600" }
</figure>

The plot type varies depending on which type makes the most sense for a given metric. For instance, energy_balance produces an area plot with snapshots on the x-axis, whereas installed_capacity produces a simple bar chart without a time dimension.

```python
>>> n_simple.statistics.installed_capacity.plot()
```
<figure markdown="span">
  ![Buses](../assets/images/ac_dc_meshed-installed_capacity-bar_plot.png){ width="350" }
</figure>

#### Complex Networks
Plots can be created from any network. Depending on factors such as network size, number of components and snapshots, the plot may be difficult to read and require further customisation. If the default option is not useful, there are three options to get useful plots for complex networks:

1. The full static plot API is also available for interactive plots. This gives you more control over data selection and data can be easily explored. See [Interactive Plots](#interactive-plots).
2. PyPSA plots are created using [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) under the hood. This means that you can use all the customization options of these libraries to further customize the plot. See [Customization with matplotlib and seaborn  ](#customization-with-matplotlib-and-seaborn).
3. The parameters that can be used to filter and aggregate data using base statistics methods can also be passed to plotting methods. See [Customization based on statistics parameters](#customization-based-on-statistics-parameters).

All options are described in the following sections.

### Plot Types
The basic plotting method ([`n.statistics.<metric>.plot()`][pypsa.Network]) is not very flexible, but it is useful for quick exploration. To gain more control over the plot, the according plot type method can be called directly. Any plot type can be used with any metric, although not all plots will be meaningful for all metrics.

Energy balance can also be shown as a bar chart, ignoring the time dimension:
```python
>>> n_simple.statistics.energy_balance.plot.bar()
```
<figure markdown="span">
  ![Buses](../assets/images/ac_dc_meshed-energy_balance-bar_plot.png){ width="350" }
</figure>


#### Available Plot Types
The following plot types are available:

- [Area Plot][pypsa.Network]
- [Bar Plot][pypsa.Network]
- [Map Plot][pypsa.Network]
- [Scatter Plot][pypsa.Network]
- [Line Plot][pypsa.Network]
- [Box Plot][pypsa.Network]
- [Violin Plot][pypsa.Network]
- [Histogram Plot][pypsa.Network]

=== "Area Plot"
    ```python
    >>> n_simple.statistics.energy_balance.plot.area()
    ```
    <figure markdown="span">
      ![Buses](../assets/images/ac_dc_meshed-energy_balance-area_plot.png){ width="600" }
    </figure>

=== "Bar Plot"
    ```python
    >>> n_simple.statistics.energy_balance.plot.bar()
    ```
    <figure markdown="span">
      ![Buses](../assets/images/ac_dc_meshed-energy_balance-bar_plot.png){ width="350" }
    </figure>

=== "Map Plot"
    ```python
    >>> n_simple.statistics.energy_balance.plot.map()
    ```
    <figure markdown="span">
      ![Buses](../assets/images/ac_dc_meshed-energy_balance-map_plot.png){ width="600" }
    </figure>

=== "Scatter Plot"
    ```python
    >>> n_simple.statistics.energy_balance.plot.scatter()
    ```
    <figure markdown="span">
      ![Buses](../assets/images/ac_dc_meshed-energy_balance-scatter_plot.png){ width="350" }
    </figure>

=== "Line Plot"
    ```python
    >>> n_simple.statistics.energy_balance.plot.line()
    ```
    <figure markdown="span">
      ![Buses](../assets/images/ac_dc_meshed-energy_balance-line_plot.png){ width="600" }
    </figure>

=== "Box Plot"
    ```python
    >>> n_simple.statistics.energy_balance.plot.box()
    ```
    <figure markdown="span">
      ![Buses](../assets/images/ac_dc_meshed-energy_balance-box_plot.png){ width="350" }
    </figure>

=== "Violin Plot"
    ```python
    >>> n_simple.statistics.energy_balance.plot.violin()
    ```
    <figure markdown="span">
      ![Buses](../assets/images/ac_dc_meshed-energy_balance-violin_plot.png){ width="350" }
    </figure>

=== "Histogram Plot"
    ```python
    >>> n_simple.statistics.energy_balance.plot.histogram()
    ```
    <figure markdown="span">
      ![Buses](../assets/images/ac_dc_meshed-energy_balance-histogram_plot.png){ width="350" }
    </figure>


### Customization with matplotlib and seaborn
All plot methods return a matplotlib figure object, a matplotlib axes object and a facet grid object. These can be used to customise the plot further. See the matplotlib documentation on [`matplotlib.figure`](https://matplotlib.org/stable/api/figure_api.html), [`matplotlib.axes`](https://matplotlib.org/stable/api/axes_api.html) and the seaborn documentation on [`seaborn.FacetGrid`](https://seaborn.pydata.org/generated/seaborn.FacetGrid.html).

```python
>>> fig, ax, facet_col = n_simple.statistics.energy_balance.plot.area()
>>> fig.set_size_inches(12, 3)
>>> fig.suptitle("My Scenario", fontsize=12)
>>> ax.grid(True, alpha=0.5, linestyle="--")
>>> ax.set_xlabel("Time", fontsize=9)
>>> ax.set_ylabel("Energy Balance (MW)", fontsize=9)
```
<figure markdown="span">
  ![Buses](../assets/images/ac_dc_meshed-energy_balance-custom_area_plot.png){ width="800" }
</figure>

### Customization based on statistics parameters
Parameters which are available in the base statistics method, to filter and aggregate data, can also be passed to the plotting methods. The full energy balance of the carbon management example yields too many different `carriers` to be shown in a single plot. However, if we filter it down to include only buses with the 'AC' carrier, via the `bus_carrier` parameter from the statistics method, we can produce an appropriate plot.

=== "Buses with AC carrier"

    ```python
    >>> n.statistics.energy_balance.plot.area(bus_carrier="AC", figsize=(12, 3))
    ```
    <figure markdown="span">
    ![Buses](../assets/images/carbon_management-energy_balance-area_plot-AC-bus_carrier.png){ width="800" }
    </figure>

=== "All Buses"

    ```python
    >>> n.statistics.energy_balance.plot.area(figsize=(10, 3))
    ```
    <figure markdown="span">
    ![Buses](../assets/images/carbon_management-energy_balance-area_plot.png){ width="800" }
    </figure>


Behind the scenes, each plotting method selects a different set of parameters to call the relevant statistics method. Therefore, the default value can be different for each plot type/metric. Most of these parameters can also be passed directly to the plotting method. Please refer to the [Statistics User Guide](statistics.md) for more details.

!!! tip
    To decide which subset of data to show in your plot, it is often helpful to experiment with the relevant statistics method first to find the right parameters. Once you are happy with your selection, simply pass the parameters to the plotting method and the same selection will be applied. You can also pass them alongside all the other available plotting parameters. Check the API reference for details.


## Interactive Plots

All the logic for static plots, based on `n.statistics.<metric>.plot.<plot_type>()`, is completely mirrored for interactive plots. While the returned object and parameters may differ, you can usually simply replace `plot` with `iplot` to get an interactive version of the same plot. Behind the scenes, Plotly [Plotly](https://plotly.com/python/) is used for the interactive plots and the returned object is a Plotly figure object. Check out the [Plotly documentation](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html) for more details.

### Examples

#### Simple Energy Balance Area Plot

=== "Interactive Version"

    ```python
    >>> n_simple.statistics.energy_balance.iplot.area()
    ```
    
    <div style="width: 100%; height: 500px;">
        <iframe src="../../assets/interactive-plots/ac_dc_meshed-energy_balance-area_iplot.html" 
                width="100%" height="100%" frameborder="0" style="border: 1px solid #ccc;">
        </iframe>
    </div>

=== "Static Plot"

    ```python
    >>> n_simple.statistics.energy_balance.plot()
    ```
    <figure markdown="span">
    ![Buses](../assets/images/ac_dc_meshed-energy_balance-area_plot.png){ width="600" }
    </figure>

#### Carbon Network Energy Balance with bus carrier `AC`

=== "Interactive Version"

    ```python
    >>> n.statistics.energy_balance.iplot.area(bus_carrier="AC")
    ```
    
    <div style="width: 100%; height: 500px;">
        <iframe src="../../assets/interactive-plots/carbon_management-energy_balance-area_iplot-AC-bus_carrier.html" 
                width="100%" height="100%" frameborder="0" style="border: 1px solid #ccc;">
        </iframe>
    </div>

=== "Static Plot"

    ```python
    >>> n.statistics.energy_balance.plot(bus_carrier="AC")
    ```
    <figure markdown="span">
    ![Buses](../assets/images/carbon_management-energy_balance-area_plot-AC-bus_carrier.png){ width="800" }
    </figure>


#### Full Carbon Network Energy Balance

=== "Interactive Version"

    ```python
    >>> n.statistics.energy_balance.iplot.area()
    ```
    <div style="width: 100%; height: 500px;">
        <iframe src="../../assets/interactive-plots/carbon_management-energy_balance-area_iplot.html" 
                width="100%" height="100%" frameborder="0" style="border: 1px solid #ccc;">
        </iframe>
    </div>

=== "Static Plot"

    ```python
    >>> n.statistics.energy_balance.plot.area(figsize=(10, 3))
    ```
    <figure markdown="span">
    ![Buses](../assets/images/carbon_management-energy_balance-area_plot.png){ width="800" }
    </figure>


<!-- ## Plotting based on network data

Any plotting functionality is optional and only extend the existing statistics module to simplify the creation of plots. All data in PyPSA is  -->

<!-- Here, we are going to import a network and plot the electricity flow:

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
```  -->