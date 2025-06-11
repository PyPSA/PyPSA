###########
Statistics
###########

.. seealso::

    See :doc:`/api/statistics` for the API reference.

Overview
=========

The ``pypsa.statistics`` module provides an accessor to the ``pypsa.Network`` object 
via ``n.statistics``, enabling users to efficiently calculate key network metrics. 
This module is designed to simplify the analysis of network data by abstracting complex 
calculations into more accessible methods. It is particularly useful for users who 
prefer to avoid manual calculations and directly obtain relevant statistical data 
about the network.


Available Metrics
==================

.. autosummary::

    ~pypsa.statistics.StatisticsAccessor.system_cost
    ~pypsa.statistics.StatisticsAccessor.capex
    ~pypsa.statistics.StatisticsAccessor.installed_capex
    ~pypsa.statistics.StatisticsAccessor.expanded_capex
    ~pypsa.statistics.StatisticsAccessor.optimal_capacity
    ~pypsa.statistics.StatisticsAccessor.installed_capacity
    ~pypsa.statistics.StatisticsAccessor.expanded_capacity
    ~pypsa.statistics.StatisticsAccessor.opex
    ~pypsa.statistics.StatisticsAccessor.supply
    ~pypsa.statistics.StatisticsAccessor.withdrawal
    ~pypsa.statistics.StatisticsAccessor.transmission
    ~pypsa.statistics.StatisticsAccessor.energy_balance
    ~pypsa.statistics.StatisticsAccessor.curtailment
    ~pypsa.statistics.StatisticsAccessor.capacity_factor
    ~pypsa.statistics.StatisticsAccessor.revenue
    ~pypsa.statistics.StatisticsAccessor.market_value

Usage
-----------------

To utilize the statistics module, instantiate the accessor from a PyPSA network object 
as follows:

.. doctest::

    >>> n = pypsa.Network()

You can then call specific methods from the ``statistics`` property to calculate various 
metrics, such as installed capacity or operational expenditure. For example:

.. doctest::

    >>> installed_capacity = n.statistics.installed_capacity()
    >>> installed_capacity
    Component  carrier
    Generator  gas        100.0
               wind        50.0
               solar       30.0
    Name: optimal_capacity, dtype: float64
    
    >>> opex = n.statistics.opex()
    >>> opex
    Component  carrier
    Generator  gas        1000.0
               wind          0.0
               solar         0.0
    Name: opex, dtype: float64


Have a look at the method's docstring for more details on the arguments.

For example, to calculate the capital expenditure (capex) as a sum for all components, 
you can use:

.. doctest::

    >>> n.statistics.capex(aggregate_groups='sum')

Similarly, to calculate the operational expenditure for all Link components, which 
connect to hydrogen (H2) buses:

.. doctest::

    >>> n.statistics.opex(comps=["Link"], bus_carrier="H2")

Statistic groupers
===================

Groupers can be used via the ``groupby`` argument in the statistic methods. 

All default groupers are defined in the :class:`pypsa.statistics.grouping.Groupers` 
class and currently included are, grouping by ..

* :meth:`carrier <pypsa.statistics.grouping.Groupers.carrier>`
* :meth:`bus_carrier <pypsa.statistics.grouping.Groupers.bus_carrier>`
* :meth:`name <pypsa.statistics.grouping.Groupers.name>`
* :meth:`bus <pypsa.statistics.grouping.Groupers.bus>`
* :meth:`country <pypsa.statistics.grouping.Groupers.country>`
* :meth:`location <pypsa.statistics.grouping.Groupers.location>`
* :meth:`unit <pypsa.statistics.grouping.Groupers.unit>`
* A list of registered groupers can be accessed via
    :meth:`pypsa.statistics.groupers.list_groupers <pypsa.statistics.grouping.Groupers.list_groupers>`

Custom groupers can be registered on module level via
:meth:`pypsa.statistics.groupers.add_grouper <pypsa.statistics.grouping.Groupers.add_grouper>`.
The key will be used as identifier in the ``groupby`` argument.

Usage
-----------------

.. doctest::
    
    >>> groupers = n.statistics.groupers
    >>> n.statistics.capex(groupby=groupers.carrier)
    carrier
    gas      10000.0
    wind      5000.0
    solar     3000.0
    Name: capex, dtype: float64
    
    >>> # or simply
    >>> n.statistics.capex(groupby='carrier')
    carrier
    gas      10000.0
    wind      5000.0
    solar     3000.0
    Name: capex, dtype: float64


Groupers can also be used to create multiindexed groupers. For example, to group by 
bus and carrier:

.. code-block:: python
    
    groupers = n.statistics.groupers
    n.statistics.capex(groupby=groupers['bus', 'carrier'])
    # or simply
    n.statistics.capex(groupby=['bus', 'carrier'])

.. autosummary::

    ~pypsa.statistics.grouping.Groupers.add_grouper
    ~pypsa.statistics.grouping.Groupers.list_groupers
    ~pypsa.statistics.grouping.Groupers.carrier
    ~pypsa.statistics.grouping.Groupers.bus_carrier
    ~pypsa.statistics.grouping.Groupers.name
    ~pypsa.statistics.grouping.Groupers.bus
    ~pypsa.statistics.grouping.Groupers.country
    ~pypsa.statistics.grouping.Groupers.location
    ~pypsa.statistics.grouping.Groupers.unit


Advanced Examples and Visualization
=======================================

In addition to basic usage, the statistics module offers advanced functionality for 
in-depth analysis and visualization of network metrics. Here are some advanced examples 
and visualization techniques:

1. **Comparative Analysis**: Users can compare different scenarios or network 
configurations by calculating metrics for each scenario and visualizing the results 
side by side. For example, compare the installed capacity of renewable energy sources 
in two different network models.

2. **Temporal Analysis**: Utilize the aggregate_time parameter to analyze temporal 
variations in network metrics. Plotting time series data can reveal patterns
and trends over time, such as seasonal variations in energy supply or demand.

3. **Geospatial Visualization**: If the network includes geospatial data, users can 
create maps to visualize the distribution of network components and metrics
geographically. This can be particularly useful for understanding spatial dependencies 
and identifying areas with high or low capacity utilization.

4. **Scenario Planning**: Explore different scenarios or what-if analyses by adjusting 
input parameters and observing the impact on network metrics. For example,
simulate the effect of increasing renewable energy penetration on curtailment and 
market value.

5. **Interactive Dashboards**: Develop interactive dashboards using visualization 
libraries like Plotly or Bokeh to allow users to dynamically explore network
metrics and drill down into specific details. Dashboards can provide a user-friendly 
interface for exploring complex network data.

Example Code Snippet:

.. doctest::
    
    >>> import matplotlib.pyplot as plt
    >>> # Calculate installed capacity
    >>> installed_capacity = n.statistics.installed_capacity().droplevel(0)
    >>> # Plot installed capacity by component type
    >>> installed_capacity.plot(kind='bar', figsize=(10, 6))
    >>> plt.title('Installed Capacity by Component Type')
    >>> plt.xlabel('Component Type')
    >>> plt.ylabel('Installed Capacity (MW)')
    >>> plt.xticks(rotation=45)
    >>> plt.grid(axis='y')
    >>> plt.tight_layout()
    >>> plt.savefig('statistics_advanced_usage.png')
    >>> plt.close()

.. figure:: ../img/statistics_advanced_usage.png
   :alt: Installed Capacity by Component Type
   :width: 100%
   
   Installed capacity by component type.

This code snippet calculates the installed capacity for each component type in the 
network and visualizes the results using a bar plot. Similar visualizations can
be created for other metrics, providing valuable insights into the network's composition
and characteristics.














# Using the statistics module in PyPSA

The `statistics` module is used to easily extract information from your networks. This is useful when inspecting your solved networks and creating first visualizations of your results.

With the `statistics` module, you can look at different metrics of your network. A list of the implemented metrics are:
    
- Capital expenditure
- Operational expenditure
- Installed capacities
- Optimal capacities
- Supply
- Withdrawal
- Curtailment
- Capacity Factor
- Revenue
- Market value
- Energy balance

Now lets look at an example.

```python
import matplotlib.pyplot as plt
import numpy as np

import pypsa
```

First, we open an example network we want to investigate.

```python
n = pypsa.examples.scigrid_de()
```

Lets run an overview of all statistics by calling:

```python
n.statistics().dropna()
```

So far the `statistics` are not so interesting, because we have not solved the network yet. We can only see that the network already has some installed capacities for different components.

You can see that `statistics` returns a `pandas.DataFrame`. The MultiIndex of the `DataFrame` provides the name of the network component (i.e. first entry of the MultiIndex, like *Generator, Line,...*) on the first index level. The `carrier` index level provides the carrier name of the given component. For example, in `n.generators`, we have the carriers *Brown Coal, Gas* and so on.

Now lets solve the network.

```python
n.optimize(n.snapshots[:4])
```

Now we can look at the `statistics` of the solved network.

```python
n.statistics().round(1)
```

As you can see there is now much more information available. There are still no capital expenditures in the network, because we only performed an operational optimization with this example network.

If you are interested in a specific metric, e.g. curtailment, you can run

```python
curtailment = n.statistics.curtailment()
curtailment[curtailment != 0]
```

Note that when calling a specific metric the `statistics` module returns a `pandas.Series`.
To find the unit of the data returned by `statistics`, you can call `attrs` on the `DataFrame` or `Series`.

```python
curtailment.attrs
```

So the unit of curtailment is given in `MWh`. You can also customize your request.

For this you have various options:
1. You can select the component from which you want to get the metric with the attribute `comps`. Careful, `comps` has to be a list of strings.

```python
n.statistics.supply(comps=["Generator"])
```

2. For metrics which have a time dimension, you can choose the aggregation method or decide to not aggregate them at all. Just use the `aggregate_time` attribute to specify what you want to do.

For example calculate the mean supply/generation per time step is

```python
n.statistics.supply(comps=["Generator"], aggregate_time="mean")
```

Or retrieve the supply time series by not aggregating the time series. 

```python
n.statistics.supply(comps=["Generator"], aggregate_time=False).iloc[:, :4]
```

3. You can choose how you want to group the components of the network and how to aggregate the groups. By default the components are grouped by their carriers and summed. However, you can change this by providing different `groupby` and `aggregate_groups` attributes.

```python
n.statistics.supply(comps=["Generator"], groupby=["bus"], aggregate_groups="max")
```

Now you obtained the maximal supply in one time step for every bus in the network.

The keys in the provided in the `groupby` argument is primarily referring to grouper functions registered in the `n.statistics.grouper` class. You can also provide a custom function to group the components. Let's say you want to group the components by the carrier and the price zone. The carrier grouping is already implemented in the `grouper` class, but the price zone grouping is not. You can provide a custom function to group the components by the price zone.

```python
# Create random number generator
rng = np.random.default_rng()

# Use new Generator API
n.buses["price_zone"] = rng.choice(
    [1, 2, 3, 4], size=n.buses.shape[0]
)  # add random price zones


def get_price_zone(n, c, port):
    bus = f"bus{port}"
    return n.static(c)[bus].map(n.buses.price_zone).rename("price_zone")


n.statistics.supply(
    comps=["Generator"], groupby=["carrier", get_price_zone], aggregate_groups="sum"
)
```

Often it is better when inspecting your network to visualize the tables. Therefore, you can easily make plots to analyze your results. For example the supply of the generators.

```python
n.statistics.supply(comps=["Generator"]).droplevel(0).div(1e3).plot.bar(
    title="Generator in GWh"
)
```

Or you could plot the generation time series of the generators.

```python
fig, ax = plt.subplots()
n.statistics.supply(comps=["Generator"], aggregate_time=False).droplevel(0).iloc[
    :, :4
].div(1e3).T.plot.area(
    title="Generation in GW",
    ax=ax,
    legend=False,
    linewidth=0,
)
ax.legend(bbox_to_anchor=(1, 0), loc="lower left", title=None, ncol=1)
```

Finally, we want to look at the energy balance of the network. The energy balance is not included in the overview of the statistics module. To calculate the energy balance, you can do

```python
n.statistics.energy_balance()
```

Note that there is now an additional index level called bus carrier. This is because an energy balance is defined for every bus carrier. The bus carriers you have in your network you can find by looking at `n.buses.carrier.unique()`. For this network, there is only one bus carrier which is AC and corresponds to electricity. However, you can have further bus carriers for example when you have a sector coupled network. You could have heat or CO $_2$ as bus carrier. Therefore, for many `statistics` functions you have to be careful about the units of the values and it is not always given by the `attr` object of the `DataFrame` or `Series`.

Finally, we want to plot the energy balance and the energy balance time series for electrcity which has the bus carrier AC. In a sector coupled network, you could also choose other bus carriers like H2 or heat. Note that in this example "-" represents the load in the system.

```python
fig, ax = plt.subplots()
n.statistics.energy_balance().loc[:, :, "AC"].groupby(
    "carrier"
).sum().to_frame().T.plot.bar(stacked=True, ax=ax, title="Energy Balance")
ax.legend(bbox_to_anchor=(1, 0), loc="lower left", title=None, ncol=1)
```

```python
fig, ax = plt.subplots()
n.statistics.energy_balance(aggregate_time=False).loc[:, :, "AC"].droplevel(0).iloc[
    :, :4
].groupby("carrier").sum().where(lambda x: np.abs(x) > 1).fillna(0).T.plot.area(
    ax=ax, title="Energy Balance Timeseries"
)
ax.legend(bbox_to_anchor=(1, 0), loc="lower left", title=None, ncol=1)
``` 