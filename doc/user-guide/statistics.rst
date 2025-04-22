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