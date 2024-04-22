#######
Statistics Module
#######

Overview
=======================================

The ``pypsa.Network.statistics`` module provides an accessor to the ``pypsa.Network`` object, enabling users to efficiently calculate key network metrics.
This module is designed to simplify the analysis of network data by abstracting complex calculations into more accessible methods. It is particularly useful
for users who prefer to avoid manual calculations and directly obtain relevant statistical data about the network.

Available Metrics
=======================================

The module offers a variety of methods to calculate different metrics, which can be crucial for analyzing the performance and configuration of power systems
modeled in PyPSA. Below is a list of currently available metrics with brief descriptions and how to access them in Python:

* ``Installed Capacity``:
  * Computes the total installed capacity of components in the network, measured in Megawatts (MW).
  * Access in Python: ``n.statistics.installed_capacity()``

* ``Expanded Capacity``:
  * Calculates the capacity expansion of network components from a given base capacity, providing insights into network growth and scale-up, measured in MW.
  * Access in Python: ``n.statistics.expanded_capacity()``

* ``Optimal Capacity``:
  * Determines the total optimal capacity that components should ideally have based on model optimization, measured in MW.
  * Access in Python: ``n.statistics.optimal_capacity()``

* ``Capex (Capital Expenditure)``:
  * Evaluates the total capital investment required for the network components, which includes newly installed and existing assets, measured in the specified
  currency.
  * Access in Python: ``n.statistics.capex()``

* ``Opex (Operational Expenditure)``:
  * Calculates ongoing operational costs of network components, taking into account various operational parameters and their costs, measured in the specified
  currency.
  * Access in Python: ``n.statistics.opex()``

* ``Supply``:
  * Measures the total energy supply provided by the components of the network, which can vary depending on the carrier of the bus.
  * Access in Python: ``n.statistics.supply()``

* ``Withdrawal``:
  * Assesses the total energy withdrawn from network components, reflecting consumption or storage activities.
  * Access in Python: ``n.statistics.withdrawal()``

* ``Dispatch``:
  * Quantifies the energy dispatched by various components across the network, crucial for understanding the flow and allocation of energy.
  * Access in Python: ``n.statistics.dispatch()``

* ``Transmission``:
  * Calculates the energy transmitted between components, often critical for assessing the efficiency and reliability of the network.
  * Access in Python: ``n.statistics.transmission()``

* ``Curtailment``:
  * Measures the potential energy not utilized due to system constraints or operational decisions, reflecting the difference between possible and actual generation.
  * Access in Python: ``n.statistics.curtailment()``

* ``Capacity Factor``:
  * Indicates the ratio of the actual output of a component over a period to its potential output if it had operated at full capacity continuously.
  * Access in Python: ``n.statistics.capacity_factor()``

* ``Revenue``:
  * Estimates the revenue generated from the operation of network components, considering both inputs and outputs in the specified currency.
  * Access in Python: ``n.statistics.revenue()``

* ``Market Value``:
  * Calculates the market value of energy produced by network components, providing a financial metric that combines energy output with market prices.
  * Access in Python: ``n.statistics.market_value()``

* ``Energy Balance``:
  * Computes the energy balance for the network, comparing the total energy supply with the total energy withdrawal and accounting for losses.
  * Access in Python: ``n.statistics.energy_balance()``



Reference Documentation
=======================================

For further and updated information on all functions and methods available in the PyPSA statistics module, users  can access the PyPSA Statistics
documentation online at the following URL:

`PyPSA Statistics API Reference <https://pypsa.readthedocs.io/en/latest/api_reference.html#module-pypsa.statistics>`_


Further Examples
=======================================

For more advanced usage and customization options, users can refer to an example under the "Examples/Basic Usage" section:

`PyPSA Statistics Example <https://pypsa.readthedocs.io/en/latest/examples/statistics.html>`_

Usage
=======================================

To utilize the statistics module, instantiate the accessor from a PyPSA network object as follows:

.. code-block:: python

    n = pypsa.Network()
    stats = n.statistics

You can then call specific methods from the ``stats`` object to calculate various metrics, such as installed capacity or operational expenditure. For example:

.. code-block:: python

    installed_capacity = stats.installed_capacity()
    opex = stats.opex()


Customization and Parameters
=======================================

Each method supports various parameters that allow customization of the computations, such as selecting specific components, defining aggregation methods, and more.

* ``comps``: Specifies the components to consider for the calculation. By default, it considers all components in the network.
* ``aggregate_time``: Defines the type of aggregation when aggregating time series. Options include 'mean', 'sum', or False.
* ``aggregate_groups``: Determines how to aggregate groups of components. Default is 'sum'.
* ``groupby``: Specifies how to group the components. This parameter is particularly useful for grouping by different attributes of the components.
* ``at_port``: Indicates whether to compute the metric at the port level. By default, it is set to False.
* ``bus_carrier``: Filters components connected to buses with the specified carrier.
* ``nice_names``: Specifies whether to use user-friendly names for the components. Default is True.

For example, to calculate the capital expenditure (capex) as a sum for all components, you can use:

.. code-block:: python

    n.statistics.capex(aggregate_groups='sum')

Similarly, to calculate the operational expenditure for all Link components, which attend to the hydrogen (H2) bus:

.. code-block:: python

    n.statistics.opex(comps=["Link"], bus_carrier="H2")

Similarly, to calculate the operational expenditure for all Link components, which attend to the hydrogen (H2) bus:


Advanced Examples and Visualization
=======================================

In addition to basic usage, the statistics module offers advanced functionality for in-depth analysis and visualization of network metrics. Here are some
advanced examples and visualization techniques:

1. **Comparative Analysis**: Users can compare different scenarios or network configurations by calculating metrics for each scenario and visualizing the
results side by side. For example, compare the installed capacity of renewable energy sources in two different network models.

2. **Temporal Analysis**: Utilize the aggregate_time parameter to analyze temporal variations in network metrics. Plotting time series data can reveal patterns
and trends over time, such as seasonal variations in energy supply or demand.

3. **Geospatial Visualization**: If the network includes geospatial data, users can create maps to visualize the distribution of network components and metrics
geographically. This can be particularly useful for understanding spatial dependencies and identifying areas with high or low capacity utilization.

4. **Scenario Planning**: Explore different scenarios or what-if analyses by adjusting input parameters and observing the impact on network metrics. For example,
simulate the effect of increasing renewable energy penetration on curtailment and market value.

5. **Interactive Dashboards**: Develop interactive dashboards using visualization libraries like Plotly or Bokeh to allow users to dynamically explore network
metrics and drill down into specific details. Dashboards can provide a user-friendly interface for exploring complex network data.

Example Code Snippet:

.. code-block:: python

    import matplotlib.pyplot as plt

    # Calculate installed capacity
    installed_capacity = n.statistics.installed_capacity().droplevel(0)

    # Plot installed capacity by component type
    installed_capacity.plot(kind='bar', figsize=(10, 6))
    plt.title('Installed Capacity by Component Type')
    plt.xlabel('Component Type')
    plt.ylabel('Installed Capacity (MW)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

This code snippet calculates the installed capacity for each component type in the network and visualizes the results using a bar plot. Similar visualizations can
be created for other metrics, providing valuable insights into the network's composition and characteristics.
