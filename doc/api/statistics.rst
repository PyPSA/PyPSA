###########
Statistics
###########

Statistic functions which can be called within a :class:`pypsa.Network` via
``n.statistics.func``. For example ``n.statistics.capex()``.

Statistic methods
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _source/

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

    
Statistic groupers
~~~~~~~~~~~~~~~~~~~

Groupers can be used in combination with the statistic methods. For example

.. code-block:: python
    
    groupers = n.statistics.groupers
    n.statistics.capex(groupby=groupers.get_carrier)

Or any other grouper could be used.

.. autosummary::
    :toctree: _source/

    ~pypsa.statistics.Groupers.get_carrier
    ~pypsa.statistics.Groupers.get_bus_and_carrier
    ~pypsa.statistics.Groupers.get_name_bus_and_carrier
    ~pypsa.statistics.Groupers.get_country_and_carrier
    ~pypsa.statistics.Groupers.get_carrier_and_bus_carrier
    ~pypsa.statistics.Groupers.get_bus_and_carrier_and_bus_carrier
    ~pypsa.statistics.Groupers.get_bus_unit_and_carrier
