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

Groupers can be used via the ``groupby`` argument in the statistic methods. 

.. code-block:: python
    
    groupers = n.statistics.groupers
    n.statistics.capex(groupby=groupers.carrier)
    # or simply
    n.statistics.capex(groupby='carrier')

All default groupers are defined in the :class:`pypsa.statistics.grouping.Groupers` 
class and currently included are, grouping by ..

* .. :meth:`carrier <pypsa.statistics.grouping.Groupers.carrier>`
* .. :meth:`bus_carrier <pypsa.statistics.grouping.Groupers.bus_carrier>`
* .. :meth:`name <pypsa.statistics.grouping.Groupers.name>`
* .. :meth:`bus <pypsa.statistics.grouping.Groupers.bus>`
* .. :meth:`country <pypsa.statistics.grouping.Groupers.country>`
* .. :meth:`unit <pypsa.statistics.grouping.Groupers.unit>`

Custom groupers can be registered on module level via
:meth:`pypsa.statistics.groupers.add_grouper <pypsa.statistics.grouping.Groupers.add_grouper>`.
The key will be used as identifier in the ``groupby`` argument.

Groupers can also be used to create multiindexed groupers. For example, to group by 
bus and carrier:

.. code-block:: python
    
    groupers = n.statistics.groupers
    n.statistics.capex(groupby=groupers['bus', 'carrier'])
    # or simply
    n.statistics.capex(groupby=['bus', 'carrier'])

.. autosummary::
    :toctree: _source/

    ~pypsa.statistics.grouping.Groupers.add_grouper
    ~pypsa.statistics.grouping.Groupers.list_groupers
    ~pypsa.statistics.grouping.Groupers.carrier
    ~pypsa.statistics.grouping.Groupers.bus_carrier
    ~pypsa.statistics.grouping.Groupers.name
    ~pypsa.statistics.grouping.Groupers.bus
    ~pypsa.statistics.grouping.Groupers.country
    ~pypsa.statistics.grouping.Groupers.unit

