<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

The `statistics` module is used to extract and calculate common metrics from a [Network][pypsa.Network]. This module is intended to simplify the process of inspecting networks and creating first visualizations of your results.

It is accessed via the [`n.statistics`][pypsa.Network.statistics] property of any [pypsa.Network][] object.

!!! info "Shorthand Alias"

    For convenience, `n.stats` is available as a shorthand alias for `n.statistics`. Both can be used interchangeably throughout your code.

## Metrics


Currently supported metrics are:

- [Capital expenditure][pypsa.statistics.StatisticsAccessor.capex]: The capital expenditure of all components.
- [Installed capital expenditure][pypsa.statistics.StatisticsAccessor.installed_capex]: The capital expenditure of all components before optimization.
- [Expanded capital expenditure][pypsa.statistics.StatisticsAccessor.expanded_capex]: The capital expenditure of all components added during optimization.
- [Overnight investment cost][pypsa.statistics.StatisticsAccessor.overnight_cost]: The overnight investment costs (excluding FOM) of all components.
- [Fixed O&M][pypsa.statistics.StatisticsAccessor.fom]: The fixed operation and maintenance costs of all components.
- [Operational expenditure][pypsa.statistics.StatisticsAccessor.opex]: The operational expenditure of all components.
- [Installed capacities][pypsa.statistics.StatisticsAccessor.installed_capacity]: The capacities of all components before optimization.
- [Expanded capacities][pypsa.statistics.StatisticsAccessor.expanded_capacity]: The capacities of all components added during optimization.
- [Optimal capacities][pypsa.statistics.StatisticsAccessor.optimal_capacity]: The total capacities of all components after optimization.
- [Supply][pypsa.statistics.StatisticsAccessor.supply]: The energy supplied by all components.
- [Withdrawal][pypsa.statistics.StatisticsAccessor.withdrawal]: The energy withdrawn by all components.
- [Curtailment][pypsa.statistics.StatisticsAccessor.curtailment]: The energy curtailed in all components.
- [Capacity Factor][pypsa.statistics.StatisticsAccessor.capacity_factor]: The capacity factor / utilization rate of all components.
- [Revenue][pypsa.statistics.StatisticsAccessor.revenue]: The revenue received by all components.
- [Market value][pypsa.statistics.StatisticsAccessor.market_value]: The market value of all components.
- [Energy balance][pypsa.statistics.StatisticsAccessor.energy_balance]: The energy balance of the network across all carriers and snapshots.
- [System costs][pypsa.statistics.StatisticsAccessor.system_cost]: The total system costs after optimization, including capital and operational expenditure.
- [Marginal prices][pypsa.statistics.StatisticsAccessor.prices]: The marginal prices at the buses for all snapshots.
- [Transmission][pypsa.statistics.StatisticsAccessor.transmission]: The energy transmitted through transmission components (links, lines, transformers connecting to buses of the same carrier).

Congestion rent, capture rate, profit, and producer and consumer surplus follow from these metrics, see [Derived Metrics](#derived-metrics).

These metrics can be calculated using the `n.statistics` accessor, for instance:

``` py
>>> installed_capacity = n.statistics.installed_capacity()
>>> installed_capacity  # doctest: +ELLIPSIS
component  carrier
Generator  gas        150000.0
...
dtype: float64

>>> opex = n.statistics.opex()
>>> opex  # doctest: +SKIP
Component  carrier
Generator  gas        1000.0
            wind          0.0
            solar         0.0
Name: opex, dtype: float64
```

## Parameters

Most statistics methods accept common parameters to control filtering, grouping, and output formatting.

!!! warning

    Not all metrics support all parameters. Please check the function docstring for details.

### Filtering

Select which components to include:

- **`components`**: List of component types to include (e.g., `["Generator", "StorageUnit"]`)
- **`carrier`**: Filter by component carrier (e.g., `"wind"` or `["wind", "solar"]`)
- **`bus_carrier`**: Filter by connected bus carrier (e.g., `"AC"`)

!!! tip

    All filtering can also be done via pandas indexing after calling the statistic method. But using the parameters is more efficient, as it avoids unnecessary calculations.

``` py
# Only generators with wind carrier on AC buses
>>> n.statistics.supply(components=["Generator"], carrier="wind", bus_carrier="AC")  # doctest: +SKIP
```

### Grouping

Control how results are grouped and aggregated:

**`groupby`**: Group by attributes or custom functions.

Built-in groupers include:

- `"carrier"` - Group by component carrier
- `"bus_carrier"` - Group by carrier of connected bus
- `"bus"` - Group by connected bus
- `"country"` - Group by country of bus location
- `"location"` - Group by location of bus
- `"name"` - Group by component name
- `"unit"` - Group by unit

You can also use component attributes directly (e.g., `"type"`, `"p_nom"`), provide custom functions, or combine multiple groupers for multi-index grouping.

**`groupby_method`**: Aggregation method like `"sum"`, `"mean"`, `"max"`. Default is `"sum"`

#### Registering custom groupers
You can register custom groupers to use them by name in the `groupby` argument:

``` py
# Define a custom grouper function
def group_by_voltage(n, c, port=""):
    """Group components by voltage level of connected bus."""
    bus = f"bus{port}"
    buses = n.c[c].static[bus]
    voltage = n.c.buses.static.v_nom.rename("voltage")
    return buses.map(voltage)

# Register the grouper on module level
pypsa.statistics.groupers.add_grouper("voltage", group_by_voltage)

# Use it by name in any statistics method
n.statistics.installed_capacity(groupby="voltage")

# Or access it as attribute
n.statistics.supply(groupby=pypsa.statistics.groupers.voltage)
```

Custom grouper functions must:

- Accept arguments: `n` (Network), `c` (component name), `port` (optional), `nice_names` (optional)
- Return a `pd.Series` with the same length as the component index

### Time Aggregation

For time-varying metrics:

- **`groupby_time`**: Method to aggregate over time (`"sum"`, `"mean"`, etc.) or `False` to keep time series

``` py
>>> # Average supply per time step (empty before solving the network)
>>> n.statistics.supply(groupby_time="mean")
Series([], dtype: float64)

>>> # Get full time series (NaN until dispatch is solved)
>>> n.statistics.supply(groupby_time=False).head()  # doctest: +NORMALIZE_WHITESPACE
snapshot           2015-01-01 00:00:00  ...  2015-01-01 09:00:00
component carrier                       ...
Generator gas                      NaN  ...                  NaN
          wind                     NaN  ...                  NaN
Load      load                     NaN  ...                  NaN
<BLANKLINE>
[3 rows x 10 columns]
```

!!! note
    Time aggregation automatically accounts for snapshot weightings when using `"sum"` or `"mean"`. Other aggregation methods like `"std"` or `"median"` do not apply snapshot weightings.

### Output Formatting

- **`nice_names`**: Use nice names for carriers
- **`drop_zero`**: Remove zero-value rows
- **`round`**: Round to decimal places

## Derived Metrics

Some common market metrics have no method of their own, because one or two existing statistics give them. The examples use the solved network, with one investment period and no scenarios. The [last section](#investment-periods-and-scenarios) adapts them to other cases.

``` py
>>> _ = n.optimize(log_to_console=False)
```

### Congestion rent

The congestion rent, or merchandising surplus, arises when transmission limits separate prices. It has two standard definitions: for one branch, its flow times the price difference between its ends; for the whole system, what consumers pay minus what producers receive. The rent of a branch is zero when the prices of its start and end buses are equal or when no power flows. In PyPSA, the [revenue][pypsa.statistics.StatisticsAccessor.revenue] of a branch is the value of the energy it delivers minus the value of the energy it withdraws, which is the first definition:

``` py
>>> n.statistics.revenue(components=["Line", "Link"])
component  carrier
Line       AC         609.62732
Link       DC         798.60693
dtype: float64
```

As the revenues of all components cancel at every bus, the sum over all branches also gives the second definition.

!!! note "Edge cases"

    - A single branch can earn a negative rent, for instance when loop flows push power from an expensive to a cheaper bus. The sum over all branches is non-negative.
    - A branch need not be congested itself to earn a rent. Congestion elsewhere in the network suffices to separate the prices at its ends.
    - For links with losses, the rent is the price difference times the withdrawn flow minus the losses valued at the price of the receiving bus.
    - For links between buses of different carriers, such as electrolysers, the rent is the margin on the price spread between the carriers. Exclude them with the `carrier` filter.

### Capture rate

The capture rate, or value factor, of a technology is the average price it earns relative to the average price over time[^1]. In PyPSA, divide its [market value][pypsa.statistics.StatisticsAccessor.market_value] by the time-weighted average [price][pypsa.statistics.StatisticsAccessor.prices] of the buses with the same `bus_carrier`:

``` py
>>> market_value = n.statistics.market_value(components="Generator", bus_carrier="AC")
>>> base_price = n.statistics.prices(bus_carrier="AC", groupby="bus_carrier", weighting="time")
>>> (market_value / base_price["AC"]).round(2)
carrier
gas     2.83
wind    1.07
dtype: float64
```

### Profit

The profit of an asset is its revenue minus all its costs, including the annualised capital cost. In PyPSA, subtract the [system cost][pypsa.statistics.StatisticsAccessor.system_cost] from the revenue at all ports. The `fill_value=0` keeps components without costs, such as loads:

``` py
>>> profit = n.statistics.revenue().sub(n.statistics.system_cost(), fill_value=0)
>>> profit.round()
component  carrier
Generator  gas         2178292.0
           wind              0.0
Line       AC                0.0
Link       DC                0.0
Load       load      -20619313.0
dtype: float64
```

Profit is in the units of the objective function. As `capital_cost` is annualised, it is a yearly value only if the snapshot weightings add up to one year.

In a long-run equilibrium, the zero-profit rule holds: an extendable asset at an interior optimum of a linear model earns exactly its costs[^2]. Under an emission cap, emitters break even only after paying the CO2 price, the shadow price of the cap, for their emissions. The system cost excludes this payment, so the profit above contains it. In this network, the gas generators are the only emitters and their emissions fill the cap:

``` py
>>> co2_limit = n.c.global_constraints.static.loc["co2_limit"]
>>> round(-co2_limit.mu * co2_limit.constant)
2178292
```

!!! warning "Exceptions to the zero-profit rule"

    - Fixed capacities and binding capacity limits, such as `p_nom_min` and `p_nom_max`, earn a positive or negative rent.
    - With a `marginal_cost_quadratic` $c_q$, the price exceeds the average operational cost, and an extendable asset earns the rent $\sum_t w_t c_q p_t^2$ on top of its costs.
    - With unit commitment or modular capacities, the problem is not convex, and prices need not recover costs.

### Producer and consumer surplus

The producer surplus, or producer rent, is what producers receive minus their variable costs. In PyPSA, it is the revenue minus the [operational expenditure][pypsa.statistics.StatisticsAccessor.opex]. It equals the profit plus the capital expenditure, and for branches without operational costs it is the congestion rent:

``` py
>>> revenue = n.statistics.revenue()
>>> producer_surplus = revenue.sub(n.statistics.opex(), fill_value=0).drop("Load")
>>> producer_surplus.round()
component  carrier
Generator  gas         2279124.0
           wind       18329600.0
Line       AC              610.0
Link       DC              799.0
dtype: float64
```

The consumer surplus, or consumer rent, is what consumers would be willing to pay for their energy minus what they pay. Loads with a fixed demand have no willingness to pay in the model. As a proxy, assume one above the highest price. If the model uses [load shedding][pypsa.optimization.optimize.OptimizationAccessor.add_load_shedding], take its cost, the value of lost load. The revenue of loads is negative; it is their payment:

``` py
>>> willingness_to_pay = 3000  # €/MWh, an assumption
>>> demand = n.statistics.withdrawal(components="Load")
>>> consumer_surplus = willingness_to_pay * demand + revenue["Load"]
>>> consumer_surplus.round()
carrier
load    77023571.0
dtype: float64
```

!!! warning

    The level of this proxy scales with the assumed willingness to pay. Compare it only between scenarios with the same demand, where the assumption cancels.

!!! info "Elastic demand"

    Elastic demand needs no proxy. Model it as a generator with `sign=-1` and a negative `marginal_cost`, the willingness to pay, as in the [data centre example](../examples/data-centre-investment.ipynb). A positive `marginal_cost_quadratic` makes the willingness to pay fall with consumption. Its revenue minus its operational expenditure, the formula of the producer surplus, is then its consumer surplus.

The welfare is the sum of consumer surplus, producer surplus and congestion rent. As the revenues of all components cancel at every bus, it equals the value of the demand minus the operational expenditure:

``` py
>>> welfare = willingness_to_pay * demand.sum() - n.statistics.opex().sum()
>>> round(consumer_surplus.sum() + producer_surplus.sum() - welfare)
0
```

### Investment periods and scenarios

With investment periods, statistics have one column per period. The system cost, capital and operational expenditure of each period are weighted by `n.investment_period_weightings.objective`, which discounts them, but the revenue is not. Weight the revenue, and the value of the demand, in the same way before you subtract costs. An asset that lives over several periods then breaks even over its lifetime, not in each period:

``` py
>>> weights = n.investment_period_weightings.objective
>>> revenue = n.statistics.revenue().mul(weights, level="period")  # doctest: +SKIP
>>> profit = revenue.sub(n.statistics.system_cost(), fill_value=0).sum(axis=1)  # doctest: +SKIP
```

!!! warning

    The [prices][pypsa.statistics.StatisticsAccessor.prices] statistic averages over all periods, so the capture rate recipe gives no value per period.

With scenarios, statistics have one row per scenario, and the capital expenditure repeats in each of them. Weight scenarios by `n.scenario_weightings.weight` for expected values. For the capture rate, divide by the base price of each scenario:

``` py
>>> base_price = base_price.xs("AC", level="bus_carrier")  # doctest: +SKIP
>>> market_value.div(base_price, level="scenario")  # doctest: +SKIP
```

[^1]: L. Hirth (2013), [The market value of variable renewables](https://doi.org/10.1016/j.eneco.2013.02.004), Energy Economics 38, 218–236.
[^2]: T. Brown, L. Reichenberg (2021), [Decreasing market value of variable renewables can be avoided by policy action](https://doi.org/10.1016/j.eneco.2021.105354), Energy Economics 100, 105354.

## Expressions

Next to the statistics module under [`n.statistics`][pypsa.Network.statistics], there is also an experimental optimization expressions module under `n.optimize.expressions`. It provides similar functionality, but creates `linopy` expressions for the optimization model instead of calculating values from the network data.
