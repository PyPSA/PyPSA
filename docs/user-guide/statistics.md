The `statistics` module is used to easily extract and calculate common metrics from a [Network][pypsa.Network]. This is useful for inspecting networks and creating first visualizations of your results.

It is accessed via the [`n.statistics`][pypsa.Network.statistics] property of any [pypsa.Network][] object.

## Metrics


Currently supported metrics are:
    
- [Capital expenditure][pypsa.statistics.StatisticsAccessor.capex]
- [Installed capital expenditure][pypsa.statistics.StatisticsAccessor.installed_capex]
- [Expanded capital expenditure][pypsa.statistics.StatisticsAccessor.expanded_capex]
- [Operational expenditure][pypsa.statistics.StatisticsAccessor.opex]
- [Installed capacities][pypsa.statistics.StatisticsAccessor.installed_capacity]
- [Expanded capacities][pypsa.statistics.StatisticsAccessor.expanded_capacity]
- [Optimal capacities][pypsa.statistics.StatisticsAccessor.optimal_capacity]
- [Supply][pypsa.statistics.StatisticsAccessor.supply]
- [Withdrawal][pypsa.statistics.StatisticsAccessor.withdrawal]
- [Curtailment][pypsa.statistics.StatisticsAccessor.curtailment]
- [Capacity Factor][pypsa.statistics.StatisticsAccessor.capacity_factor]
- [Revenue][pypsa.statistics.StatisticsAccessor.revenue]
- [Market value][pypsa.statistics.StatisticsAccessor.market_value]
- [Energy balance][pypsa.statistics.StatisticsAccessor.energy_balance]
- [System costs][pypsa.statistics.StatisticsAccessor.system_cost]
- [Marginal prices][pypsa.statistics.StatisticsAccessor.prices]
- [Transmission][pypsa.statistics.StatisticsAccessor.transmission]

Via the accessor you can call methods to calculate various metrics, such as installed capacity or operational expenditure. For example:

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
    Time aggregation automatically accounts for snapshot weightings.

### Output Formatting

- **`nice_names`**: Use nice names for carriers 
- **`drop_zero`**: Remove zero-value rows
- **`round`**: Round to decimal places

## Expressions

Next to the statistics module under [`n.statistics`][pypsa.Network.statistics], there is also an optimization expressions module under `n.optimize.expressions`. It provides similar functionality, but creates linopy expressions for the optimization model instead of calculating values from the network data.
