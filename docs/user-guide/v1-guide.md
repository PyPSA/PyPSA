<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# What's new in PyPSA v1.0

**PyPSA v1.0** is here and brings a range of new features. Alongside this release, the documentation has also been completely updated, redesigned and moved to a new URL: [`https://docs.pypsa.org`](https://docs.pypsa.org). Take a look at the new documentation to find out about the latest features, as well as those that have not yet been well documented.

!!! warning
    While breaking changes have been kept to a minimum with this release, there are still some [changes to be aware of](#breaking-changes). Especially, when deprecation warnings from previous releases have not been addressed yet.

## Features

### Stochastic Optimization

PyPSA now supports **two-stage stochastic programming** with scenario trees out of the box, which allows users to optimize investment decisions (first-stage) that are robust across multiple possible future realizations (scenarios) of uncertain parameters. In addition to the default risk-neutral formulation, a risk-averse formulation using the Conditional Value at Risk (CVaR) measure is also supported. See <!-- md:guide optimization/stochastic.md --> and [:material-notebook-multiple: Example Notebook](../examples/stochastic-optimization.ipynb).

``` py
>>> n_stoch = pypsa.examples.ac_dc_meshed()
>>> n_stoch.set_scenarios({"low": 0.4, "med": 0.3, "high": 0.3})
>>> n_stoch
Stochastic PyPSA Network 'AC-DC-Meshed'
---------------------------------------
Components:
 - Bus: 27
 - Carrier: 18
 - Generator: 18
 - GlobalConstraint: 3
 - Line: 21
 - Link: 12
 - Load: 18
Snapshots: 10
Scenarios: 3
```

### Plotting Module
#### Statistics Plotting
Any **network metric** like `n.statistics.energy_balance()` or `n.statistics.optimal_capacity()` can now be plotted as a line, bar, area charts or maps. The equivalent plotting methods extend parameters from the statistics module and therefore use the same logic. Also **interactive plots** can be created for all metrics and plot types. See [:material-bookshelf: User Guide](plotting/charts.md) and [:octicons-code-16: API Reference](../api/networks/plot.md).

``` py
>>> n_stoch.statistics.energy_balance.iplot.area()  # doctest: +SKIP
```

<div style="width: 100%; height: 540px; overflow: hidden;">
    <iframe src="../../assets/interactive/ac_dc_meshed-energy_balance-area_iplot.html"
            width="100%" height="100%" frameborder="0"
            style="border: 0px solid #ccc; transform: scale(1); transform-origin: 0 0;">
    </iframe>
</div>

#### Interactive Maps
Next to static map plotting with `n.plot()`, networks can now be rendered on a map, interactively. With `n.explore()`, you can explore the location of all components, including buses, lines, links, transformers, their component attributes and map results or other properties to the bus sizes, branch widths, colors, etc. Calling the method returns a standard `pydeck.Deck` object than can be layered on top of other `pydeck.Deck` objects (see https://deckgl.readthedocs.io/en/latest/layer.html). They can also be exported in self-contained HTML files for sharing. See [:material-bookshelf: User Guide](plotting/explore.ipynb) and [:octicons-code-16: API Reference](../api/networks/plot.md)

``` py
>>> n.explore()  # doctest: +SKIP
```

<div style="width: 100%; height: 800px; overflow: hidden;">
    <iframe src="https://bxio.ng/assets/html/scigrid-interactive-map"
            width="100%" height="100%" frameborder="0"
            style="border: 0px solid #ccc; transform: scale(1); transform-origin: 0 0;">
    </iframe>
</div>


### Network Collection
A new object called [`NetworkCollection`][pypsa.NetworkCollection] has been added to the library. It allows users to store multiple networks in a single object and perform operations on them. See [:material-bookshelf: User Guide](../user-guide/collection.md) and[:octicons-code-16: API Reference](../api/networks/collection.md).

``` py
>>> n_base = pypsa.examples.ac_dc_meshed() # docs-hide
>>> n_base.name = 'Base Case' # docs-hide
>>> n_reference = pypsa.examples.ac_dc_meshed() # docs-hide
>>> n_reference.name = 'Reference Case' # docs-hide
>>> pypsa.NetworkCollection([n_base, n_reference])
NetworkCollection
-----------------
Networks: 2
Index name: 'network'
Entries: ['Base Case', 'Reference Case']
```


### Components Class
PyPSA [`Components`][pypsa.Components] are a new intermediate layer between the network object and the components data. They provide various auxiliary functions to reduce the need for boilerplate code when working with PyPSA networks without changing the underlying `pandas`-based structure. With the release of PyPSA `v1.0`, they are officially released and complemented by a new optional breaking API. See [:material-bookshelf: User Guide](../user-guide/components.md) and [:octicons-code-16: API Reference](../api/components/components.md).

``` py
>>> n = pypsa.examples.ac_dc_meshed()
>>> n.generators
                        bus control  ... weight  p_nom_opt
name                                 ...
Manchester Wind  Manchester      PQ  ...    1.0        0.0
Manchester Gas   Manchester      PQ  ...    1.0        0.0
Norway Wind          Norway      PQ  ...    1.0        0.0
Norway Gas           Norway      PQ  ...    1.0        0.0
Frankfurt Wind    Frankfurt      PQ  ...    1.0        0.0
Frankfurt Gas     Frankfurt      PQ  ...    1.0        0.0
<BLANKLINE>
[6 rows x 42 columns]

# Opt-in to new components API
>>> pypsa.options.api.new_components_api = True

# n.generators will now return a Components object
>>> n.generators
'Generator' Components
----------------------
Attached to PyPSA Network 'AC-DC-Meshed'
Components: 6

# Static data and more is still available
>>> n.generators.static
                        bus control  ... weight  p_nom_opt
name                                 ...
Manchester Wind  Manchester      PQ  ...    1.0        0.0
Manchester Gas   Manchester      PQ  ...    1.0        0.0
Norway Wind          Norway      PQ  ...    1.0        0.0
Norway Gas           Norway      PQ  ...    1.0        0.0
Frankfurt Wind    Frankfurt      PQ  ...    1.0        0.0
Frankfurt Gas     Frankfurt      PQ  ...    1.0        0.0
<BLANKLINE>
[6 rows x 42 columns]

>>> pypsa.options.api.new_components_api = False
```

## Breaking Changes
While PyPSA has been stable for a while now, version `v1.0` is the first stable release. This means that future versions will introduce new features and improvements, but we will maintain full backward compatibility up until version `v2.0`. For more details check the [upgrade guide](../home/installation.md#upgrading).

- Inactive components (e.g. when using the `active` attribute or when `build_time` and `lifetime` never match) are now excluded from the optimization model entirely. You can list them via [pypsa.Components.inactive_assets][]. This has no effect on any of the results, but if you access the `linopy` model directly, you will see that the number of variables and constraints might be reduced.

- Index names of all pandas dataframes used to store components data (e.g. `n.generators` and `n.generators_t`) have been changed. The axis which lists the components is now called `name` across all component types, instead of the previous component type name (e.g. `Generator`), to align with the newly introduced xarray view ([pypsa.Components.da][]).

- When retrieving a list of extendable, fixed or committable components, no suffix (e.g. `"-ext"` or `"-fix"`) is added anymore. This was previously be accessible via `n.get_extendable_i()` and now via [pypsa.Components.extendables][].

- The default values for `cyclic_state_of_charge_per_period` (StorageUnit) and `e_cyclic_per_period` (Store) have been changed from `True` to `False`. This is to be more consistent with single investment period optimization where cycling behavior defaults to `False`. Users who work with multi-investment period optimizations and want per-period cycling behavior must now explicitly set these attributes to `True`.

- All features that were announced as deprecated in previous versions have now been removed. If those warnings have not been addressed yet, you will not be able to use PyPSA `v1.0`.

!!! warning

    If you are unsure if you are still using any deprecated features, first install the latest version `v0.35.2` (also see [upgrade guide](../home/installation.md#upgrading)) and resolve all warnings. You can catch them all by raising them as errors. Just add the following to the top of your script:
    ``` py
    >>> import warnings
    >>> warnings.filterwarnings('error', category=DeprecationWarning, module='pypsa')
    ```
- The new Components Class API is a major breaking change. It is opt-in with version `v1.0` and both the old and the new API are fully supported. A detailed introduction can be found in the dedicated [user guide section](../user-guide/components.md#new-components-class-api). Please have a look and provide feedback. The new API is planned to be the default in version `v2.0`.
