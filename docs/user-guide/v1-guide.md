# What's new in PyPSA v1.0

**PyPSA v1.0** is here and brings a list of exciting new features. Alongside this release, the documentation has also been completely updated, redesigned and moved to a new URL: [`https://docs.pypsa.org`](https://docs.pypsa.org). Take a look at the new documentation to find out about the latest features, as well as those that have not yet been well documented.

!!! warning
    While breaking changes have been kept to a minimum with this release, there are still some [changes to be aware of](#breaking-changes). Specially when deprecation warnings haven not been addressed yet.

## Features

### Stochastic Optimization

PyPSA now supports **two-stage stochastic programming** with scenario trees out of the box, which allows users to optimize investment decisions (first-stage) that are robust across multiple possible future realizations (scenarios) of uncertain parameters. See [:material-bookshelf: User Guide](optimization/stochastic-optimization.md) and [:material-notebook-multiple: Example Notebook](../examples/stochastic-optimization.ipynb).

```python
>>> n = pypsa.examples.ac_dc_meshed()
>>> n.set_scenarios({"low": 0.4, "med": 0.3, "high": 0.3})
>>> n
Stochastic PyPSA Network 'AC-DC-Meshed'
---------------------------------------
Components:
 - Bus: 27
 ...
 - Load: 18
Snapshots: 10
Scenarios: 3
```

### Plotting Module
Any **network metric** like `energy_balance` or `installed_capacity` can now be plotted as a line, bar, area, map and more, to allow easier exploration and better tooling to create plots. The equivalent plotting methods extend parameters from the statistics module and therefore use the same logic to be easily adaptable. Also **interactive plots** can be create for all metric/ plot type combinations. See [:material-bookshelf: User Guide](plotting.md) and [:octicons-code-16: API Reference](/api/networks/plot.md).

```python
>>> n.statistics.energy_balance.iplot.area()
```

<div style="width: 100%; height: 400px; overflow: hidden;">
    <iframe src="../../assets/interactive-plots/ac_dc_meshed-energy_balance-area_iplot.html"
            width="100%" height="100%" frameborder="0" 
            style="border: 1px solid #ccc; transform: scale(0.6); transform-origin: 0 0;">
    </iframe>
</div>

### Network Collection
A new object called [`NetworkCollection`][pypsa.Network] has been added to the library. It allows users to store multiple networks in a single object and perform operations on them. See [:material-bookshelf: User Guide](network-collection.md) and [:material-notebook-multiple: Example Notebook](/examples/network-collection.ipynb).

```python
>>> pypsa.NetworkCollection([n_base, n_reference])
NetworkCollection
-----------------
Networks: 2
Index name: 'network'
Entries: ['Base Case', 'Reference Case']
```


### Components Class
PyPSA [`Components`][pypsa.components.Components] are an intermediate layer between the network object and the components data. They allow for a lot of functionality that would otherwise always have to be reimplemented on the underlying `pandas` DataFrames. Using them removes the need for a lot of boilerplate code without changing the `pandas`-based structure. With the release of PyPSA `v1.0`, they are officially released alongside a new optional and breaking API. See the dedicated [:material-bookshelf: User Guide](../user-guide/components.md).

```python
>>> n.generators
                        bus control type    p_nom  ...  
Generator                                          ...                                                              
.                
Frankfurt Wind    Frankfurt      PQ         110.0  ...                
Frankfurt Gas     Frankfurt      PQ       80000.0  ...                

[6 rows x 37 columns]

# Opt-in to new components API
>>> pypsa.options.legacy_components_api = False

# n.generators will now return a Components object
>>> n.generators
'Generator' Components
----------------------
Attached to PyPSA Network 'AC-DC-Meshed'
Components: 6

# Static data and more is still available
>>> n.generators.static
                        bus control type    p_nom  ...  
Generator                                          ...                                                              
.                
Frankfurt Wind    Frankfurt      PQ         110.0  ...                
Frankfurt Gas     Frankfurt      PQ       80000.0  ...                

[6 rows x 37 columns]
```

## Breaking Changes
While PyPSA has been stable for a while now, version `v1.0` is the first stable release. This means that future versions will introduce new features and improvements, but we will maintain full backward compatibility up until version `v2.0`. For more details check the [upgrade guide](../home/installation.md#upgrading).

- Inactive components (e.g. when using the `active` attribute or when `build_time` and `lifetime` never match) are now excluded from the optimization model entirely. You can list them via [pypsa.Components.inactive_assets][]. This has no effect on any of the results, but if you access the linopy model directly, you will see that the number of variables and constraints might be reduced.

- Index names of all pandas dataframes used to store components data (e.g. `n.generators` and `n.generators_t`) have been changed. The axis which lists the components is now called `name` across all component types, instead of the previous component type name (e.g. `Generator`), to align with the newly introduced xarray view ([pypsa.Components.da][]).

- When retrieving a list of extendable, fixed or committable components, no suffix (e.g. `"-ext"` or `"-fix"`) is added anymore. This was for example previously be possible via `n.get_extendable_i` and now via [pypsa.Components.extendables].

- All features that were announced as deprecated in previous versions have now been removed. If those warnings have not been addressed yet, you will not be able to use PyPSA `v1.0`.

!!! warning

    If you are unsure if you are still using any deprecated features, first install the latest version `v0.35.1` and resolve all warnings. You can easily catch them all by raising them as errors. Just add the following to the top of your script:
    ```python
    import warnings
    warnings.filterwarnings('error', category=DeprecationWarning, module='pypsa')
    ```

### Major future change
A major breaking change will be the new Components Class API, as described [above](#new-components-class-api). This functionality is opt-in with version `v1.0` and both the old and the new API are fully supported. 

A detailed introduction can be found in the dedicated [user guide section](../user-guide/components.md#new-components-class-api). Please have a look and provide feedback. The new API is planned to be the default in version `v2.0`.