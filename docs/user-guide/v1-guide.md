# PyPSA Version 1.0

With the release of PyPSA 1.0, the documentation has been completely overhauled and moved to the new URL: [https://docs.pypsa.org](https://docs.pypsa.org). It is worth taking a look at the new documentation to find out about new features and those that have not yet been well documented so far.

## Features

### Stochastic Optimization

PyPSA now supports **two-stage stochastic programming** with scenario trees out of the box, which allows users to optimize investment decisions (first-stage) that are robust across multiple possible future realizations (scenarios) of uncertain parameters. See [:material-bookshelf: User Guide](stochastic-optimization.md) and [:material-notebook-multiple: Example Notebook](stochastic-optimization.ipynb).

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


### New Components Class API
PyPSA [`Components`][pypsa.components.Components] are an intermediate layer between the network object and components data and allow for a lot of functionality, which otherwise would always be reimplemented on the underlying pandas dataframes. They allow to remove a lot of boilerplate code, while not changing the underlying pandas-based data structure. While they have been available for a while, PyPSA 1.0 introduces a new API to use them. See [:material-bookshelf: User Guide](components.md) and [:material-notebook-multiple: Example Notebook](/examples/components.ipynb).

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
