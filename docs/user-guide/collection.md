<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

As the name suggests, a [NetworkCollection][pypsa.NetworkCollection] is a collection of 
multiple networks. It provides a convenient way to manage and analyse multiple networks 
simultaneously, access their combined data, and generate combined statistics and plots.

!!! example "Under Active Development"

    NetworkCollections have been introduced in <!-- md:badge-version v0.35.0 --> and will be further extended in future releases.

## Create a collection
A network collection is simply a container that references multiple networks and wraps 
around them. 

Let's take two example networks and put them into a collection:

``` py
>>> n = pypsa.examples.ac_dc_meshed() # docs-hide
>>> _ = n.optimize() # docs-hide
>>> n # doctest: +ELLIPSIS
PyPSA Network 'AC-DC-Meshed'
...
>>> n_shuffled_load # Same network as n but with shuffled load time series
PyPSA Network 'AC-DC-Meshed-Shuffled-Load'
------------------------------------------
Components:
 - Bus: 9
 - Carrier: 6
 - Generator: 6
 - GlobalConstraint: 1
 - Line: 7
 - Link: 4
 - Load: 6
 - SubNetwork: 3
Snapshots: 10
<BLANKLINE>
>>> nc = pypsa.NetworkCollection([n, n_shuffled_load])
>>> nc
NetworkCollection
-----------------
Networks: 2
Index name: 'network'
Entries: ['AC-DC-Meshed', 'AC-DC-Meshed-Shuffled-Load']
```

If no index is passed, the collection will automatically assign names based on the 
network's [`name`][pypsa.Network.name] attribute or a default name if that is not set. 
But it is also possible to assign a custom index. Or even a MultiIndex can be used, which 
allows for custom grouping (see [below](#functionality)). 

``` py
>>> nc = pypsa.NetworkCollection(
...     [n, n_shuffled_load],
...     index=pd.Index(["network1", "network2"], name='custom-dim')
... )
>>> nc
NetworkCollection
-----------------
Networks: 2
Index name: 'custom-dim'
Entries: ['network1', 'network2']
>>> nc.networks
custom-dim
network1                  PyPSA Network 'AC-DC-Meshed'
network2    PyPSA Network 'AC-DC-Meshed-Shuffled-Load'
dtype: object
```

!!! warning

    It is important to note that the collection contains **a reference to the networks**, 
    not a copy. Therefore, any changes made to the individual networks will be reflected 
    in the collection and vice versa.

### Network dimension compatibility
Currently networks with the same dimensions should be used in a collection. E.g. for 
comparing different scenarios of the same system. That means all dimensions should align: 
[`n.snapshots`][pypsa.Network.snapshots] with [`n.periods`][pypsa.Network.periods] and 
[`n.timesteps`][pypsa.Network.timesteps], [`c.names`][pypsa.Components.names] 
across all components as well as [`n.scenarios`][pypsa.Network.scenarios] for stochastic 
networks. If dimensions do not align, some functionality may not work as expected. 
Future releases will add more flexibility here and better error messages. It is 
plausible and planned to safly compare stochastic networks with non-stochastic networks, 
different clustered networks with each other as well as having single node reference 
scenarios.

## Functionality
Once created, a NetworkCollection behaves similarly to a single [Network][pypsa.Network]. 
While it it is not a Subclass of Network, it "duck-types" as one. This means that 
many methods and properties of a Network can be accessed directly on the collection, 
with the results being aggregated or concatenated across all networks in the collection. 
Currently mainly the data accessors and statistics module are supported (see below). 
More functionality will be added in future releases.

### Data Access
You can access all components data (static and dynamic) in the same way as for a single 
network. They will just be concatenated across all networks in the collection.

Accessing static data:
``` py
>>> nc.buses  # doctest: +NORMALIZE_WHITESPACE
                       v_nom type      x  ...        generator sub_network country
custom-dim name                           ...
network1   London      380.0       -0.13  ...                            0      UK
           Norwich     380.0        1.30  ...                            0      UK
           Norwich DC  200.0        1.30  ...                            1      UK
           Manchester  380.0       -2.20  ...  Manchester Wind           0      UK
           Bremen      380.0        8.80  ...                            2      DE
           Bremen DC   200.0        8.80  ...                            1      DE
           Frankfurt   380.0        8.70  ...   Frankfurt Wind           2      DE
           Norway      380.0       10.75  ...                            3      NO
           Norway DC   200.0       10.75  ...                            1      NO
network2   London      380.0       -0.13  ...                            0      UK
           Norwich     380.0        1.30  ...                            0      UK
           Norwich DC  200.0        1.30  ...                            1      UK
           Manchester  380.0       -2.20  ...  Manchester Wind           0      UK
           Bremen      380.0        8.80  ...                            2      DE
           Bremen DC   200.0        8.80  ...                            1      DE
           Frankfurt   380.0        8.70  ...   Frankfurt Wind           2      DE
           Norway      380.0       10.75  ...                            3      NO
           Norway DC   200.0       10.75  ...                            1      NO
<BLANKLINE>
[18 rows x 14 columns]
```

Accessing time series data:
``` py
>>> nc.loads_t.p_set.iloc[:3, [0, 1, 6, 7]]  # First 3 snapshots, London and Norwich from each network  # doctest: +ELLIPSIS
custom-dim             network1              network2
name                     London     Norwich    London    Norwich
snapshot
2015-01-01 00:00:00   35.796244  415.462564   ...        ...
2015-01-01 01:00:00  976.824561  262.606146   ...        ...
2015-01-01 02:00:00  250.587312  418.476353   ...        ...
```

!!! info

    If a pd.MultiIndex is used for the collection, the different levels will be 
    preserved in the concatenated dataframes.

### Statistics Module

The [statistics module][pypsa.Network.statistics] is fully supported for 
NetworkCollections. All statistics expressions can be accessed in the same way as for 
a single network. This includes dataframes and plots. How plots are combined might
change in future releases.

Get a combined energy balance across all networks:
``` py
>>> nc.statistics.energy_balance()  # doctest: +ELLIPSIS
component  custom-dim  carrier  bus_carrier
Generator  network1    gas      AC              1465.27439
                       wind     AC             31082.35370
           network2    gas      AC              ...
                       wind     AC             ...
Load       network1    load     AC            -32547.62808
           network2    load     AC            -32547.62808
dtype: float64
```
Since only the load time series have been shuffled, the aggregated energy balance is
the same for both networks.

Create a plot for the same energy balance:
``` py
>>> fig = nc.statistics.energy_balance.iplot()
>>> fig.write_html("docs/assets/interactive/ac_dc_collection_energy_balance_iplot.html") # docs-hide    
```
<div style="width: 100%; height: 500px;">
    <iframe src="../../assets/interactive/ac_dc_collection_energy_balance_iplot.html" 
            width="100%" height="100%" frameborder="0" style="border: 0px solid #ccc;">
    </iframe>
</div>
For the full range of plotting options, see <!-- md:guide plotting/charts.md -->.

