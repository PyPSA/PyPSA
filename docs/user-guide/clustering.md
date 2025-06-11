# Network Clustering 

In this example, we show how pypsa can deal with spatial clustering of networks. 

```python
import re

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pypsa
```

```python
n = pypsa.examples.scigrid_de()
n.calculate_dependent_values()
```

The important information that pypsa needs for spatial clustering is in the `busmap`. It contains the mapping of which buses should be grouped together, similar to the groupby groups as we know it from pandas.

You can either calculate a `busmap` from the provided clustering algorithms or you can create/use your own busmap.

## Cluster by custom busmap

Let's start with creating our own. 
In the following, we group all buses together which belong to the same operator. Buses which do not have a specific operator just stay on its own.

```python
groups = n.buses.operator.apply(lambda x: re.split(" |,|;", x)[0])
busmap = groups.where(groups != "", n.buses.index)
```

The clustering routine will raise an error if values in non-standard columns are not the same when combined to a common cluster. Therefore, we adjust the columns of the components and delete problematic non-standard values. 

```python
n.lines = n.lines.reindex(columns=n.components["Line"]["attrs"].index[1:])
n.lines["type"] = np.nan
n.buses = n.buses.reindex(columns=n.components["Bus"]["attrs"].index[1:])
n.buses["frequency"] = 50
```

Now we cluster the network based on the busmap.

```python
C = n.cluster.get_clustering_from_busmap(busmap)
```

`C` is a Clustering object which contains all important information.
Among others, the new network is now stored in that Clustering object.

```python
nc = C.network
```

We have a look at the original and the clustered topology

```python
fig, (ax, ax1) = plt.subplots(
    1, 2, subplot_kw={"projection": ccrs.EqualEarth()}, figsize=(12, 12)
)
plot_kwrgs = {"bus_sizes": 1e-3, "line_widths": 0.5}
n.plot(ax=ax, title="original", **plot_kwrgs)
nc.plot(ax=ax1, title="clustered by operator", **plot_kwrgs)
fig.tight_layout()
```

Looks a bit messy as over 120 buses do not have assigned operators.

## Clustering by busmap created from K-means

Let's now make a clustering based on the kmeans algorithm.
Therefore we calculate the `busmap` from a non-weighted kmeans clustering.

```python
weighting = pd.Series(1, n.buses.index)
busmap2 = n.cluster.busmap_by_kmeans(bus_weightings=weighting, n_clusters=50)
```

We use this new kmeans-based `busmap` to create a new clustered method.

```python
nc2 = n.cluster.cluster_by_busmap(busmap2)
```

Again, let's plot the networks to compare:

```python
fig, (ax, ax1) = plt.subplots(
    1, 2, subplot_kw={"projection": ccrs.EqualEarth()}, figsize=(12, 12)
)
plot_kwrgs = {"bus_sizes": 1e-3, "line_widths": 0.5}
n.plot(ax=ax, title="original", **plot_kwrgs)
nc2.plot(ax=ax1, title="clustered by kmeans", **plot_kwrgs)
fig.tight_layout()
```

There are other clustering algorithms in the pipeline of pypsa as the hierarchical
clustering which performs better than the kmeans. Also the `get_clustering_from_busmap` function supports various arguments on how components in the network should be aggregated. 