# -*- coding: utf-8 -*-
## Script to add load, generators, missing lines and transformers to SciGRID
#
#
## WARNING: This script is no longer supported, since the libraries and data no longer exist in their former versions
#
## It is kept here for interest's sake
#
## See https://github.com/PyPSA/pypsa-eur for a newer model that covers all of Europe
#
#
# This Jupyter Notebook is also available to download at: <https://pypsa.readthedocs.io/en/latest/examples/add_load_gen_trafos_to_scigrid.ipynb>  and can be viewed as an HTML page at: https://pypsa.readthedocs.io/en/latest/examples/add_load_gen_trafos_to_scigrid.html.
#
# This script does some post-processing on the original SciGRID dataset version 0.2 and then adds load, generation, transformers and missing lines to the SciGRID dataset.
#
# The intention is to create a model of the German electricity system that is transparent in the sense that all steps from openly-available raw data to the final model can be followed. The model is NOT validated and may contain errors.
#
# Some of the libraries used for attaching the load and generation are not on github, but can be downloaded at
#
# http://fias.uni-frankfurt.de/~hoersch/
#
# The intention is to release these as free software soon. We cannot guarantee to support you when using these libraries.
#
#
#
### Data sources
#
# Grid: based on [SciGRID](http://scigrid.de/) Version 0.2 which is based on [OpenStreetMap](http://www.openstreetmap.org/).
#
# Load size and location: based on Landkreise (NUTS 3) GDP and population.
#
# Load time series: from ENTSO-E hourly data, scaled up uniformly by factor 1.12 (a simplification of the methodology in Schumacher, Hirth (2015)).
#
# Conventional power plant capacities and locations: BNetzA list.
#
# Wind and solar capacities and locations: EEG Stammdaten, based on  http://www.energymap.info/download.html, which represents capacities at the end of 2014. Units without PLZ are removed.
#
# Wind and solar time series: REatlas, Andresen et al, "Validation of Danish wind time series from a new global renewable energy atlas for energy system analysis," Energy 93 (2015) 1074 - 1088.
#
# NB:
#
# All times in the dataset are UTC.
#
# Where SciGRID nodes have been split into 220kV and 380kV substations, all load and generation is attached to the 220kV substation.
#
### Warning
#
# This dataset is ONLY intended to demonstrate the capabilities of PyPSA and is NOT (yet) accurate enough to be used for research purposes.
#
# Known problems include:
#
# i) Rough approximations have been made for missing grid data, e.g. 220kV-380kV transformers and connections between close sub-stations missing from OSM.
#
# ii) There appears to be some unexpected congestion in parts of the network, which may mean for example that the load attachment method (by Voronoi cell overlap with Landkreise) isn't working, particularly in regions with a high density of substations.
#
# iii) Attaching power plants to the nearest high voltage substation may not reflect reality.
#
# iv) There is no proper n-1 security in the calculations - this can either be simulated with a blanket e.g. 70% reduction in thermal limits (as done here) or a proper security constrained OPF (see e.g.  <https://pypsa.readthedocs.io/en/latest/examples/scigrid-sclopf.ipynb>).
#
# v) The borders and neighbouring countries are not represented.
#
# vi) Hydroelectric power stations are not modelled accurately.
#
# viii) The marginal costs are illustrative, not accurate.
#
# ix) Only the first day of 2011 is in the github dataset, which is not representative. The full year of 2011 can be downloaded at <https://pypsa.readthedocs.io/en/latest/examples/scigrid-with-load-gen-trafos-2011.zip>.
#
# x) The ENTSO-E total load for Germany may not be scaled correctly; it is scaled up uniformly by factor 1.12 (a simplification of the methodology in Schumacher, Hirth (2015), which suggests monthly factors).
#
# xi) Biomass from the EEG Stammdaten are not read in at the moment.
#
# xii) Power plant start up costs, ramping limits/costs, minimum loading rates are not considered.

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pypsa

#%matplotlib inline

### Read in the raw SciGRID data

# You may have to adjust this path to where
# you downloaded the github repository
# https://github.com/PyPSA/PyPSA

folder_prefix = os.path.dirname(pypsa.__file__) + "/../examples/scigrid-de/"

# note that some columns have 'quotes because of fields containing commas'
vertices = pd.read_csv(
    folder_prefix + "scigrid-151109/vertices_de_power_151109.csvdata",
    sep=",",
    quotechar="'",
    index_col=0,
)

vertices.rename(columns={"lon": "x", "lat": "y", "name": "osm_name"}, inplace=True)

print(vertices["voltage"].value_counts(dropna=False))

links = pd.read_csv(
    folder_prefix + "scigrid-151109/links_de_power_151109.csvdata",
    sep=",",
    quotechar="'",
    index_col=0,
)
links.rename(
    columns={"v_id_1": "bus0", "v_id_2": "bus1", "name": "osm_name"}, inplace=True
)

links["cables"].fillna(3, inplace=True)
links["wires"].fillna(2, inplace=True)

links["length"] = links["length_m"] / 1000.0

print(links["voltage"].value_counts(dropna=False))

## Drop the DC lines

for voltage in [300000, 400000, 450000]:
    links.drop(links[links.voltage == voltage].index, inplace=True)

## Build the network

network = pypsa.Network()

pypsa.io.import_components_from_dataframe(network, vertices, "Bus")

pypsa.io.import_components_from_dataframe(network, links, "Line")

### Add specific missing AC lines

# Add AC lines known to be missing in SciGRID
# E.g. lines missing because of OSM mapping errors.
# This is no systematic list, just what we noticed;
# please tell SciGRID and/or Tom Brown (brown@fias.uni-frankfurt.de)
# if you know of more examples

columns = ["bus0", "bus1", "wires", "cables", "voltage"]

data = [
    ["100", "255", 2, 6, 220000],  # Niederstedem to Wengerohr
    ["384", "351", 4, 6, 380000],  # Raitersaich to Ingolstadt
    ["351", "353", 4, 6, 380000],  # Ingolstadt to Irsching
]

last_scigrid_line = int(network.lines.index[-1])

index = [
    str(i) for i in range(last_scigrid_line + 1, last_scigrid_line + 1 + len(data))
]

missing_lines = pd.DataFrame(data, index, columns)

# On average, SciGRID lines are 25% longer than the direct distance
length_factor = 1.25

missing_lines["length"] = [
    length_factor
    * pypsa.geo.haversine(
        network.buses.loc[r.bus0, ["x", "y"]], network.buses.loc[r.bus1, ["x", "y"]]
    )[0, 0]
    for i, r in missing_lines.iterrows()
]

pypsa.io.import_components_from_dataframe(network, missing_lines, "Line")

network.lines.tail()

### Determine the voltage of the buses by the lines which end there

network.lines.voltage.value_counts()


buses_by_voltage = {}

for voltage in network.lines.voltage.value_counts().index:
    buses_by_voltage[voltage] = set(
        network.lines[network.lines.voltage == voltage].bus0
    ) | set(network.lines[network.lines.voltage == voltage].bus1)

# give priority to 380 kV
network.buses["v_nom"] = 380
network.buses.loc[buses_by_voltage[220000], "v_nom"] = 220
network.buses.loc[buses_by_voltage[380000], "v_nom"] = 380

network.buses.v_nom.value_counts(dropna=False)

### Connect buses which are < 850m apart
#
# There are pairs of buses less than 850m apart which are not connected in SciGRID, but clearly connected in OpenStreetMap (OSM).
#
# The reason is that the relations for connections between close substations do not appear in OSM.
#
# Here they are connected with 2 circuits of the appropriate voltage level (an asumption).
#
# 850m is chosen as a limit based on manually looking through the examples.
#
# The example 46-48 (Marzahn) at 892 m apart is the first example of close substations which are not connected in reality.

# Compute the distances for unique pairs

pairs = pd.Series()

for i, u in enumerate(network.buses.index):
    vs = network.buses[["x", "y"]].iloc[i + 1 :]
    distance_km = pypsa.geo.haversine(vs, network.buses.loc[u, ["x", "y"]])

    to_add = pd.Series(data=distance_km[:, 0], index=[(u, v) for v in vs.index])

    pairs = pd.concat((pairs, to_add))

pairs.sort_values().head()

# determine topology so we can look what's actually connected
network.determine_network_topology()

# Example all substations which are close to
# each other geographically by not connected in network.adj

start = 0  # km
stop = 1  # km

for (u, v), dist in pairs.sort_values().iteritems():

    if dist < start:
        continue

    # only go up to pairs stop km apart
    if dist > stop:
        break

    # ignore if they're already connected
    if u in network.graph().adj[v]:
        continue

    print(u, v, dist)

    u_x = network.buses.at[u, "x"]
    u_y = network.buses.at[u, "y"]
    v_x = network.buses.at[v, "x"]
    v_y = network.buses.at[v, "y"]

    # have a look what's going on in OSM
    print("https://www.openstreetmap.org/#map=18/{}/{}".format(u_y, u_x))
    print("https://www.openstreetmap.org/#map=18/{}/{}".format(v_y, v_x))

# From examining the map, it seems that all cases where substations
# are less than 850m apart are connected in reality
# The first one to fail is 46-48 (Marzahn) at 892 m

# Connect these substations

limit = 0.85

for (u, v), dist in pairs.sort_values().iteritems():

    # only go up to pairs stop km apart
    if dist > limit:
        break

    # ignore if they're already connected
    if u in network.graph().adj[v]:
        continue

    kv_u = network.buses.at[u, "v_nom"]
    kv_v = network.buses.at[v, "v_nom"]

    print(u, v, dist, kv_u, kv_v)

    last_scigrid_line = int(network.lines.index[-1])

    voltage = max(kv_u, kv_v) * 1000

    wires = {220000: 2, 380000: 4}[voltage]

    cables = 6

    df = pd.DataFrame(
        [[u, v, length_factor * dist, wires, cables, voltage]],
        columns=["bus0", "bus1", "length", "wires", "cables", "voltage"],
        index=[str(last_scigrid_line + 1)],
    )

    pypsa.io.import_components_from_dataframe(network, df, "Line")

### Split buses with more than one voltage; add trafos between
#
# This code splits the buses where you have 220 and 380 kV lines landing.

network.lines.voltage.value_counts()


buses_by_voltage = {}

for voltage in network.lines.voltage.value_counts().index:
    buses_by_voltage[voltage] = set(
        network.lines[network.lines.voltage == voltage].bus0
    ) | set(network.lines[network.lines.voltage == voltage].bus1)

network.buses.v_nom = 380
network.buses.loc[buses_by_voltage[220000], "v_nom"] = 220
network.buses.loc[buses_by_voltage[380000], "v_nom"] = 380

overlap = buses_by_voltage[220000] & buses_by_voltage[380000]
len(overlap)

## build up new buses and transformers to import


buses_to_split = [str(i) for i in sorted([int(item) for item in overlap])]
buses_to_split_df = network.buses.loc[buses_to_split]

buses_to_split_df.v_nom = 220

buses_to_split_220kV = [name + "_220kV" for name in buses_to_split_df.index]

buses_to_split_df.index = buses_to_split_220kV

trafos_df = pd.DataFrame(index=buses_to_split)
trafos_df["bus0"] = buses_to_split
trafos_df["bus1"] = buses_to_split_220kV
trafos_df["x"] = 0.1
# This high a nominal power is required for feasibility in LOPF
trafos_df["s_nom"] = 2000

pypsa.io.import_components_from_dataframe(network, buses_to_split_df, "Bus")
pypsa.io.import_components_from_dataframe(network, trafos_df, "Transformer")

##reconnect lines to the correct voltage bus

for line in network.lines.index:
    bus0 = network.lines.at[line, "bus0"]
    bus1 = network.lines.at[line, "bus1"]
    v0 = network.buses.at[bus0, "v_nom"]
    v1 = network.buses.at[bus1, "v_nom"]
    v = network.lines.at[line, "voltage"]
    if v0 != v / 1000.0:
        print(line, v0, v)
        network.lines.at[line, "bus0"] = bus0 + "_220kV"
    if v1 != v / 1000.0:
        network.lines.at[line, "bus1"] = bus1 + "_220kV"

# determine the connected components

network.determine_network_topology()


# remove small isolated networks
for sn in network.sub_networks.obj:
    buses = sn.buses().index
    branches = sn.branches().index

    if len(buses) < 5:
        print(
            "Dropping Sub-Network {} because it only has {} buses".format(
                sn, len(buses)
            )
        )
        # print(buses.index)
        # print(len(branches),branches.index)
        for bus in buses:
            network.remove("Bus", bus)
        for branch in branches:
            network.remove("Line", branch[1])
    else:
        print("Keeping Sub-Network {} because it has {} buses".format(sn, len(buses)))

# rebuild topology

network.determine_network_topology()

colors = network.lines.voltage.map(
    lambda v: "g" if v == 220000 else "r" if v == 380000 else "c"
)

network.plot(line_colors=colors)

### Recalculate all electrical properties

network.lines["type"] = network.lines.voltage.map(
    {220000: "Al/St 240/40 2-bundle 220.0", 380000: "Al/St 240/40 4-bundle 380.0"}
)

network.lines["num_parallel"] = (
    network.lines.cables
    / 3.0
    * network.lines.wires
    / network.lines.voltage.map({220000: 2.0, 380000: 4.0})
)

network.lines["s_nom"] = (
    3.0**0.5
    * network.lines.voltage
    / 1000.0
    * network.lines.num_parallel
    * network.lines.voltage.map({220000: 2.0, 380000: 4.0})
    * 0.65
)

## Attach the load

# import FIAS libraries for attaching data

# this script uses old versions of the FIAS libraries and
# has not yet been updated to the new versions

# the latest versions are available at
# https://github.com/FRESNA/vresutils

# if you get it working with the new versions, please
# tell us! It shouldn't be too hard...

try:
    import load
    import vresutils
except:
    print("Oh dear! You don't have FIAS libraries, so you cannot add load :-(")

import load
import networkx as nx
from load import germany as DEload
from shapely.geometry import Polygon
from vresutils import shapes as vshapes

# bounding poly for Germany for the Voronoi - necessary
# because some SciGRID points lie outside border vshapes.germany()
poly = Polygon([[5.8, 47.0], [5.8, 55.5], [15.2, 55.5], [15.2, 47.0]])


def generate_dummy_graph(network):
    """
    Generate a dummy graph to feed to the FIAS libraries.

    It adds the "pos" attribute and removes the 380 kV duplicate buses
    when the buses have been split, so that all load and generation is
    attached to the 220kV bus.
    """

    graph = pypsa.descriptors.OrderedGraph()

    graph.add_nodes_from(
        [bus for bus in network.buses.index if bus not in buses_to_split]
    )

    # add positions to graph for voronoi cell computation
    for node in graph.nodes():
        graph.node[node]["pos"] = np.array(
            network.buses.loc[node, ["x", "y"]], dtype=float
        )

    return graph


graph = generate_dummy_graph(network)

graph.name = "scigrid_v2"


def voronoi_partition(G, outline):
    """
    For 2D-embedded graph `G`, within the boundary given by the shapely polygon
    `outline`, returns `G` with the Voronoi cell region as an additional node
    attribute.
    """
    # following line from vresutils.graph caused a bug
    # G = polygon_subgraph(G, outline, copy=False)
    points = list(vresutils.graph.get_node_attributes(G, "pos").values())
    regions = vresutils.graph.voronoi_partition_pts(
        points, outline, no_multipolygons=True
    )
    nx.set_node_attributes(G, "region", dict(zip(G.nodes(), regions)))

    return G


voronoi_partition(graph, poly)

# NB: starts at midnight CET, 23:00 UTC
load = DEload.timeseries(graph, years=[2011, 2012, 2013, 2014])

# Kill the Timezone information to avoid pandas bugs
load.index = load.index.values

# Take the first year (in UTC time - we don't set time zone because of a Pandas bug)
network.set_snapshots(pd.date_range("2011-01-01 00:00", "2011-12-31 23:00", freq="H"))

print(network.snapshots)

# temporary load scaling factor for Germany load in relation to ENTSO-E hourly load
# based roughly on Schumacher & Hirth (2015)
# http://www.feem.it/userfiles/attach/20151191122284NDL2015-088.pdf
# In principle rescaling should happen on a monthly basis

load_factor = 1.12

for bus in graph.nodes():
    network.add(
        "Load",
        bus,
        bus=bus,
        p_set=pd.Series(
            data=load_factor * 1000 * load.loc[network.snapshots, bus],
            index=network.snapshots,
        ),
    )

#%matplotlib inline

pd.DataFrame(load.sum(axis=1)).plot()

load_distribution = (
    network.loads_t.p_set.loc[network.snapshots[0]].groupby(network.loads.bus).sum()
)

network.plot(bus_sizes=load_distribution)

total_load = load.sum(axis=1)

monthly_load = total_load.resample("M").sum()

monthly_load.plot(grid=True)

## Attach conventional generators from BNetzA list


def read_kraftwerksliste(with_latlon=True):
    kraftwerke = pd.read_csv(
        "../../lib/vresutils/data/Kraftwerksliste_CSV_deCP850ed.csv",
        delimiter=";",
        encoding="utf-8",
        thousands=".",
        decimal=",",
    )

    def sanitize_names(x):
        try:
            x = x[: x.index("(")]
        except ValueError:
            pass
        return x.replace("\n", " ").strip()

    kraftwerke.columns = kraftwerke.columns.map(sanitize_names)

    def sanitize_plz(x):
        try:
            x = x.strip()
            if len(x) > 5:
                x = x[:5]
            return float(x)
        except (ValueError, AttributeError):
            return np.NAN

    kraftwerke.PLZ = kraftwerke.PLZ.apply(sanitize_plz)
    if with_latlon:
        postcodes = {
            pc: sh.centroid
            for pc, sh in vshapes.postcodeareas().items()
            if sh is not None
        }
        kraftwerke["lon"] = kraftwerke.PLZ.map({pc: c.x for pc, c in postcodes.items()})
        kraftwerke["lat"] = kraftwerke.PLZ.map({pc: c.y for pc, c in postcodes.items()})
        # kraftwerke.dropna(subset=('lon','lat'), inplace=True)

    kraftwerke["Type"] = kraftwerke["Auswertung Energieträger"].map(
        {
            "Erdgas": "Gas",
            "Grubengas": "Gas",
            "Laufwasser": "Run of River",
            "Pumpspeicher": "Pumped Hydro",
            "Speicherwasser (ohne Pumpspeicher)": "Storage Hydro",
            "Mineralölprodukte": "Oil",
            "Steinkohle": "Hard Coal",
            "Braunkohle": "Brown Coal",
            "Abfall": "Waste",
            "Kernenergie": "Nuclear",
            "Sonstige Energieträger\n(nicht erneuerbar) ": "Other",
            "Mehrere Energieträger\n(nicht erneuerbar)": "Multiple",
            "Biomasse": "Biomass",
            "Deponiegas": "Gas",
            "Klärgas": "Gas",
            "Geothermie": "Geothermal",
            "Windenergie (Onshore-Anlage)": "Wind Onshore",
            "Windenergie (Offshore-Anlage)": "Wind Offshore",
            "Solare Strahlungsenergie": "Solar",
            "Unbekannter Energieträger\n(nicht erneuerbar)": "Other",
        }
    )

    return kraftwerke


power_plants = read_kraftwerksliste()

power_plants[power_plants["Unternehmen"] == "EEG-Anlagen < 10 MW"].groupby("Type").sum()

power_plants.groupby("Type").sum()

import random

# NB: bnetza extracted from BNetzA using

# ./Kraftwerksdaten.ipynb


def backup_capacity_german_grid(G):
    from shapely.geometry import Point

    plants = power_plants
    plants = plants[plants["Kraftwerksstatus"] == "in Betrieb"]

    # remove EEG-receiving power plants - except biomass, these will be added later

    # it's necessary to remove biomass because we don't have coordinates for it

    for tech in ["Solar", "Wind Onshore", "Wind Offshore", "Biomass"]:
        plants = plants[plants["Type"] != tech]

    cells = {n: d["region"] for n, d in G.nodes_iter(data=True)}

    def nodeofaplant(x):
        if np.isnan(x["lon"]) or np.isnan(x["lat"]):
            return random.choice(list(cells.keys()))
        p = Point(x["lon"], x["lat"])
        for n, cell in cells.items():
            if cell.contains(p):
                return n
            else:
                return min(cells, key=lambda n: cells[n].distance(p))

    nodes = plants.apply(nodeofaplant, axis=1)

    capacity = plants["Netto-Nennleistung"].groupby((nodes, plants["Type"])).sum() / 1e3
    capacity.name = "Capacity"

    return capacity


cap = backup_capacity_german_grid(graph)

cap.describe(), cap.sum(), type(cap)

print(cap[pd.isnull(cap)])

cap.fillna(0.1, inplace=True)


cap.index.levels[1]

m_costs = {
    "Gas": 50.0,
    "Brown Coal": 10.0,
    "Hard Coal": 25.0,
    "Oil": 100.0,
    "Nuclear": 8.0,
    "Pumped Hydro": 3.0,
    "Storage Hydro": 3.0,
    "Run of River": 3.0,
    "Geothermal": 26.0,
    "Waste": 6.0,
    "Multiple": 28.0,
    "Other": 32.0,
}

default_cost = 10.0

for (bus_name, tech_name) in cap.index:
    print(bus_name, tech_name, cap[(bus_name, tech_name)])
    if tech_name == "Pumped Hydro":
        network.add(
            "StorageUnit",
            bus_name + " " + tech_name,
            bus=bus_name,
            p_nom=1000 * cap[(bus_name, tech_name)],
            marginal_cost=m_costs.get(tech_name, default_cost),
            carrier=tech_name,
            max_hours=6,
            efficiency_store=0.95,
            efficiency_dispatch=0.95,
        )
    else:
        network.add(
            "Generator",
            bus_name + " " + tech_name,
            bus=bus_name,
            p_nom=1000 * cap[(bus_name, tech_name)],
            marginal_cost=m_costs.get(tech_name, default_cost),
            carrier=tech_name,
        )

## Add renewables

import generation.germany as DEgen

generation = DEgen.timeseries_eeg(graph)


generation.items

# Kill the Timezone information to avoid pandas bugs
generation.major_axis = generation.major_axis.values

generation.loc[["wind", "solar"], network.snapshots, :].sum(axis=2).plot()

solar = generation.loc["solar", network.snapshots, :].sum(axis=1)
solar.describe()

# make sure the ordering of the minor axis is correc
generation.minor_axis = graph.nodes()

## Get the capacities correct

cutout = vresutils.reatlas.Cutout(cutoutname="Europe_2011_2014", username="becker")


def panel_capacity(panel):
    """
    Returns the panel capacity in MW.

    Parameters
    ----------
    panel : string
        Panel name, e.g. "Sunpower"

    Returns
    -------
    capacity : float
        In MW
    """
    c = vresutils.reatlas.solarpanelconf_to_solar_panel_config_object(panel)
    return c["A"] + c["B"] * 1000 + c["C"] * np.log(1000)


solar_layouts = DEgen.eeg_solarlayouts(graph, cutout)

panel_cap = panel_capacity(solar_layouts[0]["panel"])
solar_caps = pd.Series(solar_layouts[1].sum(axis=(1, 2)) * panel_cap, graph.nodes())

solar_caps.describe(), solar_caps.sum()

(generation.solar.max() / solar_caps).describe()

windon_layouts = DEgen.eeg_windonlayouts_per_class(graph, cutout)

windon_capacities = pd.DataFrame(index=graph.nodes())
for turbine_items in windon_layouts:
    name = turbine_items[0]["onshore"]
    turbine_cap = np.array(
        vresutils.reatlas.turbineconf_to_powercurve_object(name)["POW"]
    ).max()
    print(name, turbine_cap)
    windon_capacities[name] = turbine_items[1].sum(axis=(1, 2)) * turbine_cap / 1000.0

windon_caps = windon_capacities.sum(axis=1)
windon_caps.describe(), windon_caps.sum()

(generation.windon.max() / windon_caps).describe()

windoff_layouts = DEgen.eeg_windofflayouts_per_class(graph, cutout)

windoff_capacities = pd.DataFrame(index=graph.nodes())
for i, turbine_items in enumerate(windoff_layouts):
    name = turbine_items[0]["offshore"]
    turbine_cap = np.array(
        vresutils.reatlas.turbineconf_to_powercurve_object(name)["POW"]
    ).max()
    print(name, turbine_cap)
    # add an index to name to avoid duplication of names
    windoff_capacities[name + "-" + str(i)] = (
        turbine_items[1].sum(axis=(1, 2)) * turbine_cap / 1000.0
    )

windoff_capacities.sum()

windoff_caps = windoff_capacities.sum(axis=1)
windoff_caps.describe(), windoff_caps.sum()

(generation.windoff.max() / windoff_caps).describe()

network.plot(bus_sizes=1000 * windoff_caps)

network.plot(bus_sizes=1000 * windon_caps)

network.plot(bus_sizes=1000 * solar_caps)


d = {
    "windoff": {"full_name": "Wind Offshore", "caps": windoff_caps},
    "windon": {"full_name": "Wind Onshore", "caps": windon_caps},
    "solar": {"full_name": "Solar", "caps": solar_caps},
}

for tech in ["windoff", "windon", "solar"]:
    caps = d[tech]["caps"]
    caps = caps[caps != 0]

    for i in caps.index:
        network.add(
            "Generator",
            "{} {}".format(i, d[tech]["full_name"]),
            p_nom=caps[i] * 1000.0,
            dispatch="variable",
            bus=i,
            carrier=d[tech]["full_name"],
            p_max_pu=generation[tech].loc[network.snapshots, i] / caps[i],
        )

csv_folder_name = "../../lib/data/de_model/scigrid-with-load-gen-trafos"

network.export_to_csv_folder(csv_folder_name)

network.set_snapshots(network.snapshots[:24])

csv_folder_name = "../../lib/pypsa/examples/scigrid-de/scigrid-with-load-gen-trafos"


network.export_to_csv_folder(csv_folder_name)
