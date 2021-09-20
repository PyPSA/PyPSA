# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""
Creates summaries of aggregated energy and costs as ``.csv`` files.

Relevant Settings
-----------------

.. code:: yaml

    costs:
        USD2013_to_EUR2013:
        discountrate:
        marginal_cost:
        capital_cost:

    electricity:
        max_hours:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`costs_cf`, :ref:`electricity_cf`

Inputs
------

Outputs
-------

Description
-----------

The following rule can be used to summarize the results in seperate .csv files:

.. code::

    snakemake results/summaries/elec_s_all_lall_Co2L-3H_all
                                         clusters
                                             line volume or cost cap
                                                - options
                                                        - all countries

the line volume/cost cap field can be set to one of the following:
* ``lv1.25`` for a particular line volume extension by 25%
* ``lc1.25`` for a line cost extension by 25 %
* ``lall`` for all evalutated caps
* ``lvall`` for all line volume caps
* ``lcall`` for all line cost caps

Replacing '/summaries/' with '/plots/' creates nice colored maps of the results.

"""

import logging
from _helpers import configure_logging

import os
import pypsa
import pandas as pd

from add_electricity import load_costs, update_transmission_costs

idx = pd.IndexSlice

logger = logging.getLogger(__name__)

opt_name = {"Store": "e", "Line" : "s", "Transformer" : "s"}


def _add_indexed_rows(df, raw_index):
    new_index = df.index.union(pd.MultiIndex.from_product(raw_index))
    if isinstance(new_index, pd.Index):
        new_index = pd.MultiIndex.from_tuples(new_index)

    return df.reindex(new_index)


def assign_carriers(n):

    if "carrier" not in n.loads:
        n.loads["carrier"] = "electricity"
        for carrier in ["transport","heat","urban heat"]:
            n.loads.loc[n.loads.index.str.contains(carrier),"carrier"] = carrier

    n.storage_units['carrier'].replace({'hydro': 'hydro+PHS', 'PHS': 'hydro+PHS'}, inplace=True)

    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"

    n.lines["carrier"].replace({"AC": "lines"}, inplace=True)

    if n.links.empty: n.links["carrier"] = pd.Series(dtype=str)
    n.links["carrier"].replace({"DC": "lines"}, inplace=True)

    if "EU gas store" in n.stores.index and n.stores.loc["EU gas Store","carrier"] == "":
        n.stores.loc["EU gas Store","carrier"] = "gas Store"


def calculate_costs(n, label, costs):

    for c in n.iterate_components(n.branch_components|n.controllable_one_port_components^{"Load"}):
        capital_costs = c.df.capital_cost*c.df[opt_name.get(c.name,"p") + "_nom_opt"]
        capital_costs_grouped = capital_costs.groupby(c.df.carrier).sum()

        # Index tuple(s) indicating the newly to-be-added row(s)
        raw_index = tuple([[c.list_name],["capital"],list(capital_costs_grouped.index)])
        costs = _add_indexed_rows(costs, raw_index)

        costs.loc[idx[raw_index],label] = capital_costs_grouped.values

        if c.name == "Link":
            p = c.pnl.p0.multiply(n.snapshot_weightings.generators,axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.multiply(n.snapshot_weightings.generators,axis=0)
            p_all[p_all < 0.] = 0.
            p = p_all.sum()
        else:
            p = c.pnl.p.multiply(n.snapshot_weightings.generators,axis=0).sum()

        marginal_costs = p*c.df.marginal_cost

        marginal_costs_grouped = marginal_costs.groupby(c.df.carrier).sum()

        costs = costs.reindex(costs.index.union(pd.MultiIndex.from_product([[c.list_name],["marginal"],marginal_costs_grouped.index])))

        costs.loc[idx[c.list_name,"marginal",list(marginal_costs_grouped.index)],label] = marginal_costs_grouped.values

    return costs

def calculate_curtailment(n, label, curtailment):

    avail = n.generators_t.p_max_pu.multiply(n.generators.p_nom_opt).sum().groupby(n.generators.carrier).sum()
    used = n.generators_t.p.sum().groupby(n.generators.carrier).sum()

    curtailment[label] = (((avail - used)/avail)*100).round(3)

    return curtailment

def calculate_energy(n, label, energy):

    for c in n.iterate_components(n.one_port_components|n.branch_components):

        if c.name in {'Generator', 'Load', 'ShuntImpedance'}:
            c_energies = c.pnl.p.multiply(n.snapshot_weightings.generators,axis=0).sum().multiply(c.df.sign).groupby(c.df.carrier).sum()
        elif c.name in {'StorageUnit', 'Store'}:
            c_energies = c.pnl.p.multiply(n.snapshot_weightings.stores,axis=0).sum().multiply(c.df.sign).groupby(c.df.carrier).sum()
        else:
            c_energies = (-c.pnl.p1.multiply(n.snapshot_weightings.generators,axis=0).sum() - c.pnl.p0.multiply(n.snapshot_weightings.generators,axis=0).sum()).groupby(c.df.carrier).sum()

        energy = include_in_summary(energy, [c.list_name], label, c_energies)

    return energy

def include_in_summary(summary, multiindexprefix, label, item):

    # Index tuple(s) indicating the newly to-be-added row(s)
    raw_index = tuple([multiindexprefix,list(item.index)])
    summary = _add_indexed_rows(summary, raw_index)

    summary.loc[idx[raw_index], label] = item.values

    return summary

def calculate_capacity(n,label,capacity):

    for c in n.iterate_components(n.one_port_components):
        if 'p_nom_opt' in c.df.columns:
            c_capacities = abs(c.df.p_nom_opt.multiply(c.df.sign)).groupby(c.df.carrier).sum()
            capacity = include_in_summary(capacity, [c.list_name], label, c_capacities)

    for c in n.iterate_components(n.passive_branch_components):
        c_capacities = c.df['s_nom_opt'].groupby(c.df.carrier).sum()
        capacity = include_in_summary(capacity, [c.list_name], label, c_capacities)

    for c in n.iterate_components(n.controllable_branch_components):
        c_capacities = c.df.p_nom_opt.groupby(c.df.carrier).sum()
        capacity = include_in_summary(capacity, [c.list_name], label, c_capacities)

    return capacity

def calculate_supply(n, label, supply):
    """calculate the max dispatch of each component at the buses where the loads are attached"""

    load_types = n.loads.carrier.value_counts().index

    for i in load_types:

        buses = n.loads.bus[n.loads.carrier == i].values

        bus_map = pd.Series(False,index=n.buses.index)

        bus_map.loc[buses] = True

        for c in n.iterate_components(n.one_port_components):

            items = c.df.index[c.df.bus.map(bus_map)]

            if len(items) == 0 or c.pnl.p.empty:
                continue

            s = c.pnl.p[items].max().multiply(c.df.loc[items,'sign']).groupby(c.df.loc[items,'carrier']).sum()

            # Index tuple(s) indicating the newly to-be-added row(s)
            raw_index = tuple([[i],[c.list_name],list(s.index)])
            supply = _add_indexed_rows(supply, raw_index)

            supply.loc[idx[raw_index],label] = s.values


        for c in n.iterate_components(n.branch_components):

            for end in ["0","1"]:

                items = c.df.index[c.df["bus" + end].map(bus_map)]

                if len(items) == 0 or c.pnl["p"+end].empty:
                    continue

                #lots of sign compensation for direction and to do maximums
                s = (-1)**(1-int(end))*((-1)**int(end)*c.pnl["p"+end][items]).max().groupby(c.df.loc[items,'carrier']).sum()

                supply = supply.reindex(supply.index.union(pd.MultiIndex.from_product([[i],[c.list_name],s.index])))
                supply.loc[idx[i,c.list_name,list(s.index)],label] = s.values

    return supply


def calculate_supply_energy(n, label, supply_energy):
    """calculate the total dispatch of each component at the buses where the loads are attached"""

    load_types = n.loads.carrier.value_counts().index

    for i in load_types:

        buses = n.loads.bus[n.loads.carrier == i].values

        bus_map = pd.Series(False,index=n.buses.index)

        bus_map.loc[buses] = True

        for c in n.iterate_components(n.one_port_components):

            items = c.df.index[c.df.bus.map(bus_map)]

            if len(items) == 0 or c.pnl.p.empty:
                continue

            s = c.pnl.p[items].sum().multiply(c.df.loc[items,'sign']).groupby(c.df.loc[items,'carrier']).sum()

            # Index tuple(s) indicating the newly to-be-added row(s)
            raw_index = tuple([[i],[c.list_name],list(s.index)])
            supply_energy = _add_indexed_rows(supply_energy, raw_index)

            supply_energy.loc[idx[raw_index],label] = s.values


        for c in n.iterate_components(n.branch_components):

            for end in ["0","1"]:

                items = c.df.index[c.df["bus" + end].map(bus_map)]

                if len(items) == 0  or c.pnl['p' + end].empty:
                    continue

                s = (-1)*c.pnl["p"+end][items].sum().groupby(c.df.loc[items,'carrier']).sum()

                supply_energy = supply_energy.reindex(supply_energy.index.union(pd.MultiIndex.from_product([[i],[c.list_name],s.index])))
                supply_energy.loc[idx[i,c.list_name,list(s.index)],label] = s.values

    return supply_energy


def calculate_metrics(n,label,metrics):

    metrics = metrics.reindex(metrics.index.union(pd.Index(["line_volume","line_volume_limit","line_volume_AC","line_volume_DC","line_volume_shadow","co2_shadow"])))

    metrics.at["line_volume_DC",label] = (n.links.length*n.links.p_nom_opt)[n.links.carrier == "DC"].sum()
    metrics.at["line_volume_AC",label] = (n.lines.length*n.lines.s_nom_opt).sum()
    metrics.at["line_volume",label] = metrics.loc[["line_volume_AC","line_volume_DC"],label].sum()

    if hasattr(n,"line_volume_limit"):
        metrics.at["line_volume_limit",label] = n.line_volume_limit

    if hasattr(n,"line_volume_limit_dual"):
        metrics.at["line_volume_shadow",label] = n.line_volume_limit_dual

    if "CO2Limit" in n.global_constraints.index:
        metrics.at["co2_shadow",label] = n.global_constraints.at["CO2Limit","mu"]

    return metrics


def calculate_prices(n,label,prices):

    bus_type = pd.Series(n.buses.index.str[3:],n.buses.index).replace("","electricity")

    prices = prices.reindex(prices.index.union(bus_type.value_counts().index))

    logger.warning("Prices are time-averaged, not load-weighted")
    prices[label] = n.buses_t.marginal_price.mean().groupby(bus_type).mean()

    return prices


def calculate_weighted_prices(n,label,weighted_prices):

    logger.warning("Weighted prices don't include storage units as loads")

    weighted_prices = weighted_prices.reindex(pd.Index(["electricity","heat","space heat","urban heat","space urban heat","gas","H2"]))

    link_loads = {"electricity" :  ["heat pump", "resistive heater", "battery charger", "H2 Electrolysis"],
                  "heat" : ["water tanks charger"],
                  "urban heat" : ["water tanks charger"],
                  "space heat" : [],
                  "space urban heat" : [],
                  "gas" : ["OCGT","gas boiler","CHP electric","CHP heat"],
                  "H2" : ["Sabatier", "H2 Fuel Cell"]}

    for carrier in link_loads:

        if carrier == "electricity":
            suffix = ""
        elif carrier[:5] == "space":
            suffix = carrier[5:]
        else:
            suffix =  " " + carrier

        buses = n.buses.index[n.buses.index.str[2:] == suffix]

        if buses.empty:
            continue

        if carrier in ["H2","gas"]:
            load = pd.DataFrame(index=n.snapshots,columns=buses,data=0.)
        elif carrier[:5] == "space":
            load = heat_demand_df[buses.str[:2]].rename(columns=lambda i: str(i)+suffix)
        else:
            load = n.loads_t.p_set[buses]


        for tech in link_loads[carrier]:

            names = n.links.index[n.links.index.to_series().str[-len(tech):] == tech]

            if names.empty:
                continue

            load += n.links_t.p0[names].groupby(n.links.loc[names,"bus0"],axis=1).sum(axis=1)

        # Add H2 Store when charging
        if carrier == "H2":
            stores = n.stores_t.p[buses+ " Store"].groupby(n.stores.loc[buses+ " Store","bus"],axis=1).sum(axis=1)
            stores[stores > 0.] = 0.
            load += -stores

        weighted_prices.loc[carrier,label] = (load*n.buses_t.marginal_price[buses]).sum().sum()/load.sum().sum()

        if carrier[:5] == "space":
            print(load*n.buses_t.marginal_price[buses])

    return weighted_prices


outputs = ["costs",
           "curtailment",
           "energy",
           "capacity",
           "supply",
           "supply_energy",
           "prices",
           "weighted_prices",
           "metrics",
           ]


def make_summaries(networks_dict, country='all'):

    columns = pd.MultiIndex.from_tuples(networks_dict.keys(),names=["simpl","clusters","ll","opts"])

    dfs = {}

    for output in outputs:
        dfs[output] = pd.DataFrame(columns=columns,dtype=float)

    for label, filename in networks_dict.items():
        print(label, filename)
        if not os.path.exists(filename):
            print("does not exist!!")
            continue

        try:
            n = pypsa.Network(filename)
        except OSError:
            logger.warning("Skipping {filename}".format(filename=filename))
            continue

        if country != 'all':
            n = n[n.buses.country == country]

        Nyears = n.snapshot_weightings.objective.sum() / 8760.
        costs = load_costs(Nyears, snakemake.input[0],
                           snakemake.config['costs'], snakemake.config['electricity'])
        update_transmission_costs(n, costs, simple_hvdc_costs=False)

        assign_carriers(n)

        for output in outputs:
            dfs[output] = globals()["calculate_" + output](n, label, dfs[output])

    return dfs


def to_csv(dfs):
    dir = snakemake.output[0]
    os.makedirs(dir, exist_ok=True)
    for key, df in dfs.items():
        df.to_csv(os.path.join(dir, f"{key}.csv"))


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('make_summary', network='elec', simpl='',
                           clusters='5', ll='copt', opts='Co2L-24H', country='all')
        network_dir = os.path.join('..', 'results', 'networks')
    else:
        network_dir = os.path.join('results', 'networks')
    configure_logging(snakemake)

    def expand_from_wildcard(key):
        w = getattr(snakemake.wildcards, key)
        return snakemake.config["scenario"][key] if w == "all" else [w]

    if snakemake.wildcards.ll.endswith("all"):
        ll = snakemake.config["scenario"]["ll"]
        if len(snakemake.wildcards.ll) == 4:
            ll = [l for l in ll if l[0] == snakemake.wildcards.ll[0]]
    else:
        ll = [snakemake.wildcards.ll]

    networks_dict = {(simpl,clusters,l,opts) :
        os.path.join(network_dir, f'elec_s{simpl}_'
                                  f'{clusters}_ec_l{l}_{opts}.nc')
                     for simpl in expand_from_wildcard("simpl")
                     for clusters in expand_from_wildcard("clusters")
                     for l in ll
                     for opts in expand_from_wildcard("opts")}

    dfs = make_summaries(networks_dict, country=snakemake.wildcards.country)

    to_csv(dfs)
