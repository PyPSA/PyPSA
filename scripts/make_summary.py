
from six import iteritems

import sys

sys.path = ['/home/vres/lib/python3.5/site-packages'] + sys.path

import pandas as pd

import pypsa

from vresutils.costdata import annuity

from prepare_network import generate_periodic_profiles

import yaml

import helper

idx = pd.IndexSlice

opt_name = {"Store": "e", "Line" : "s", "Transformer" : "s"}


#separator to find group name
find_by = " "

#defaults for group name
defaults = {"Load" : "electricity", "Link" : "transmission lines"}


def assign_groups(n):
    for c in n.iterate_components(n.one_port_components|n.branch_components):

        c.df["group"] = defaults.get(c.name,"default")

        ifind = pd.Series(c.df.index.str.find(find_by),c.df.index)

        for i in ifind.value_counts().index:
            #these have already been assigned defaults
            if i == -1:
                continue

            names = ifind.index[ifind == i]

            c.df.loc[names,'group'] = names.str[i+1:]




def calculate_costs(n,label,costs):

    for c in n.iterate_components(n.branch_components|n.controllable_one_port_components^{"Load"}):
        capital_costs = c.df.capital_cost*c.df[opt_name.get(c.name,"p") + "_nom_opt"]
        capital_costs_grouped = capital_costs.groupby(c.df.group).sum()

        costs = costs.reindex(costs.index|pd.MultiIndex.from_product([[c.list_name],["capital"],capital_costs_grouped.index]))

        costs.loc[idx[c.list_name,"capital",list(capital_costs_grouped.index)],label] = capital_costs_grouped.values

        if c.name == "Link":
            p = c.pnl.p0.sum()
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.copy()
            p_all[p_all < 0.] = 0.
            p = p_all.sum()
        else:
            p = c.pnl.p.sum()

        marginal_costs = p*c.df.marginal_cost

        marginal_costs_grouped = marginal_costs.groupby(c.df.group).sum()

        costs = costs.reindex(costs.index|pd.MultiIndex.from_product([[c.list_name],["marginal"],marginal_costs_grouped.index]))

        costs.loc[idx[c.list_name,"marginal",list(marginal_costs_grouped.index)],label] = marginal_costs_grouped.values

    #add back in costs of links if there is a line volume limit
    if label[1] != "opt":
        costs.loc[("links-added","capital","transmission lines"),label] = ((400*1.25*n.links.length+150000.)*n.links.p_nom_opt)[n.links.group == "transmission lines"].sum()*1.5*(annuity(40., 0.07)+0.02)

    #add back in all hydro
    costs.loc[("storage_units","capital","hydro"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="hydro","p_nom"].sum()
    costs.loc[("storage_units","capital","PHS"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="PHS","p_nom"].sum()
    costs.loc[("generators","capital","ror"),label] = (0.02)*3e6*n.generators.loc[n.generators.group=="ror","p_nom"].sum()

    return costs



def calculate_curtailment(n,label,curtailment):

    avail = n.generators_t.p_max_pu.multiply(n.generators.p_nom_opt).sum().groupby(n.generators.group).sum()
    used = n.generators_t.p.sum().groupby(n.generators.group).sum()

    curtailment[label] = (((avail - used)/avail)*100).round(3)

    return curtailment

def calculate_energy(n,label,energy):

    for c in n.iterate_components(n.one_port_components|n.branch_components):

        if c.name in n.one_port_components:
            c_energies = c.pnl.p.sum().multiply(c.df.sign).groupby(c.df.group).sum()
        else:
            c_energies = (-c.pnl.p1.sum() - c.pnl.p0.sum()).groupby(c.df.group).sum()

        energy = energy.reindex(energy.index|pd.MultiIndex.from_product([[c.list_name],c_energies.index]))

        energy.loc[idx[c.list_name,list(c_energies.index)],label] = c_energies.values

    return energy


def calculate_supply(n,label,supply):
    """calculate the max dispatch of each component at the buses where the loads are attached"""

    load_types = n.loads.group.value_counts().index

    for i in load_types:

        buses = n.loads.bus[n.loads.group == i].values

        bus_map = pd.Series(False,index=n.buses.index)

        bus_map.loc[buses] = True

        for c in n.iterate_components(n.one_port_components):

            items = c.df.index[c.df.bus.map(bus_map)]

            if len(items) == 0:
                continue

            s = c.pnl.p[items].max().multiply(c.df.loc[items,'sign']).groupby(c.df.loc[items,'group']).sum()

            supply = supply.reindex(supply.index|pd.MultiIndex.from_product([[i],[c.list_name],s.index]))
            supply.loc[idx[i,c.list_name,list(s.index)],label] = s.values


        for c in n.iterate_components(n.branch_components):

            for end in ["0","1"]:

                items = c.df.index[c.df["bus" + end].map(bus_map)]

                if len(items) == 0:
                    continue

                #lots of sign compensation for direction and to do maximums
                s = (-1)**(1-int(end))*((-1)**int(end)*c.pnl["p"+end][items]).max().groupby(c.df.loc[items,'group']).sum()

                supply = supply.reindex(supply.index|pd.MultiIndex.from_product([[i],[c.list_name],s.index]))
                supply.loc[idx[i,c.list_name,list(s.index)],label] = s.values

    return supply

def calculate_supply_energy(n,label,supply_energy):
    """calculate the total dispatch of each component at the buses where the loads are attached"""

    load_types = n.loads.group.value_counts().index

    for i in load_types:

        buses = n.loads.bus[n.loads.group == i].values

        bus_map = pd.Series(False,index=n.buses.index)

        bus_map.loc[buses] = True

        for c in n.iterate_components(n.one_port_components):

            items = c.df.index[c.df.bus.map(bus_map)]

            if len(items) == 0:
                continue

            s = c.pnl.p[items].sum().multiply(c.df.loc[items,'sign']).groupby(c.df.loc[items,'group']).sum()

            supply_energy = supply_energy.reindex(supply_energy.index|pd.MultiIndex.from_product([[i],[c.list_name],s.index]))
            supply_energy.loc[idx[i,c.list_name,list(s.index)],label] = s.values


        for c in n.iterate_components(n.branch_components):

            for end in ["0","1"]:

                items = c.df.index[c.df["bus" + end].map(bus_map)]

                if len(items) == 0:
                    continue

                s = (-1)*c.pnl["p"+end][items].sum().groupby(c.df.loc[items,'group']).sum()

                supply_energy = supply_energy.reindex(supply_energy.index|pd.MultiIndex.from_product([[i],[c.list_name],s.index]))
                supply_energy.loc[idx[i,c.list_name,list(s.index)],label] = s.values

    return supply_energy

def calculate_metrics(n,label,metrics):

    metrics = metrics.reindex(metrics.index|pd.Index(["line_volume","line_volume_shadow","co2_shadow"]))

    metrics.at["line_volume",label] = (n.links.length*n.links.p_nom_opt)[n.links.group == "transmission lines"].sum()

    if "line_volume_limit" in n.shadow_prices.index:
        metrics.at["line_volume_shadow",label] = n.shadow_prices.at["line_volume_limit","value"]

    if "co2_constraint" in n.shadow_prices.index:
        metrics.at["co2_shadow",label] = n.shadow_prices.at["co2_constraint","value"]

    return metrics


def calculate_prices(n,label,prices):

    bus_type = pd.Series(n.buses.index.str[3:],n.buses.index).replace("","electricity")

    prices = prices.reindex(prices.index|bus_type.value_counts().index)

    #WARNING: this is time-averaged, should really be load-weighted average
    prices[label] = n.buses_t.marginal_price.mean().groupby(bus_type).mean()

    return prices



def calculate_weighted_prices(n,label,weighted_prices):
    # Warning: doesn't include storage units as loads


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

        #Add H2 Store when charging
        if carrier == "H2":
            stores = n.stores_t.p[buses+ " Store"].groupby(n.stores.loc[buses+ " Store","bus"],axis=1).sum(axis=1)
            stores[stores > 0.] = 0.
            load += -stores

        weighted_prices.loc[carrier,label] = (load*n.buses_t.marginal_price[buses]).sum().sum()/load.sum().sum()

        if carrier[:5] == "space":
            print(load*n.buses_t.marginal_price[buses])

    return weighted_prices




def calculate_market_values(n, label, market_values):
    # Warning: doesn't include storage units

    n.buses["suffix"] = n.buses.index.str[2:]

    suffix = ""

    buses = n.buses.index[n.buses.suffix == suffix]


    ## First do market value of generators ##

    generators = n.generators.index[n.buses.loc[n.generators.bus,"suffix"] == suffix]

    techs = n.generators.loc[generators,"carrier"].value_counts().index

    market_values = market_values.reindex(market_values.index | techs)


    for tech in techs:
        gens = generators[n.generators.loc[generators,"carrier"] == tech]

        dispatch = n.generators_t.p[gens].groupby(n.generators.loc[gens,"bus"],axis=1).sum().reindex(columns=buses,fill_value=0.)

        revenue = dispatch*n.buses_t.marginal_price[buses]

        market_values.at[tech,label] = revenue.sum().sum()/dispatch.sum().sum()



    ## Now do market value of links ##

    for i in ["0","1"]:
        all_links = n.links.index[n.buses.loc[n.links["bus"+i],"suffix"] == suffix]

        techs = n.links.loc[all_links,"group"].value_counts().index

        market_values = market_values.reindex(market_values.index | techs)

        for tech in techs:
            links = all_links[n.links.loc[all_links,"group"] == tech]

            dispatch = n.links_t["p"+i][links].groupby(n.links.loc[links,"bus"+i],axis=1).sum().reindex(columns=buses,fill_value=0.)

            revenue = dispatch*n.buses_t.marginal_price[buses]

            market_values.at[tech,label] = revenue.sum().sum()/dispatch.sum().sum()

    return market_values


def calculate_price_statistics(n, label, price_statistics):


    price_statistics = price_statistics.reindex(price_statistics.index|pd.Index(["zero_hours","mean","standard_deviation"]))

    n.buses["suffix"] = n.buses.index.str[2:]

    suffix = ""

    buses = n.buses.index[n.buses.suffix == suffix]

    
    threshold = 0.1 #higher than phoney marginal_cost of wind/solar

    df = pd.DataFrame(data=0.,columns=buses,index=n.snapshots)

    df[n.buses_t.marginal_price[buses] < threshold] = 1.

    price_statistics.at["zero_hours", label] = df.sum().sum()/(df.shape[0]*df.shape[1])

    price_statistics.at["mean", label] = n.buses_t.marginal_price[buses].unstack().mean()

    price_statistics.at["standard_deviation", label] = n.buses_t.marginal_price[buses].unstack().std()

    return price_statistics


outputs = ["costs",
           "curtailment",
           "energy",
           "supply",
           "supply_energy",
           "prices",
           "weighted_prices",
           "price_statistics",
           "market_values",
           "metrics",
           ]

def make_summaries(networks_dict):

    columns = pd.MultiIndex.from_tuples(networks_dict.keys(),names=["scenario","line_volume_limit","co2_reduction"])

    df = {}

    for output in outputs:
        df[output] = pd.DataFrame(columns=columns,dtype=float)

    for label, filename in iteritems(networks_dict):
        print(label, filename)

        n = helper.Network(filename)

        assign_groups(n)

        for output in outputs:
            df[output] = globals()["calculate_" + output](n, label, df[output])

    return df


def to_csv(df):

    for key in df:
        df[key].to_csv(snakemake.output[key])


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        with open('config.yaml') as f:
            snakemake.config = yaml.load(f)
        snakemake.input = Dict()
        snakemake.input['heat_demand_name'] = 'data/heating/daily_heat_demand.h5'
        snakemake.output = Dict()
        for item in outputs:
            snakemake.output[item] = snakemake.config['summary_dir'] + 'version-{version}/csvs/{item}.csv'.format(version=snakemake.config['version'],item=item)

    suffix = "nc" if int(snakemake.config['version']) > 100 else "h5"

    networks_dict = {(flexibility,line_limit,co2_reduction) :
                     '{results_dir}version-{version}/postnetworks/postnetwork-{flexibility}-{line_limit}-{co2_reduction}.{suffix}'\
                     .format(results_dir=snakemake.config['results_dir'],
                             version=snakemake.config['version'],
                             flexibility=flexibility,
                             line_limit=line_limit,
                             co2_reduction=co2_reduction,
                             suffix=suffix)\
                     for flexibility in snakemake.config['scenario']['flexibility'] for line_limit in snakemake.config['scenario']['line_limits']
                     for co2_reduction in snakemake.config['scenario']['co2_reduction']}

    options = yaml.load(open("options.yml","r"))

    with pd.HDFStore(snakemake.input.heat_demand_name, mode='r') as store:
        #the ffill converts daily values into hourly values
        heat_demand_df = store['heat_demand_profiles'].reindex(index=pd.date_range(options['tmin'],options['tmax'],freq='H'), method="ffill")


    intraday_profiles = pd.read_csv("data/heating/heat_load_profile_DK_AdamJensen.csv",index_col=0)

    intraday_year_profiles = generate_periodic_profiles(heat_demand_df.index.tz_localize("UTC"),weekly_profile=(list(intraday_profiles["weekday"])*5 + list(intraday_profiles["weekend"])*2)).tz_localize(None)

    heat_demand_df = heat_demand_df*intraday_year_profiles

    df = make_summaries(networks_dict)

    to_csv(df)
