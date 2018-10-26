import pandas as pd
import numpy as np
from six import iteritems, iterkeys, itervalues

import pypsa

from add_electricity import load_costs, update_transmission_costs

def pdbcast(v, h):
    return pd.DataFrame(v.values.reshape((-1, 1)) * h.values,
                        index=v.index, columns=h.index)

def load_network(fn, tech_costs, config, combine_hydro_ps=True):
    opts = config['plotting']

    n = pypsa.Network(fn)

    n.loads["carrier"] = n.loads.bus.map(n.buses.carrier) + " load"
    n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)

    n.links["carrier"] = (n.links.bus0.map(n.buses.carrier) + "-" + n.links.bus1.map(n.buses.carrier))
    n.lines["carrier"] = "AC line"
    n.transformers["carrier"] = "AC transformer"

    n.lines['s_nom'] = n.lines['s_nom_min']
    n.links['p_nom'] = n.links['p_nom_min']

    if combine_hydro_ps:
        n.storage_units.loc[n.storage_units.carrier.isin({'PHS', 'hydro'}), 'carrier'] = 'hydro+PHS'

    # #if the carrier was not set on the heat storage units
    # bus_carrier = n.storage_units.bus.map(n.buses.carrier)
    # n.storage_units.loc[bus_carrier == "heat","carrier"] = "water tanks"

    for name in opts['heat_links'] + opts['heat_generators']:
        n.links.loc[n.links.index.to_series().str.endswith(name), "carrier"] = name

    Nyears = n.snapshot_weightings.sum()/8760.
    costs = load_costs(Nyears, tech_costs, config['costs'], config['electricity'])
    update_transmission_costs(n, costs)

    return n

def aggregate_p_nom(n):
    return pd.concat([
        n.generators.groupby("carrier").p_nom_opt.sum(),
        n.storage_units.groupby("carrier").p_nom_opt.sum(),
        n.links.groupby("carrier").p_nom_opt.sum(),
        n.loads_t.p.groupby(n.loads.carrier,axis=1).sum().mean()
    ])

def aggregate_p(n):
    return pd.concat([
        n.generators_t.p.sum().groupby(n.generators.carrier).sum(),
        n.storage_units_t.p.sum().groupby(n.storage_units.carrier).sum(),
        n.stores_t.p.sum().groupby(n.stores.carrier).sum(),
        -n.loads_t.p.sum().groupby(n.loads.carrier).sum()
    ])

def aggregate_e_nom(n):
    return pd.concat([
        (n.storage_units["p_nom_opt"]*n.storage_units["max_hours"]).groupby(n.storage_units["carrier"]).sum(),
        n.stores["e_nom_opt"].groupby(n.stores.carrier).sum()
    ])

def aggregate_p_curtailed(n):
    return pd.concat([
        ((n.generators_t.p_max_pu.sum().multiply(n.generators.p_nom_opt) - n.generators_t.p.sum())
         .groupby(n.generators.carrier).sum()),
        ((n.storage_units_t.inflow.sum() - n.storage_units_t.p.sum())
         .groupby(n.storage_units.carrier).sum())
    ])

def aggregate_costs(n, flatten=False, opts=None, existing_only=False):
    components = dict(Link=("p_nom", "p0"),
                      Generator=("p_nom", "p"),
                      StorageUnit=("p_nom", "p"),
                      Store=("e_nom", "p"),
                      Line=("s_nom", None),
                      Transformer=("s_nom", None))

    costs = {}
    for c, (p_nom, p_attr) in zip(
        n.iterate_components(iterkeys(components), skip_empty=False),
        itervalues(components)
    ):
        if not existing_only: p_nom += "_opt"
        costs[(c.list_name, 'capital')] = (c.df[p_nom] * c.df.capital_cost).groupby(c.df.carrier).sum()
        if p_attr is not None:
            p = c.pnl[p_attr].sum()
            if c.name == 'StorageUnit':
                p = p.loc[p > 0]
            costs[(c.list_name, 'marginal')] = (p*c.df.marginal_cost).groupby(c.df.carrier).sum()
    costs = pd.concat(costs)

    if flatten:
        assert opts is not None
        conv_techs = opts['conv_techs']

        costs = costs.reset_index(level=0, drop=True)
        costs = costs['capital'].add(
            costs['marginal'].rename({t: t + ' marginal' for t in conv_techs}),
            fill_value=0.
        )

    return costs
