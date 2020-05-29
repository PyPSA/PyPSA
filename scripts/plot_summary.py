# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Plots energy and cost summaries for solved networks.

Relevant Settings
-----------------

Inputs
------

Outputs
-------

Description
-----------

"""

import os
import logging
logger = logging.getLogger(__name__)
from _helpers import configure_logging

import pandas as pd
import matplotlib.pyplot as plt

#consolidate and rename
def rename_techs(label):
    if label.startswith("central "):
        label = label[len("central "):]
    elif label.startswith("urban "):
        label = label[len("urban "):]

    if "retrofitting" in label:
        label = "building retrofitting"
    elif "H2" in label:
        label = "hydrogen storage"
    elif "CHP" in label:
        label = "CHP"
    elif "water tank" in label:
        label = "water tanks"
    elif label == "water tanks":
        label = "hot water storage"
    elif "gas" in label and label != "gas boiler":
        label = "natural gas"
    elif "solar thermal" in label:
        label = "solar thermal"
    elif label == "solar":
        label = "solar PV"
    elif label == "heat pump":
        label = "air heat pump"
    elif label == "Sabatier":
        label = "methanation"
    elif label == "offwind":
        label = "offshore wind"
    elif label == "offwind-ac":
        label = "offshore wind ac"
    elif label == "offwind-dc":
        label = "offshore wind dc"
    elif label == "onwind":
        label = "onshore wind"
    elif label == "ror":
        label = "hydroelectricity"
    elif label == "hydro":
        label = "hydroelectricity"
    elif label == "PHS":
        label = "hydroelectricity"
    elif label == "co2 Store":
        label = "DAC"
    elif "battery" in label:
        label = "battery storage"

    return label


preferred_order = pd.Index(["transmission lines","hydroelectricity","hydro reservoir","run of river","pumped hydro storage","onshore wind","offshore wind ac", "offshore wind dc","solar PV","solar thermal","building retrofitting","ground heat pump","air heat pump","resistive heater","CHP","OCGT","gas boiler","gas","natural gas","methanation","hydrogen storage","battery storage","hot water storage"])

def plot_costs(infn, fn=None):

    ## For now ignore the simpl header
    cost_df = pd.read_csv(infn,index_col=list(range(3)),header=[1,2,3])

    df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

    #convert to billions
    df = df/1e9

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[df.max(axis=1) < snakemake.config['plotting']['costs_threshold']]

    print("dropping")

    print(df.loc[to_drop])

    df = df.drop(to_drop)

    print(df.sum())

    new_index = (preferred_order&df.index).append(df.index.difference(preferred_order))

    new_columns = df.sum().sort_values().index

    fig, ax = plt.subplots()
    fig.set_size_inches((12,8))

    df.loc[new_index,new_columns].T.plot(kind="bar",ax=ax,stacked=True,color=[snakemake.config['plotting']['tech_colors'][i] for i in new_index])


    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_ylim([0,snakemake.config['plotting']['costs_max']])

    ax.set_ylabel("System Cost [EUR billion per year]")

    ax.set_xlabel("")

    ax.grid(axis="y")

    ax.legend(handles,labels,ncol=4,loc="upper left")


    fig.tight_layout()

    if fn is not None:
        fig.savefig(fn, transparent=True)


def plot_energy(infn, fn=None):

    energy_df = pd.read_csv(infn, index_col=list(range(2)),header=[1,2,3])

    df = energy_df.groupby(energy_df.index.get_level_values(1)).sum()

    #convert MWh to TWh
    df = df/1e6

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[df.abs().max(axis=1) < snakemake.config['plotting']['energy_threshold']]

    print("dropping")

    print(df.loc[to_drop])

    df = df.drop(to_drop)

    print(df.sum())

    new_index = (preferred_order&df.index).append(df.index.difference(preferred_order))

    new_columns = df.columns.sort_values()

    fig, ax = plt.subplots()
    fig.set_size_inches((12,8))

    df.loc[new_index,new_columns].T.plot(kind="bar",ax=ax,stacked=True,color=[snakemake.config['plotting']['tech_colors'][i] for i in new_index])


    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_ylim([snakemake.config['plotting']['energy_min'],snakemake.config['plotting']['energy_max']])

    ax.set_ylabel("Energy [TWh/a]")

    ax.set_xlabel("")

    ax.grid(axis="y")

    ax.legend(handles,labels,ncol=4,loc="upper left")


    fig.tight_layout()

    if fn is not None:
        fig.savefig(fn, transparent=True)


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('plot_summary', summary='energy', network='elec',
                                  simpl='', clusters=5, ll='copt', opts='Co2L-24H',
                                  attr='', ext='png', country='all')
    configure_logging(snakemake)

    summary = snakemake.wildcards.summary
    try:
        func = globals()[f"plot_{summary}"]
    except KeyError:
        raise RuntimeError(f"plotting function for {summary} has not been defined")

    func(os.path.join(snakemake.input[0], f"{summary}.csv"), snakemake.output[0])
