# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""
Plots renewable installation potentials per capacity factor.

Relevant Settings
-----------------

Inputs
------

Outputs
-------

Description
-----------

"""
import logging
from _helpers import configure_logging

import pypsa
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def cum_p_nom_max(net, tech, country=None):
    carrier_b = net.generators.carrier == tech

    generators = pd.DataFrame(dict(
        p_nom_max=net.generators.loc[carrier_b, 'p_nom_max'],
        p_max_pu=net.generators_t.p_max_pu.loc[:,carrier_b].mean(),
        country=net.generators.loc[carrier_b, 'bus'].map(net.buses.country)
    )).sort_values("p_max_pu", ascending=False)

    if country is not None:
        generators = generators.loc[generators.country == country]

    generators["cum_p_nom_max"] = generators["p_nom_max"].cumsum() / 1e6

    return generators


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('plot_p_nom_max', network='elec', simpl='',
                                  techs='solar,onwind,offwind-dc', ext='png',
                                  clusts= '5,full', country= 'all')
    configure_logging(snakemake)

    plot_kwds = dict(drawstyle="steps-post")

    clusters = snakemake.wildcards.clusts.split(',')
    techs = snakemake.wildcards.techs.split(',')
    country = snakemake.wildcards.country
    if country == 'all':
        country = None
    else:
        plot_kwds['marker'] = 'x'

    fig, axes = plt.subplots(1, len(techs))

    for j, cluster in enumerate(clusters):
        net = pypsa.Network(snakemake.input[j])

        for i, tech in enumerate(techs):
            cum_p_nom_max(net, tech, country).plot(x="p_max_pu", y="cum_p_nom_max",
                         label=cluster, ax=axes[i], **plot_kwds)

    for i, tech in enumerate(techs):
        ax = axes[i]
        ax.set_xlabel(f"Capacity factor of {tech}")
        ax.set_ylabel("Cumulative installable capacity / TW")

    plt.legend(title="Cluster level")

    fig.savefig(snakemake.output[0], transparent=True, bbox_inches='tight')
