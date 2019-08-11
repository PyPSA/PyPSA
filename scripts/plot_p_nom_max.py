"""
Plot renewable installation potentials per capacity factor

Relevant Settings
-----------------

Inputs
------

Outputs
-------

Description
-----------

"""

import pypsa
import pandas as pd
import matplotlib.pyplot as plt

def cum_p_nom_max(net, tech, country=None):
    carrier_b = net.generators.carrier == tech

    generators = \
    pd.DataFrame(dict(
        p_nom_max=net.generators.loc[carrier_b, 'p_nom_max'],
        p_max_pu=net.generators_t.p_max_pu.loc[:,carrier_b].mean(),
        country=net.generators.loc[carrier_b, 'bus'].map(net.buses.country)
    )).sort_values("p_max_pu", ascending=False)

    if country is not None:
        generators = generators.loc[generators.country == country]

    generators["cum_p_nom_max"] = generators["p_nom_max"].cumsum() / 1e6

    return generators


if __name__ == __main__:
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake, Dict
        snakemake = MockSnakemake(
            path='..',
            wildcards={'clusters': '45,90,181,full',
                       'country': 'all'},
            params=dict(techs=['onwind', 'offwind-ac', 'offwind-dc', 'solar']),
            input=Dict(
                **{
                    'full': 'networks/elec_s.nc',
                    '45': 'networks/elec_s_45.nc',
                    '90': 'networks/elec_s_90.nc',
                    '181': 'networks/elec_s_181.nc',
                }
            ),
            output=['results/plots/cum_p_nom_max_{clusters}_{country}.pdf']
        )

    logging.basicConfig(level=snakemake.config['logging_level'])

    plot_kwds = dict(drawstyle="steps-post")

    clusters = snakemake.wildcards.clusters.split(',')
    techs = snakemake.params.techs
    country = snakemake.wildcards.country
    if country == 'all':
        country = None
    else:
        plot_kwds['marker'] = 'x'

    fig, axes = plt.subplots(1, len(techs))

    for cluster in clusters:
        net = pypsa.Network(getattr(snakemake.input, cluster))

        for i, tech in enumerate(techs):
            cum_p_nom_max(net, tech, country).plot(x="p_max_pu", y="c_p_nom_max", label=cluster, ax=axes[0][i], **plot_kwds)

    for i, tech in enumerate(techs):
        ax = axes[0][i]
        ax.set_xlabel(f"Capacity factor of {tech}")
        ax.set_ylabel("Cumulative installable capacity / TW")

    plt.legend(title="Cluster level")

    fig.savefig(snakemake.output[0], transparent=True, bbox_inches='tight')
