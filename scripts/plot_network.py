# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Plots map with pie charts and cost box bar charts.

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
from _helpers import (load_network_for_plots, aggregate_p, aggregate_costs,
                      configure_logging)

import pandas as pd
import numpy as np

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Ellipse
from matplotlib.legend_handler import HandlerPatch
to_rgba = mpl.colors.colorConverter.to_rgba

logger = logging.getLogger(__name__)


def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
    fig = ax.get_figure()
    def axes2pt():
        return np.diff(ax.transData.transform([(0,0), (1,1)]), axis=0)[0] * (72./fig.dpi)

    ellipses = []
    if not dont_resize_actively:
        def update_width_height(event):
            dist = axes2pt()
            for e, radius in ellipses: e.width, e.height = 2. * radius * dist
        fig.canvas.mpl_connect('resize_event', update_width_height)
        ax.callbacks.connect('xlim_changed', update_width_height)
        ax.callbacks.connect('ylim_changed', update_width_height)

    def legend_circle_handler(legend, orig_handle, xdescent, ydescent,
                              width, height, fontsize):
        w, h = 2. * orig_handle.get_radius() * axes2pt()
        e = Ellipse(xy=(0.5*width-0.5*xdescent, 0.5*height-0.5*ydescent), width=w, height=w)
        ellipses.append((e, orig_handle.get_radius()))
        return e
    return {Circle: HandlerPatch(patch_func=legend_circle_handler)}


def make_legend_circles_for(sizes, scale=1.0, **kw):
    return [Circle((0,0), radius=(s/scale)**0.5, **kw) for s in sizes]


def set_plot_style():
    plt.style.use(['classic', 'seaborn-white',
                {'axes.grid': False, 'grid.linestyle': '--', 'grid.color': u'0.6',
                    'hatch.color': 'white',
                    'patch.linewidth': 0.5,
                    'font.size': 12,
                    'legend.fontsize': 'medium',
                    'lines.linewidth': 1.5,
                    'pdf.fonttype': 42,
                }])


def plot_map(n, ax=None, attribute='p_nom', opts={}):
    if ax is None:
        ax = plt.gca()

    ## DATA
    line_colors = {'cur': "purple",
                   'exp': mpl.colors.rgb2hex(to_rgba("red", 0.7), True)}
    tech_colors = opts['tech_colors']

    if attribute == 'p_nom':
        # bus_sizes = n.generators_t.p.sum().loc[n.generators.carrier == "load"].groupby(n.generators.bus).sum()
        bus_sizes = pd.concat((n.generators.query('carrier != "load"').groupby(['bus', 'carrier']).p_nom_opt.sum(),
                               n.storage_units.groupby(['bus', 'carrier']).p_nom_opt.sum()))
        line_widths_exp = dict(Line=n.lines.s_nom_opt, Link=n.links.p_nom_opt)
        line_widths_cur = dict(Line=n.lines.s_nom_min, Link=n.links.p_nom_min)
    else:
        raise 'plotting of {} has not been implemented yet'.format(attribute)


    line_colors_with_alpha = \
    dict(Line=(line_widths_cur['Line'] / n.lines.s_nom > 1e-3)
        .map({True: line_colors['cur'], False: to_rgba(line_colors['cur'], 0.)}),
        Link=(line_widths_cur['Link'] / n.links.p_nom > 1e-3)
        .map({True: line_colors['cur'], False: to_rgba(line_colors['cur'], 0.)}))

    ## FORMAT
    linewidth_factor = opts['map'][attribute]['linewidth_factor']
    bus_size_factor  = opts['map'][attribute]['bus_size_factor']

    ## PLOT
    n.plot(line_widths=pd.concat(line_widths_exp)/linewidth_factor,
           line_colors=dict(Line=line_colors['exp'], Link=line_colors['exp']),
           bus_sizes=bus_sizes/bus_size_factor,
           bus_colors=tech_colors,
           boundaries=map_boundaries,
           geomap=True,
           ax=ax)
    n.plot(line_widths=pd.concat(line_widths_cur)/linewidth_factor,
           line_colors=pd.concat(line_colors_with_alpha),
           bus_sizes=0,
           bus_colors=tech_colors,
           boundaries=map_boundaries,
           geomap=False,
           ax=ax)
    ax.set_aspect('equal')
    ax.axis('off')

    # Rasterize basemap
    # TODO : Check if this also works with cartopy
    for c in ax.collections[:2]: c.set_rasterized(True)

    # LEGEND
    handles = []
    labels = []

    for s in (10, 1):
        handles.append(plt.Line2D([0],[0],color=line_colors['exp'],
                                linewidth=s*1e3/linewidth_factor))
        labels.append("{} GW".format(s))
    l1_1 = ax.legend(handles, labels,
                     loc="upper left", bbox_to_anchor=(0.24, 1.01),
                     frameon=False,
                     labelspacing=0.8, handletextpad=1.5,
                     title='Transmission Exist./Exp.             ')
    ax.add_artist(l1_1)

    handles = []
    labels = []
    for s in (10, 5):
        handles.append(plt.Line2D([0],[0],color=line_colors['cur'],
                                linewidth=s*1e3/linewidth_factor))
        labels.append("/")
    l1_2 = ax.legend(handles, labels,
                loc="upper left", bbox_to_anchor=(0.26, 1.01),
                frameon=False,
                labelspacing=0.8, handletextpad=0.5,
                title=' ')
    ax.add_artist(l1_2)

    handles = make_legend_circles_for([10e3, 5e3, 1e3], scale=bus_size_factor, facecolor="w")
    labels = ["{} GW".format(s) for s in (10, 5, 3)]
    l2 = ax.legend(handles, labels,
                loc="upper left", bbox_to_anchor=(0.01, 1.01),
                frameon=False, labelspacing=1.0,
                title='Generation',
                handler_map=make_handler_map_to_scale_circles_as_in(ax))
    ax.add_artist(l2)

    techs =  (bus_sizes.index.levels[1]).intersection(pd.Index(opts['vre_techs'] + opts['conv_techs'] + opts['storage_techs']))
    handles = []
    labels = []
    for t in techs:
        handles.append(plt.Line2D([0], [0], color=tech_colors[t], marker='o', markersize=8, linewidth=0))
        labels.append(opts['nice_names'].get(t, t))
    l3 = ax.legend(handles, labels, loc="upper center",  bbox_to_anchor=(0.5, -0.), # bbox_to_anchor=(0.72, -0.05),
                handletextpad=0., columnspacing=0.5, ncol=4, title='Technology')

    return fig


def plot_total_energy_pie(n, ax=None):
    if ax is None: ax = plt.gca()

    ax.set_title('Energy per technology', fontdict=dict(fontsize="medium"))

    e_primary = aggregate_p(n).drop('load', errors='ignore').loc[lambda s: s>0]

    patches, texts, autotexts = ax.pie(e_primary,
        startangle=90,
        labels = e_primary.rename(opts['nice_names']).index,
        autopct='%.0f%%',
        shadow=False,
        colors = [opts['tech_colors'][tech] for tech in e_primary.index])
    for t1, t2, i in zip(texts, autotexts, e_primary.index):
        if e_primary.at[i] < 0.04 * e_primary.sum():
            t1.remove()
            t2.remove()

def plot_total_cost_bar(n, ax=None):
    if ax is None: ax = plt.gca()

    total_load = (n.snapshot_weightings * n.loads_t.p.sum(axis=1)).sum()
    tech_colors = opts['tech_colors']

    def split_costs(n):
        costs = aggregate_costs(n).reset_index(level=0, drop=True)
        costs_ex = aggregate_costs(n, existing_only=True).reset_index(level=0, drop=True)
        return (costs['capital'].add(costs['marginal'], fill_value=0.),
                costs_ex['capital'], costs['capital'] - costs_ex['capital'], costs['marginal'])

    costs, costs_cap_ex, costs_cap_new, costs_marg = split_costs(n)

    costs_graph = pd.DataFrame(dict(a=costs.drop('load', errors='ignore')),
                            index=['AC-AC', 'AC line', 'onwind', 'offwind-ac',
                                   'offwind-dc', 'solar', 'OCGT','CCGT', 'battery', 'H2']).dropna()
    bottom = np.array([0., 0.])
    texts = []

    for i,ind in enumerate(costs_graph.index):
        data = np.asarray(costs_graph.loc[ind])/total_load
        ax.bar([0.5], data, bottom=bottom, color=tech_colors[ind],
               width=0.7, zorder=-1)
        bottom_sub = bottom
        bottom = bottom+data

        if ind in opts['conv_techs'] + ['AC line']:
            for c in [costs_cap_ex, costs_marg]:
                if ind in c:
                    data_sub = np.asarray([c.loc[ind]])/total_load
                    ax.bar([0.5], data_sub, linewidth=0,
                        bottom=bottom_sub, color=tech_colors[ind],
                        width=0.7, zorder=-1, alpha=0.8)
                    bottom_sub += data_sub

        if abs(data[-1]) < 5:
            continue

        text = ax.text(1.1,(bottom-0.5*data)[-1]-3,opts['nice_names'].get(ind,ind))
        texts.append(text)

    ax.set_ylabel("Average system cost [Eur/MWh]")
    ax.set_ylim([0, opts.get('costs_max', 80)])
    ax.set_xlim([0, 1])
    ax.set_xticklabels([])
    ax.grid(True, axis="y", color='k', linestyle='dotted')


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('plot_network', network='elec', simpl='',
                                  clusters='5', ll='copt', opts='Co2L-24H',
                                  attr='p_nom', ext="pdf")
    configure_logging(snakemake)

    set_plot_style()

    opts = snakemake.config['plotting']
    map_figsize = opts['map']['figsize']
    map_boundaries = opts['map']['boundaries']

    n = load_network_for_plots(snakemake.input.network, snakemake.input.tech_costs, snakemake.config)

    scenario_opts = snakemake.wildcards.opts.split('-')

    fig, ax = plt.subplots(figsize=map_figsize, subplot_kw={"projection": ccrs.PlateCarree()})
    plot_map(n, ax, snakemake.wildcards.attr, opts)

    fig.savefig(snakemake.output.only_map, dpi=150, bbox_inches='tight')

    ax1 = fig.add_axes([-0.115, 0.625, 0.2, 0.2])
    plot_total_energy_pie(n, ax1)

    ax2 = fig.add_axes([-0.075, 0.1, 0.1, 0.45])
    plot_total_cost_bar(n, ax2)

    ll = snakemake.wildcards.ll
    ll_type = ll[0]
    ll_factor = ll[1:]
    lbl = dict(c='line cost', v='line volume')[ll_type]
    amnt = '{ll} x today\'s'.format(ll=ll_factor) if ll_factor != 'opt' else 'optimal'
    fig.suptitle('Expansion to {amount} {label} at {clusters} clusters'
                .format(amount=amnt, label=lbl, clusters=snakemake.wildcards.clusters))

    fig.savefig(snakemake.output.ext, transparent=True, bbox_inches='tight')
