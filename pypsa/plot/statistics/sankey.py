# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Sankey diagrams based on the energy balance of a network."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import to_rgb

from pypsa.plot.statistics.base import sanitize_mathtext
from pypsa.plot.statistics.charts import ChartGenerator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Network

HUB, TECH, SOURCE, SINK, SPLIT = "hub:", "tech:", "source:", "sink:", "split:"
GAINS, LOSSES, OTHER = "gains", "losses", "other"
NEUTRAL = "#b3b3b3"
RTOL = 1e-4
LINK_OPACITY = 0.5


def _throughput(flows: pd.DataFrame) -> pd.Series:
    """Get the larger of inflow and outflow of each node."""
    inflow = flows.groupby("target").value.sum()
    outflow = flows.groupby("source").value.sum()
    return pd.concat([inflow, outflow]).groupby(level=0).max()


def _aggregate_small(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Rename carriers and bus carriers below the threshold to `other`."""
    supply = df.value.clip(lower=0)

    def size(column: str) -> pd.Series:
        withdrawal = supply - df.value
        groups = df[column]
        return np.maximum(
            supply.groupby(groups).sum(), withdrawal.groupby(groups).sum()
        )

    limit = threshold * size("bus_carrier").max()
    for column in ["bus_carrier", "carrier"]:
        small = size(column)[lambda s: s < limit].index
        df[column] = df[column].where(~df[column].isin(small), OTHER)
    return df


def _raw_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the energy balance into flows between hubs and technologies.

    Carriers that only supply or only withdraw become sources and sinks. Branch
    components with inputs and outputs become technologies between hubs. The net
    withdrawal of transmission carriers is a loss.
    """
    by = ["carrier", "bus_carrier", "transmission", "branch"]
    df = df.groupby(by, as_index=False).value.sum()
    # Drop numerical noise.
    df = df[df.value.abs() > RTOL**2 * df.value.abs().max()]

    # Only branches with inputs and outputs are technologies between hubs.
    branch = df.branch & ~df.transmission
    net = df.value[branch].groupby(df.carrier[branch])
    imbalance = net.sum()[(net.min() < 0) & (net.max() > 0)]
    imbalance.index = TECH + imbalance.index

    supply = df.value > 0
    converter = branch & (TECH + df.carrier).isin(imbalance.index)
    kind = np.where(converter, TECH, np.where(supply, SOURCE, SINK))
    label = df.carrier.where(~df.transmission, np.where(supply, GAINS, LOSSES))
    node = kind + label
    hub = HUB + df.bus_carrier
    flows = pd.DataFrame(
        {
            "source": node.where(supply, hub),
            "target": hub.where(supply, node),
            "carrier": df.carrier,
            "value": df.value.abs(),
        }
    )

    # Technologies are balanced with losses or gains.
    imbalance = imbalance[imbalance.abs() > RTOL * _throughput(flows)[imbalance.index]]
    surplus = imbalance > 0
    techs = imbalance.index.to_series()
    balancing = pd.DataFrame(
        {
            "source": techs.where(~surplus, SOURCE + GAINS),
            "target": techs.where(surplus, SINK + LOSSES),
            "carrier": techs.str.removeprefix(TECH),
            "value": imbalance.abs(),
        }
    )
    return pd.concat([flows, balancing], ignore_index=True)


def _collapse(flows: pd.DataFrame, techs: set[str]) -> pd.DataFrame:
    """Replace technologies by direct flows from their inputs to their outputs.

    Each input is shared among the outputs in proportion to their size.
    """
    inside = flows.source.isin(techs) | flows.target.isin(techs)
    inflow = flows[flows.target.isin(techs)].rename(columns={"target": "tech"})
    outflow = flows[flows.source.isin(techs)]
    share = outflow.value / outflow.groupby("source").value.transform("sum")
    outflow = pd.DataFrame(
        {"tech": outflow.source, "target": outflow.target, "share": share}
    )
    links = inflow.merge(outflow, on="tech")
    links["value"] *= links.share
    return pd.concat([flows[~inside], links[flows.columns]], ignore_index=True)


def _check_balance(flows: pd.DataFrame) -> None:
    """Raise if a hub or a technology is not balanced."""
    inflow = flows.groupby("target").value.sum()
    outflow = flows.groupby("source").value.sum()
    imbalance = inflow.sub(outflow, fill_value=0)
    imbalance = imbalance[~imbalance.index.str.startswith((SOURCE, SINK))]
    imbalance = imbalance[imbalance.abs() > RTOL * inflow.max()]
    if not imbalance.empty:
        msg = f"Inflow and outflow differ for the nodes:\n{imbalance}"
        raise ValueError(msg)


def _cyclic_hubs(flows: pd.DataFrame) -> list[str]:
    """Get hubs to split so that the flows have no cycles.

    The smallest hub in a cycle is split until no cycle is left.
    """
    graph = nx.DiGraph(zip(flows.source, flows.target, strict=True))
    throughput = _throughput(flows)
    hubs: list[str] = []
    while True:
        cycles = [c for c in nx.strongly_connected_components(graph) if len(c) > 1]
        cyclic = [node for c in cycles for node in c if node.startswith(HUB)]
        if not cyclic:
            return hubs
        hub = min(cyclic, key=throughput.get)
        graph.remove_edges_from(list(graph.out_edges(hub)))
        hubs.append(hub)


def _split(flows: pd.DataFrame, hubs: list[str]) -> pd.DataFrame:
    """Show hubs as a node that only supplies and a node that only receives."""
    replace = {hub: SPLIT + hub.removeprefix(HUB) for hub in hubs}
    return flows.assign(source=flows.source.replace(replace))


def get_flows(
    n: Network,
    balance: pd.Series,
    bus_carrier: str | Sequence[str] | None = None,
    weights: dict[str, float] | None = None,
    merge: dict[str, str] | None = None,
    collapse: bool | Sequence[str] = False,
    split: bool | Sequence[str] = True,
    threshold: float = 0.01,
    nice_names: bool = True,
) -> pd.DataFrame:
    """Get the flows of a Sankey diagram from the energy balance.

    Parameters
    ----------
    n : pypsa.Network
        Network object.
    balance : pd.Series
        Energy balance of all bus carriers, grouped by carrier and bus carrier
        without nice names.
    bus_carrier, weights, merge, collapse, split, threshold, nice_names
        See [`pypsa.statistics.StatisticsAccessor.sankey`][].

    Returns
    -------
    pd.DataFrame
        Flows with the columns `source`, `target`, `carrier` and `value`. Node
        names have the prefix `hub:`, `tech:`, `source:`, `sink:` or `split:`.

    """
    if balance.index.nlevels != 3:
        msg = "Sankey diagrams do not support network collections."
        raise NotImplementedError(msg)

    nice = ChartGenerator(n).get_carrier_labels(nice_names=nice_names)

    def label(carrier: str) -> str:
        return nice.get(carrier, carrier)

    weights = {label(k): v for k, v in (weights or {}).items()}
    merge = {label(k): label(v) for k, v in (merge or {}).items()}

    def hub(bus_carrier: str) -> str:
        return merge.get(label(bus_carrier), label(bus_carrier))

    df = balance.rename("value").reset_index()
    df["carrier"] = df.carrier.where(df.carrier != "", df.component).map(label)
    df["value"] *= df.bus_carrier.map(label).map(weights).astype(float).fillna(1)
    df["bus_carrier"] = df.bus_carrier.map(hub)
    df["branch"] = df.component.isin(n.branch_components)
    # Branches within one hub only transmit.
    hubs = df[df.branch].groupby("carrier").bus_carrier.nunique()
    df["transmission"] = df.branch & df.carrier.map(hubs).eq(1)
    if isinstance(bus_carrier, str):
        bus_carrier = [bus_carrier]
    if bus_carrier is not None:
        df = df[df.bus_carrier.isin(map(hub, bus_carrier))]
    if df.empty:
        msg = "The energy balance is empty. Is the network solved?"
        raise ValueError(msg)

    flows = _raw_flows(_aggregate_small(df, threshold))
    techs = flows.source[flows.source.str.startswith(TECH)]
    if collapse is not True:
        techs = techs[techs.isin([TECH + label(name) for name in collapse or []])]
    flows = _collapse(flows, set(techs))
    by = ["source", "target", "carrier"]
    flows = flows.groupby(by, as_index=False, sort=False).value.sum()
    _check_balance(flows)

    if split is False:
        return flows
    if split is not True:
        nodes = [HUB + hub(name) for name in split]
        available = set(flows.source[flows.source.str.startswith(HUB)])
        if not available.issuperset(nodes):
            names = sorted(node.removeprefix(HUB) for node in available)
            msg = f"Cannot split {split}. Bus carriers with outflow are {names}."
            raise ValueError(msg)
        flows = _split(flows, nodes)
    return _split(flows, _cyclic_hubs(flows))


def _rgba(color: str, opacity: float) -> str:
    r, g, b = to_rgb(color)
    return f"rgba({255 * r:.0f},{255 * g:.0f},{255 * b:.0f},{opacity})"


def plot_sankey(
    n: Network,
    flows: pd.DataFrame,
    losses: str = "losses",
    gains: str = "gains",
    scale: float = 1.0,
    unit: str = "",
    annotate: bool = True,
    nice_names: bool = True,
) -> go.Figure:
    """Plot flows from [`get_flows`][pypsa.plot.statistics.sankey.get_flows].

    Parameters
    ----------
    n : pypsa.Network
        Network object. Its carriers define the colors.
    flows : pd.DataFrame
        Flows with the columns `source`, `target`, `carrier` and `value`.
    losses, gains, scale, unit, annotate, nice_names
        See [`pypsa.statistics.StatisticsAccessor.sankey`][].

    Returns
    -------
    plotly.graph_objects.Figure
        Sankey diagram.

    """
    nodes = pd.Index(pd.unique(flows[["source", "target"]].to_numpy().ravel()))
    names = nodes.str.partition(":").get_level_values(2)

    colors = ChartGenerator(n).get_carrier_colors(nice_names=nice_names)
    colors = {k: v for k, v in colors.items() if v}
    missing = pd.Index([*names, *flows.carrier]).unique().difference(pd.Index(colors))
    palette = sns.color_palette("colorblind", len(missing)).as_hex()
    colors |= dict(zip(missing, palette, strict=True))
    colors |= dict.fromkeys([GAINS, LOSSES, OTHER], NEUTRAL)

    labels = names.map(lambda name: {GAINS: gains, LOSSES: losses}.get(name, name))
    labels = labels.map(sanitize_mathtext)
    if annotate:
        values = _throughput(flows)[nodes] * scale
        # Three significant digits with thousands separator.
        values = values.map(lambda v: f"{float(f'{v:.3g}'):,f}".rstrip("0").rstrip("."))
        labels = (labels + " " + values.to_numpy() + " " + unit).str.strip()

    trace = go.Sankey(
        arrangement="snap",
        valueformat=",.3r",
        valuesuffix=f" {unit}".rstrip(),
        node={
            "label": labels,
            "color": [_rgba(colors[name], 1) for name in names],
            "line": {"width": 0},
        },
        link={
            "source": nodes.get_indexer(flows.source),
            "target": nodes.get_indexer(flows.target),
            "value": flows.value * scale,
            "label": flows.carrier.map(sanitize_mathtext),
            "color": [_rgba(colors[c], LINK_OPACITY) for c in flows.carrier],
        },
    )
    fig = go.Figure(trace)
    fig.update_layout(
        template="plotly_white", margin={"l": 50, "r": 50, "t": 50, "b": 50}
    )
    return fig
