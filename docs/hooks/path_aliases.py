# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""
Path aliases hook for MkDocs to enable simplified cross-references.

Registers aliases so that e.g. `pypsa.Network.static` resolves to the actual
documented identifier `pypsa.network.components.NetworkComponentsMixin.static`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.pages import Page

# Maps alias prefix → {source_class: page_slug}.
# Members of each source_class are registered as alias_prefix.member,
# pointing to page_slug/#source_class.member.
# Also used for display name substitution: source_class → alias_prefix in HTML.
ALIASES = {
    "pypsa.Network": {
        "pypsa.network.index.NetworkIndexMixin": "api/networks/indexing",
        "pypsa.network.components.NetworkComponentsMixin": "api/networks/components",
        "pypsa.network.io.NetworkIOMixin": "api/networks/io",
        "pypsa.network.transform.NetworkTransformMixin": "api/networks/transform",
        "pypsa.network.descriptors.NetworkDescriptorsMixin": "api/networks/descriptors",
        "pypsa.network.power_flow.NetworkPowerFlowMixin": "api/networks/power-flow",
        "pypsa.consistency.NetworkConsistencyMixin": "api/other/consistency",
    },
    "pypsa.Network.optimize": {
        "pypsa.optimization.OptimizationAccessor": "api/networks/optimize",
    },
    "pypsa.Network.cluster": {
        "pypsa.clustering.ClusteringAccessor": "api/networks/cluster",
    },
    "pypsa.Network.statistics": {
        "pypsa.statistics.StatisticsAccessor": "api/networks/statistics",
    },
    "pypsa.Network.plot": {
        "pypsa.plot.accessor.PlotAccessor": "api/networks/plot",
    },
    "pypsa.SubNetwork": {
        "pypsa.networks.SubNetwork": "api/networks/subnetwork",
    },
    "pypsa.Components": {
        "pypsa.components.Components": "api/components/components",
    },
    "pypsa.plot.PlotAccessor": {
        "pypsa.plot.accessor.PlotAccessor": "api/networks/plot",
    },
    "pypsa.plot.StatisticPlotter": {
        "pypsa.plot.statistics.plotter.StatisticPlotter": "api/networks/plot",
    },
    "pypsa.plot.StatisticInteractivePlotter": {
        "pypsa.plot.statistics.plotter.StatisticInteractivePlotter": "api/networks/plot",
    },
    "pypsa.statistics.Groupers": {
        "pypsa.statistics.grouping.Groupers": "api/networks/statistics",
    },
    "pypsa": {
        "pypsa._options.OptionsNode": "api/other/options",
    },
    "pypsa.options": {
        "pypsa._options.OptionsNode": "api/other/options",
    },
}

# Explicit overrides for references to attributes that are dynamically generated
# (e.g. based on old or new Components API)
EXPLICIT_ALIASES = {
    "pypsa.Components.static": "api/components/components/#pypsa.components.Components.df",
    "pypsa.Components.dynamic": "api/components/components/#pypsa.components.Components.pnl",
    "pypsa.Components.da": "api/components/components/#pypsa.components.Components.ds",
    "pypsa.Network.static": "api/networks/components/#pypsa.network.components.NetworkComponentsMixin.df",
    "pypsa.Network.dynamic": "api/networks/components/#pypsa.network.components.NetworkComponentsMixin.pnl",
}



def _import_class(full_path: str):
    """Import a class from its full module path."""
    module_path, class_name = full_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def on_post_page(output: str, *, page: Page, config: MkDocsConfig, **kwargs):
    """Post-process the rendered HTML to replace path displays."""
    for alias_prefix, sources in ALIASES.items():
        for class_path in sources:
            output = output.replace(class_path, alias_prefix)
    return output


def on_env(env, *, config: MkDocsConfig, **kwargs):
    """Register path aliases with autorefs for cross-references."""
    autorefs_plugin = None
    for plugin_name, plugin_instance in config.plugins.items():
        if plugin_name == "autorefs":
            autorefs_plugin = plugin_instance
            break

    if not autorefs_plugin:
        return env

    # Register explicit aliases (overrides for non-existent members)
    for alias_id, target_url in EXPLICIT_ALIASES.items():
        autorefs_plugin.register_url(alias_id, target_url)

    # Register dynamically discovered aliases (base classes + their members)
    for alias_prefix, sources in ALIASES.items():
        multiple_sources = len(sources) > 1
        for class_path, page_slug in sources.items():
            # Register the base class alias itself
            autorefs_plugin.register_url(
                alias_prefix, f"{page_slug}/#{class_path}"
            )

            try:
                cls = _import_class(class_path)
            except (ImportError, AttributeError):
                continue

            # When multiple classes map to the same prefix (e.g. Network mixins),
            # only use own members to avoid duplicates across pages.
            members = vars(cls) if multiple_sources else dir(cls)

            for name in members:
                if name.startswith("_") and name != "__call__":
                    continue
                alias_id = f"{alias_prefix}.{name}"
                target_url = f"{page_slug}/#{class_path}.{name}"
                autorefs_plugin.register_url(alias_id, target_url)

    return env
