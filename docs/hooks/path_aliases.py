# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""
Path aliases hook for MkDocs to enable simplified cross-references.

This module provides MkDocs hooks to register simplified path aliases with autorefs,
allowing cross-references like [pypsa.get_option][] to work while keeping internal
paths unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.pages import Page

# Path mappings - real path to simplified display name
PATH_MAPPINGS = {
    # Options API
    "pypsa._options.option_context": "pypsa.option_context",
    "pypsa._options.OptionsNode.get_option": "pypsa.get_option",
    "pypsa._options.OptionsNode.set_option": "pypsa.set_option",
    "pypsa._options.OptionsNode.reset_option": "pypsa.reset_option",
    "pypsa._options.OptionsNode": "pypsa.options",
    # Network Mixins
    "pypsa.consistency.NetworkConsistencyMixin": "pypsa.Network",
    "pypsa.network.index.NetworkIndexMixin": "pypsa.Network",
    "pypsa.network.components.NetworkComponentsMixin": "pypsa.Network",
    "pypsa.network.transform.NetworkTransformMixin": "pypsa.Network",
    "pypsa.network.descriptors.NetworkDescriptorsMixin": "pypsa.Network",
    "pypsa.network.io.NetworkIOMixin": "pypsa.Network",
    "pypsa.network.power_flow.NetworkPowerFlowMixin": "pypsa.Network",
    # Network Accesors
    "pypsa.optimization.OptimizationAccessor": "pypsa.Network.optimize",
    "pypsa.clustering.ClusteringAccessor": "pypsa.Network.cluster",
    "pypsa.statistics.StatisticsAccessor": "pypsa.Network.statistics",
    "pypsa.plot.accessor.PlotAccessor": "pypsa.Network.plot",
    # Plot Accessors
    "pypsa.plot.statistics.plotter": "pypsa.plot",
    # SubNetwork
    "pypsa.networks.SubNetwork": "pypsa.SubNetwork",
    # Components
    "pypsa.components.Components": "pypsa.Components",
    # Groupers
    "pypsa.statistics.grouping.Groupers": "pypsa.statistics.Groupers",
}


def on_post_page(output: str, *, page: Page, config: MkDocsConfig, **kwargs):
    """Post-process the rendered HTML to replace path displays."""
    for original_path, simplified_path in PATH_MAPPINGS.items():
        output = output.replace(original_path, simplified_path)
    return output


def on_env(env, *, config: MkDocsConfig, **kwargs):
    """Register path aliases with autorefs for cross-references."""
    # Get the autorefs plugin
    autorefs_plugin = None
    for plugin_name, plugin_instance in config.plugins.items():
        if plugin_name == "autorefs":
            autorefs_plugin = plugin_instance
            break

    if not autorefs_plugin:
        return env

    # Register aliases for commonly used methods/attributes
    common_suffixes = [
        "__getattr__",
        "__setattr__",
        "__init__",
        "__call__",
        "get",
        "set",
        "add",
        "remove",
        "update",
        "clear",
        "load",
        "save",
        "export",
        "import_",
        "describe",
        "to_dict",
        "from_dict",
        "copy",
        "keys",
        "values",
        "items",
    ]

    for original_path, simplified_path in PATH_MAPPINGS.items():
        # Register the base class/module alias
        try:
            original_url, title = autorefs_plugin.get_item_url(original_path)
            autorefs_plugin.register_url(simplified_path, original_url)
            print(f"Registered alias: {simplified_path} -> {original_url}")
        except KeyError:
            print(f"Could not find original URL for: {original_path}")

        # Register common method/attribute aliases
        for suffix in common_suffixes:
            full_original = f"{original_path}.{suffix}"
            full_simplified = f"{simplified_path}.{suffix}"
            try:
                original_url, title = autorefs_plugin.get_item_url(full_original)
                autorefs_plugin.register_url(full_simplified, original_url)
                print(f"Registered alias: {full_simplified} -> {original_url}")
            except KeyError:
                # This is expected for many methods that don't exist
                pass

    return env
