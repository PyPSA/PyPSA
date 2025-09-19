"""Schema for plotting statistics in PyPSA.

This module defines default parameters for different statistics/plot combinations.
The schema system allows different default values based on the specific combination
of statistic function and plot type being used.

The module contains the following configuration dictionaries:

1. DEFAULTS:
   General default values for all statistics/plot combinations.

2. METHOD_OVERRIDES:
   Plot-type-specific defaults that override the general defaults.

3. ALLOWED_PARAMS:
   Additional parameters that are allowed for specific statistics functions.
   Parameters listed here are restricted by default and only enabled for the
   specified statistics.

4. EXCLUDED_PARAMS:
   Parameters that should be excluded for specific statistics functions
   across all plot types.

5. STAT_OVERRIDES:
   Statistic and plot-type specific overrides for parameter defaults.
"""

from pypsa.plot.statistics.charts import CHART_TYPES

# Base defaults for all parameters
DEFAULTS = {
    "x": "carrier",
    "y": "value",
    "color": "carrier",
    "height": 4,
    "aspect": 1,
    "bus_split_circles": False,
    "transmission_flow": False,
    "draw_legend_arrows": False,
    "draw_legend_lines": True,
    # general plot method
    "kind": "bar",
    # Optional parameters
    "storage": None,
    "direction": None,
}

# Method-specific overrides
METHOD_OVERRIDES: dict = {
    "area": {
        "x": "carrier",
        "y": "value",
        "color": None,
        "height": 3,
        "aspect": 2,
        "linewidth": 0,
    },
    "line": {"x": "carrier", "y": "value", "color": None, "height": 3, "aspect": 2},
    "bar": {"x": "value", "y": "carrier", "color": "carrier"},
}

# Additional allowed params per statistic
ALLOWED_PARAMS = {
    "optimal_capacity": ["storage"],
    "installed_capacity": ["storage"],
    "energy_balance": ["direction"],
    "revenue": ["direction"],
}

# Excluded params per statistic
EXCLUDED_PARAMS = {
    "prices": ["carrier", "nice_names"],
}

# Statistic-specific overrides per plot type
STAT_OVERRIDES: dict = {
    "optimal_capacity": {
        "line": {"storage": False},
        "area": {"storage": False},
        "plot": {"storage": False},
    },
    "installed_capacity": {
        "line": {"storage": False},
        "area": {"storage": False},
        "plot": {"storage": False},
    },
    "opex": {"line": {"x": "snapshot"}, "area": {"x": "snapshot", "color": "carrier"}},
    "supply": {
        "line": {"x": "snapshot"},
        "area": {"x": "snapshot", "color": "carrier"},
        "map": {
            "transmission_flow": True,
            "draw_legend_arrows": True,
            "draw_legend_lines": False,
            "kind": "supply",
        },
    },
    "withdrawal": {
        "line": {"x": "snapshot"},
        "area": {"x": "snapshot", "color": "carrier"},
        "map": {
            "transmission_flow": True,
            "draw_legend_arrows": True,
            "draw_legend_lines": False,
            "kind": "withdrawal",
        },
    },
    "transmission": {
        "line": {"x": "snapshot"},
        "area": {"x": "snapshot", "color": "carrier"},
    },
    "energy_balance": {
        "plot": {"kind": "area"},
        "line": {"x": "snapshot"},
        "area": {"x": "snapshot", "color": "carrier"},
        "map": {
            "bus_split_circles": True,
            "transmission_flow": True,
            "draw_legend_arrows": True,
            "draw_legend_lines": False,
        },
    },
    "curtailment": {
        "line": {"x": "snapshot"},
        "area": {"x": "snapshot", "color": "carrier"},
    },
    "capacity_factor": {
        "line": {"x": "snapshot"},
        "area": {"x": "snapshot", "color": "carrier"},
    },
    "revenue": {
        "line": {"x": "snapshot"},
        "area": {"x": "snapshot", "color": "carrier"},
    },
    "market_value": {
        "line": {"x": "snapshot"},
        "area": {"x": "snapshot", "color": "carrier"},
    },
    # Statistics with no special overrides
    "capex": {},
    "installed_capex": {},
    "expanded_capex": {},
    "expanded_capacity": {},
    "system_cost": {},
    "prices": {
        "area": {"x": "snapshot", "y": "value", "color": None},
        "line": {"x": "snapshot", "y": "value", "color": "name"},
        "bar": {"y": "name", "color": None},
        "box": {"x": "value", "y": "name", "color": None},
        "violin": {"x": "value", "y": "name", "color": None},
        "histogram": {"x": "value", "color": None},
        "scatter": {"x": "name", "color": None},
    },
}


def _combine_schemas() -> dict:
    """Build the complete schema by combining all configuration dictionaries.

    Returns
    -------
    dict
        Combined schema with defaults and allowed values for all combinations
        of statistics and plot types.

    """
    schema: dict = {}
    plot_types = ["map", "plot"] + CHART_TYPES
    all_stats = list(STAT_OVERRIDES.keys())

    # Gather all additional params to determine which are restricted by default
    restricted_params = {p for params in ALLOWED_PARAMS.values() for p in params}

    for stat in all_stats:
        schema[stat] = {}

        for plot_type in plot_types:
            schema[stat][plot_type] = {}

            # Start with global defaults
            for param, default in DEFAULTS.items():
                schema[stat][plot_type][param] = {
                    "default": default,
                    "allowed": param not in restricted_params,
                }

            # Apply method-specific defaults
            if plot_type in METHOD_OVERRIDES:
                for param, default in METHOD_OVERRIDES[plot_type].items():
                    schema[stat][plot_type][param] = {
                        "default": default,
                        "allowed": True,
                    }

            # Enable additional params for this stat
            for param in ALLOWED_PARAMS.get(stat, []):
                if param in schema[stat][plot_type]:
                    schema[stat][plot_type][param]["allowed"] = True

            # Apply stat-specific overrides
            if stat in STAT_OVERRIDES and plot_type in STAT_OVERRIDES[stat]:
                for param, value in STAT_OVERRIDES[stat][plot_type].items():
                    schema[stat][plot_type][param] = {"default": value, "allowed": True}

            # Apply exclusions
            for param in EXCLUDED_PARAMS.get(stat, []):
                if param in schema[stat][plot_type]:
                    schema[stat][plot_type][param]["allowed"] = False

    return schema


# Generate the schema
schema = _combine_schemas()


def apply_parameter_schema(
    stats_name: str, plot_name: str, kwargs: dict, context: dict | None = None
) -> dict:
    """Apply parameter schema to kwargs.

    Filters and sets default values for parameters based on the schema
    for the given statistics function and plot type combination.

    Parameters
    ----------
    stats_name : str
        Name of the statistics function.
    plot_name : str
        Name of the plot type.
    kwargs : dict
        Dictionary of keyword arguments to be filtered based on the schema.
    context : dict | None, optional
        Additional context for parameter processing (e.g., {"index_names": [...]})

    Returns
    -------
    dict
        Filtered dictionary of keyword arguments with defaults applied.

    """
    to_remove = []

    for param, value in kwargs.items():
        # Check if parameter is explicitly excluded for this statistic
        if param in EXCLUDED_PARAMS.get(stats_name, []):
            to_remove.append(param)
            continue

        if param not in schema[stats_name][plot_name]:
            continue

        # Check if parameter is not allowed and remove it
        if not schema[stats_name][plot_name][param]["allowed"]:
            to_remove.append(param)
            continue

        # Apply default if value is None
        if value is None:
            kwargs[param] = schema[stats_name][plot_name][param]["default"]

    for param in to_remove:
        kwargs.pop(param)

    # Auto-faceting logic
    if (
        context
        and context.get("index_names")
        and not kwargs.get("facet_col")
        and not kwargs.get("facet_row")
    ):
        index_names = context["index_names"]
        if len(index_names) == 1:
            kwargs["facet_col"] = index_names[0]
        elif len(index_names) >= 2:
            kwargs["facet_row"] = index_names[0]
            kwargs["facet_col"] = index_names[1]

    return kwargs


def get_relevant_plot_values(plot_kwargs: dict, context: dict | None = None) -> list:
    """Extract values relevant for statistics, excluding index names.

    Parameters
    ----------
    plot_kwargs : dict
        Plot keyword arguments
    context : dict | None
        Context containing index_names

    Returns
    -------
    list
        Values that should be passed to derive_statistic_parameters

    """
    index_names = context.get("index_names", []) if context else []
    relevant_keys = {"x", "y", "color", "facet_col", "facet_row"}
    values = [
        v
        for k, v in plot_kwargs.items()
        if k in relevant_keys and v not in index_names and v is not None
    ]
    return list(set(values))
