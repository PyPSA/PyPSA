"""
Schema for plotting statistics in PyPSA.

Statistics can be plotted in any statistics/ plotting combination. E.g.
n.plot.supply.line() or n.plot.transmission.area(). Different combinations require
different default parameters. This schema defines default parameters if they are
different based on the plot type. If they do not differ, the signature's default
is used and no entry is made in the schema.

The module defines two dictionaries:
1. SCHEMA_DEFAULTS:
    {"<parameter_name>": {"default": <default_value>, "allowed": <allowed_values>}}
    - 'parameter_name': Name of the parameter
    - 'default': Default value used as replacement of signature default
    - 'allowed': List of allowed values for the parameter. If the value is not
        in the list, a ValueError is raised.
2. SCHEMA:
    {"<statistics_name>": {"<plot_type>": SCHEMA_DEFAULTS}}

Both dictionaries are combined to create a final schema which has a default and optional
allowed value list for each statistics/ plot combination.
"""

SCHEMA_DEFAULTS: dict = {
    # Defaults for required parameters
    "x": {"default": "carrier", "allowed": True},
    "bus_split_circles": {"default": False, "allowed": True},
    "transmission_flow": {"default": False, "allowed": True},
    "draw_legend_arrows": {"default": False, "allowed": True},
    "draw_legend_lines": {"default": True, "allowed": True},
    "kind": {"default": None, "allowed": True},
    # Only allow default for parameters which can only be used for specific
    # statistics/ plots
    "storage": {"default": None, "allowed": False},
}
SCHEMA: dict = {
    "capex": {},
    "installed_capex": {},
    "expanded_capex": {},
    "optimal_capacity": {
        "_all": {"storage": {"default": False, "allowed": True}},
    },
    "installed_capacity": {
        "_all": {"storage": {"default": False, "allowed": True}},
    },
    "expanded_capacity": {},
    "opex": {
        "line": {"x": {"default": "snapshot"}},
        "area": {"x": {"default": "snapshot"}},
    },
    "supply": {
        "line": {"x": {"default": "snapshot"}},
        "area": {"x": {"default": "snapshot"}},
        "map": {
            "transmission_flow": {"default": True},
            "draw_legend_arrows": {"default": True},
            "draw_legend_lines": {"default": False},
            "kind": {"default": "supply"},
        },
    },
    "withdrawal": {
        "line": {"x": {"default": "snapshot"}},
        "area": {"x": {"default": "snapshot"}},
        "map": {
            "transmission_flow": {"default": True},
            "draw_legend_arrows": {"default": True},
            "draw_legend_lines": {"default": False},
            "kind": {"default": "withdrawal"},
        },
    },
    "transmission": {
        "line": {"x": {"default": "snapshot"}},
        "area": {"x": {"default": "snapshot"}},
    },
    "energy_balance": {
        "line": {"x": {"default": "snapshot"}},
        "area": {"x": {"default": "snapshot"}},
        "map": {
            "bus_split_circles": {"default": True},
            "transmission_flow": {"default": True},
            "draw_legend_arrows": {"default": True},
            "draw_legend_lines": {"default": False},
        },
    },
    "curtailment": {
        "line": {"x": {"default": "snapshot"}},
        "area": {"x": {"default": "snapshot"}},
    },
    "capacity_factor": {
        "line": {"x": {"default": "snapshot"}},
        "area": {"x": {"default": "snapshot"}},
    },
    "revenue": {
        "line": {"x": {"default": "snapshot"}},
        "area": {"x": {"default": "snapshot"}},
    },
    "market_value": {
        "line": {"x": {"default": "snapshot"}},
        "area": {"x": {"default": "snapshot"}},
    },
}


def _combine_schemas() -> dict:
    """
    Combine the default schema with the specific schema for statistics and plot types.

    Parameters
    ----------
    schema_defaults : dict
        Dictionary containing default parameter values and allowed values
    schema : dict
        Dictionary containing specific parameter values for statistics and plot types

    Returns
    -------
    dict
        Combined schema with defaults and allowed values for all combinations

    Example
    -------
    >>> combined_schema = _combine_schemas()
    >>> print(combined_schema) # doctest: +ELLIPSIS
    {'capex': {'line': {'x': {'default': 'carrier', 'allowed': True}, 'bus_split_...

    """
    combined_schema: dict = {}
    plot_types = ["line", "area", "bar", "map"]

    # Initialize combined schema for each statistics
    for stat_name in SCHEMA.keys():
        combined_schema[stat_name] = {}

        # Add entries for each plot type
        for plot_type in plot_types:
            combined_schema[stat_name][plot_type] = {}

            # Add global defaults first
            for param, param_dict in SCHEMA_DEFAULTS.items():
                combined_schema[stat_name][plot_type][param] = param_dict.copy()

            # Add statistics-wide defaults ("_all" key)
            if "_all" in SCHEMA[stat_name]:
                for param, param_dict in SCHEMA[stat_name]["_all"].items():
                    if param not in combined_schema[stat_name][plot_type]:
                        combined_schema[stat_name][plot_type][param] = {}
                    combined_schema[stat_name][plot_type][param].update(param_dict)

            # Add plot-type specific values
            if plot_type in SCHEMA[stat_name]:
                for param, param_dict in SCHEMA[stat_name][plot_type].items():
                    if param not in combined_schema[stat_name][plot_type]:
                        combined_schema[stat_name][plot_type][param] = {}
                    combined_schema[stat_name][plot_type][param].update(param_dict)

    return combined_schema


schema = _combine_schemas()
