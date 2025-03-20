"""
Schema for plotting statistics in PyPSA.

Statistics can be plotted in any statistics/ plotting combination. E.g.
n.plot.supply.line() or n.plot.transmission.area(). Different combinations require
different default parameters. This schema defines default parameters if they are
different based on the plot type. If they do not differ, the signature's default
is used and no entry is made in the schema.

The module defines two dictionaries:
1. SCHEMA_DEFAULTS:
    {"<parameter_name>": <default_value>}
    General default values for all statistics/ plot combinations. Will be used if
    no specific overwrite value is defined in the SCHEMA dictionary.
2. SCHEMA_ADDITIONAL_PARAMETERS:
    {"<statistics_name>": <allowed_boolean>}
    Different statistics functions can have different signatures, therefore not all
    parameters are allowed for all statistics functions. When a statistics function
    is set to False here, all methods will raise an error if the parameter is
    provided. It can be used only if also defined in the SCHEMA dictionary.
    For arguments of the plotting functions, the allowed values do not need to be
    defined, since they are already defined in the function signature.
    {"<parameter_name>": <allowed_value_list>}
2. SCHEMA:
    {"<statistics_name>": {"<plot_type>": {"<parameter_name>": <default_value>}}}
    The schema for each statistics function and plot type. The default values are
    defined in the SCHEMA_DEFAULTS dictionary and overwritten here if they are
    different for the specific statistics/ plot combination.

Both dictionaries are combined to create a final schema which has a default and optional
allowed value list for each statistics/ plot combination.
"""

SCHEMA_DEFAULTS: dict = {
    # Defaults for required parameters
    "x": "carrier",
    "bus_split_circles": False,
    "transmission_flow": False,
    "draw_legend_arrows": False,
    "draw_legend_lines": True,
    # general plot method
    "kind": "bar",
    # Only allow default for parameters which can only be used for specific
    # statistics/ plots
    # Optional parameters
    "storage": None,
    "direction": None,
}

SCHEMA_ADDITIONAL_PARAMETERS: dict = {
    "optimal_capacity": ["storage"],
    "installed_capacity": ["storage"],
    "energy_balance": ["direction"],
    "revenue": ["direction"],
}

SCHEMA: dict = {
    "capex": {
        "area": {"x": "carrier", "y": "value", "color": None},
    },
    "installed_capex": {
        "line": {"x": "carrier", "y": "value", "color": None},
        "area": {"x": "carrier", "y": "value", "color": None},
    },
    "expanded_capex": {
        "area": {"x": "carrier", "y": "value", "color": None},
    },
    "optimal_capacity": {
        "line": {"storage": False},
        "area": {"x": "carrier", "y": "value", "color": None, "storage": False},
        "plot": {"storage": False},
    },
    "installed_capacity": {
        "line": {"storage": False},
        "area": {"x": "carrier", "y": "value", "color": None, "storage": False},
        "plot": {"storage": False},
    },
    "expanded_capacity": {
        "area": {"x": "carrier", "y": "value", "color": None},
    },
    "opex": {
        "area": {"x": "snapshot", "color": "carrier"},
    },
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
}


def _combine_schemas() -> dict:
    """
    Combine the default schema with the specific schema for statistics and plot types.

    Returns
    -------
    dict
        Combined schema with defaults and allowed values for all combinations

    """
    combined_schema: dict = {}
    plot_types = ["line", "area", "bar", "map", "plot"]

    additional_parameters = {
        value
        for param_list in SCHEMA_ADDITIONAL_PARAMETERS.values()
        for value in param_list
    }

    for stat_name in SCHEMA.keys():
        combined_schema[stat_name] = {}

        for plot_type in plot_types:
            combined_schema[stat_name][plot_type] = {}

            # Add global defaults first
            for param, default_value in SCHEMA_DEFAULTS.items():
                allowed_by_default = param not in additional_parameters
                combined_schema[stat_name][plot_type][param] = {
                    "default": default_value,
                    "allowed": allowed_by_default,
                }

                # Allow if additional parameters selection again
                if param in SCHEMA_ADDITIONAL_PARAMETERS.get(stat_name, []):
                    combined_schema[stat_name][plot_type][param]["allowed"] = True

            # Override with specific values from SCHEMA
            sub_schema = SCHEMA[stat_name].get(plot_type, {})
            for param, value in sub_schema.items():
                combined_schema[stat_name][plot_type][param] = {
                    "default": value,
                    "allowed": True,
                }

    return combined_schema


schema = _combine_schemas()


def apply_parameter_schema(stats_name: str, plot_name: str, kwargs: dict) -> dict:
    """
    Apply parameter schema to kwargs.

    The schema is used to for different statistics functions signatures based on
    plot type/ choosed statistics function. The schema is defined in
    :mod:`pypsa.plot.statistics.schema`.

    Parameters
    ----------
    stats_name : str
        Name of the statistics function.
    plot_name : str
        Name of the plot type.
    kwargs : dict
        Dictionary of keyword arguments to be filtered based on the schema.

    Returns
    -------
    dict
        Filtered dictionary of keyword arguments.

    """
    to_remove = []
    # Filter kwargs based on different statistics functions signatures
    for param, value in kwargs.items():
        if param not in schema[stats_name][plot_name]:
            continue
        if value is None:
            kwargs[param] = schema[stats_name][plot_name][param]["default"]
            if not schema[stats_name][plot_name][param]["allowed"]:
                to_remove.append(param)
        else:
            if not schema[stats_name][plot_name][param]["allowed"]:
                msg = f"Parameter {param} can not be used for {stats_name} {plot_name}."
                raise ValueError(msg)

    for param in to_remove:
        kwargs.pop(param)

    return kwargs
