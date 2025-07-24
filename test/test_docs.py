import importlib
import pkgutil

import numpy as np
import pandas as pd

import pypsa

try:
    import cartopy  # noqa

    cartopy_available = True
except ImportError:
    cartopy_available = False

sub_network_parent = pypsa.examples.ac_dc_meshed().determine_network_topology()
# Warning: Keep in sync with settings in doc/conf.py
n = pypsa.examples.ac_dc_meshed()
n.optimize()

doctest_globals = {
    "np": np,
    "pd": pd,
    "pypsa": pypsa,
    "n": n,
    "c": pypsa.examples.ac_dc_meshed().components.generators,
    "sub_network_parent": pypsa.examples.ac_dc_meshed().determine_network_topology(),
    "sub_network": sub_network_parent.sub_networks.loc["0", "obj"],
}

modules = [
    importlib.import_module(name)
    for _, name, _ in pkgutil.walk_packages(pypsa.__path__, pypsa.__name__ + ".")
    if name not in ["pypsa.utils", "pypsa.components.utils", "pypsa.typing"]
]
