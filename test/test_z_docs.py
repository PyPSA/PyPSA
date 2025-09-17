import doctest
import importlib
import pkgutil
import sys

import numpy as np
import pandas as pd
import pytest

import pypsa

try:
    import cartopy  # noqa

    cartopy_available = True
except ImportError:
    cartopy_available = False

new_api = pypsa.options.api.new_components_api

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
    "sub_network": sub_network_parent.c.sub_networks.static.loc["0", "obj"],
}

modules = [
    importlib.import_module(name)
    for _, name, _ in pkgutil.walk_packages(pypsa.__path__, pypsa.__name__ + ".")
    if name not in ["pypsa.utils", "pypsa.components.utils", "pypsa.typing"]
]


@pytest.mark.skipif(
    sys.version_info[:2] == (3, 10),
    reason="Doctest fail until linopy supports numpy 2 on all python versions",
)
@pytest.mark.skipif(new_api, reason="New components API not yet shown in docs")
@pytest.mark.skipif(not cartopy_available, reason="Cartopy not available")
@pytest.mark.parametrize("module", modules)
def test_doctest(module):
    finder = doctest.DocTestFinder()

    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)

    tests = finder.find(module)

    failures = 0

    for test in tests:
        # Create a fresh copy of the globals for each test

        test_globals = dict(doctest_globals)

        test.globs.update(test_globals)

        # Run the test

        failures += runner.run(test).failed

    assert failures == 0, f"{failures} doctest(s) failed in module {module.__name__}"
