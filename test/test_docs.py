import doctest
import importlib
import pkgutil
import sys
from pathlib import Path

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


# Create a pytest fixture to check for the test-docs flag
@pytest.fixture(scope="session")
def test_docs_flag(pytestconfig):
    """Check if --test-docs flag is provided."""
    return pytestconfig.getoption("--test-docs", default=False)


sub_network_parent = pypsa.examples.ac_dc_meshed().determine_network_topology()
# Warning: Keep in sync with settings in doc/conf.py
n = pypsa.examples.ac_dc_meshed()
n.optimize()


doctest_globals = {
    "np": np,
    "pd": pd,
    "pypsa": pypsa,
    "n": n,
    "n_stochastic": pypsa.examples.stochastic_network(),
    "network_collection": pypsa.NetworkCollection(
        [
            pypsa.examples.ac_dc_meshed(),
            pypsa.examples.storage_hvdc(),
        ]
    ),
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
def test_doctest_code(module, close_matplotlib_figures, test_docs_flag):
    if not test_docs_flag:
        pytest.skip("Need --test-docs option to run documentation tests")
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


@pytest.mark.skipif(
    sys.version_info[:2] == (3, 10),
    reason="Doctest fail until linopy supports numpy 2 on all python versions",
)
@pytest.mark.skipif(new_api, reason="New components API not yet shown in docs")
@pytest.mark.skipif(not cartopy_available, reason="Cartopy not available")
@pytest.mark.parametrize(
    "fpath", [*Path("docs").glob("**/*.md"), Path("README.md")], ids=str
)
def test_doctest_docs(fpath, test_docs_flag):
    """Test Python code blocks in markdown files using doctest."""
    if not test_docs_flag:
        pytest.skip("Need --test-docs option to run documentation tests")
    import re

    # Read the markdown file
    content = fpath.read_text()

    # Extract Python code blocks
    python_blocks = re.findall(r"``` py\n(.*?)\n```", content, re.DOTALL)

    if not python_blocks:
        return  # No Python code blocks to test

    # Combine all Python blocks into one docstring-like content
    combined_content = "\n\n".join(python_blocks)

    # Create a pseudo-module with the combined content as docstring
    class PseudoModule:
        def __init__(self, content):
            self.__doc__ = content
            self.__name__ = str(fpath)

    module = PseudoModule(combined_content)

    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    tests = finder.find(module)
    failures = 0

    for test in tests:
        # Create a fresh copy of the globals for each test
        test_globals = dict(doctest_globals)

        # For docs files, use a fresh network to avoid optimization artifacts
        if str(fpath).startswith("docs/"):
            test_globals["n"] = pypsa.examples.ac_dc_meshed()

        test.globs.update(test_globals)

        # Run the test
        failures += runner.run(test).failed

    assert failures == 0, f"{failures} doctest(s) failed in {fpath}"
