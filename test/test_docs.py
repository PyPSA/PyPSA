# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import doctest
import importlib
import json
import pkgutil
import re
import sys
from pathlib import Path
from urllib.request import urlopen

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


rng = np.random.default_rng(42)


# Create a pytest fixture to check for the test-docs flag
@pytest.fixture(scope="session")
def test_docs_flag(pytestconfig):
    """Check if --test-docs flag is provided."""
    return pytestconfig.getoption("--test-docs", default=False)


sub_network_parent = pypsa.examples.ac_dc_meshed().determine_network_topology()
# Warning: Keep in sync with settings in doc/conf.py
n = pypsa.examples.ac_dc_meshed()
n.optimize(include_objective_constant=True)

# Create another network with shuffled load time series for collection comparisons
n_shuffled_load = pypsa.examples.ac_dc_meshed()
df = n.loads_t.p_set
flat_values = df.values.ravel()
shuffled_series = pd.Series(flat_values).sample(frac=1).values
df_shuffled = pd.DataFrame(
    shuffled_series.reshape(df.shape), index=df.index, columns=df.columns
)
n_shuffled_load.loads_t.p_set = df_shuffled
n_shuffled_load.name = "AC-DC-Meshed-Shuffled-Load"
n_shuffled_load.optimize(include_objective_constant=True)

# Remove solver model to allow copying
n.model.solver_model = None
n_shuffled_load.model.solver_model = None

# Create a network collection
nc = pypsa.NetworkCollection([n.copy(), n_shuffled_load.copy()])


doctest_globals = {
    "np": np,
    "pd": pd,
    "pypsa": pypsa,
    "n": n,
    "n_shuffled_load": n_shuffled_load,
    "n_stochastic": pypsa.examples.stochastic_network(),
    "nc": nc,
    "network_collection": nc,
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


def test_notebooks(test_docs_flag, pytestconfig):
    """Test and manage warning filter injection in Jupyter notebooks.

    This test validates that all notebooks have the correct warning filter cell
    injected as the first cell. When run with --fix-notebooks flag, it will
    automatically inject or update the warning filters for self-healing.
    """
    if not test_docs_flag:
        pytest.skip("Need --test-docs option to run documentation tests")

    fix_notebooks = pytestconfig.getoption("--fix-notebooks", default=False)
    expected_tags = ["injected-warnings", "hide-cell"]
    expected_source = [
        "# General notebook settings\n",
        "import logging\n",
        "import warnings\n",
        "\n",
        "import pypsa\n",
        "\n",
        'warnings.filterwarnings("error", category=DeprecationWarning)\n',
        'logging.getLogger("gurobipy").propagate = False\n',
        "pypsa.options.params.optimize.log_to_console = False",
    ]
    expected_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": expected_tags},
        "outputs": [],
        "source": expected_source,
    }

    notebook_paths = list(Path("docs").glob("**/*.ipynb"))
    if not notebook_paths:
        pytest.skip("No notebooks found to test")

    failed_notebooks = []
    injection_count = 0

    for notebook_path in notebook_paths:
        with notebook_path.open(encoding="utf-8") as f:
            notebook = json.load(f)

        cells = notebook.get("cells", [])
        if not cells:
            continue

        # Check if first cell matches expected warning filter
        first_cell = cells[0]
        is_correct = (
            first_cell.get("metadata", {}).get("tags") == expected_tags
            and first_cell.get("source") == expected_source
        )

        if not is_correct:
            if fix_notebooks:
                # Replace or insert warning cell
                if first_cell.get("metadata", {}).get("tags") == expected_tags:
                    cells[0] = expected_cell.copy()  # Update existing
                else:
                    cells.insert(0, expected_cell.copy())  # Insert new

                with notebook_path.open("w", encoding="utf-8") as f:
                    json.dump(notebook, f, indent=1, ensure_ascii=False)
                injection_count += 1
            else:
                failed_notebooks.append(notebook_path)

    if fix_notebooks and injection_count > 0:
        print(f"Fixed {injection_count} notebooks")

    if failed_notebooks and not fix_notebooks:
        pytest.fail(
            f"{len(failed_notebooks)} notebook(s) have missing or incorrect warning filters. "
            f"Run `pytest test/test_docs.py::test_notebooks --test-docs --fix-notebooks` and commit the changes."
        )


def _collect_go_urls():
    """Collect all unique go.pypsa.org URLs from the project."""
    pattern = re.compile(r"https://go\.pypsa\.org/[a-zA-Z0-9_-]+")
    urls = set()
    for fpath in Path("pypsa").glob("**/*.py"):
        urls.update(pattern.findall(fpath.read_text(encoding="utf-8")))
    for fpath in Path("docs").glob("**/*.md"):
        urls.update(pattern.findall(fpath.read_text(encoding="utf-8")))
    return sorted(urls)


@pytest.mark.parametrize("url", _collect_go_urls())
def test_go_links(url, test_docs_flag):
    """Test that all go.pypsa.org short-links resolve (no 404)."""
    if not test_docs_flag:
        pytest.skip("Need --test-docs option to run documentation tests")
    response = urlopen(url)  # noqa: S310
    assert response.status == 200, f"{url} returned {response.status}"
