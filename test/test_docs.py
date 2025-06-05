import doctest
import importlib
import pkgutil
import re
import shutil
import subprocess
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


@pytest.mark.skipif(
    sys.version_info[:2] == (3, 10),
    reason="Doctest fail until linopy supports numpy 2 on all python versions",
)
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


@pytest.mark.skip(reason="Currently broken and not catching all warnings")  # TODO
@pytest.mark.test_sphinx_build
def test_sphinx_build(pytestconfig):
    if not pytestconfig.getoption("--test-docs-build"):
        pytest.skip("need --test-docs-build option to run")

    source_dir = Path("doc")
    build_dir = Path("doc") / "_build"
    # List of warnings to ignore during the build
    # (Warning message, number of subsequent lines to ignore)
    warnings_to_ignore = [
        (r"WARNING: cannot cache unpickable", 0),
        (r"DeprecationWarning: Jupyter is migrating its paths", 5),
        (r"RemovedInSphinx90Warning", 1),
        (r"DeprecationWarning: nodes.reprunicode", 1),
        (r"DeprecationWarning: The `docutils.utils.error_reporting` module is", 2),
        (r"UserWarning: resource_tracker", 1),
        (r"WARNING: The file [^ ]+ couldn't be copied\. Error:", 1),
    ]

    shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    # Build the documentation
    try:
        subprocess.run(
            [
                "sphinx-build",
                "-W",
                "--keep-going",
                "-b",
                "html",
                str(source_dir),
                str(build_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        lines = e.stderr.splitlines()
        # Save lines to file for debugging
        with Path("sphinx_build_stderr.txt").open("w") as f:
            f.write("\n".join(lines))

        filtered_stderr = []
        i = 0

        while i < len(lines):
            # Check if the line contains any of the warnings to ignore
            for warning, ignore_lines in warnings_to_ignore:
                if re.search(warning, lines[i]):
                    # Skip current line and specified number of subsequent lines
                    i += ignore_lines + 1
                    break
            else:
                filtered_stderr.append(lines[i])
                i += 1

        with Path("sphinx_build_stderr_filtered.txt").open("w") as f:
            f.write("\n".join(filtered_stderr))

        if filtered_stderr:
            pytest.fail(
                "Sphinx build failed with warnings:\n" + "\n".join(filtered_stderr)
            )

    # Check if the build was successful by looking for an index.html file
    assert (build_dir / "index.html").exists(), "Build failed: index.html not found"
