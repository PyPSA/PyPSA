import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.test_docs
def test_sphinx_build(pytestconfig):
    if not pytestconfig.getoption("--test-docs"):
        pytest.skip("need --test-docs option to run")

    source_dir = Path("doc")
    build_dir = Path("doc") / "_build"
    # List of warnings to ignore during the build
    # (Warning message, number of subsequent lines to ignore)
    warnings_to_ignore = [
        ("WARNING: cannot cache unpickable", 0),
        ("DeprecationWarning: Jupyter is migrating its paths", 5),
        ("RemovedInSphinx90Warning", 1),
        ("DeprecationWarning: nodes.reprunicode", 1),
        ("DeprecationWarning: The `docutils.utils.error_reporting` module is", 2),
        ("UserWarning: resource_tracker", 1),
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
        with open("sphinx_build_stderr.txt", "w") as f:
            f.write("\n".join(lines))

        filtered_stderr = []
        i = 0

        while i < len(lines):
            # Check if the line contains any of the warnings to ignore
            for warning, ignore_lines in warnings_to_ignore:
                if warning in lines[i]:
                    # Skip current line and specified number of subsequent lines
                    i += ignore_lines + 1
                    break
            else:
                filtered_stderr.append(lines[i])
                i += 1

        with open("sphinx_build_stderr_filtered.txt", "w") as f:
            f.write("\n".join(filtered_stderr))

        if filtered_stderr:
            pytest.fail(
                "Sphinx build failed with warnings:\n" + "\n".join(filtered_stderr)
            )

    # Check if the build was successful by looking for an index.html file
    assert (build_dir / "index.html").exists(), "Build failed: index.html not found"
