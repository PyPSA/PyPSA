# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for runtime verification guards."""

import subprocess
import sys

import pypsa


def test_log_free_import():
    """Test that importing pypsa does not print anything."""
    code = "import pypsa"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.stdout == "", "No stdout should be produced"
    assert result.stderr == "", "No stderr should be produced"


def test_runtime_verification_enabled():
    """Test that runtime verification is enabled during tests."""
    assert pypsa.options.debug.runtime_verification is True, (
        "Runtime verification should be enabled during tests"
    )


def test_runtime_verification_disabled_by_default():
    """Test that runtime verification is disabled by default without conftest."""
    code = "import pypsa; print(pypsa.options.debug.runtime_verification)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    # Use last line to ignore solver license messages (e.g. Gurobi)
    last_line = result.stdout.strip().splitlines()[-1]
    assert last_line == "False", "Runtime verification should be False by default"
