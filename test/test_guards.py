# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for runtime verification guards."""

import subprocess
import sys

import pypsa


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
    assert result.stdout.strip() == "False", (
        "Runtime verification should be False by default"
    )
