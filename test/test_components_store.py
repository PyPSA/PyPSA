# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

from pypsa.components.store import ComponentsStore


def test_init():
    # Dict like initialization
    components = ComponentsStore()
    components["a"] = "b"
    assert components["a"] == "b"
    components = ComponentsStore({"a": "b", "c": "d"})
    assert components["a"] == "b"
    assert components["c"] == "d"
