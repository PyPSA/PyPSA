import warnings

import pytest

import pypsa
from pypsa.components.store import ComponentsStore


def test_init():
    # Dict like initialization
    components = ComponentsStore()
    components["a"] = "b"
    assert components["a"] == "b"
    components = ComponentsStore({"a": "b", "c": "d"})
    assert components["a"] == "b"
    assert components["c"] == "d"


def test_deprecations():
    components = ComponentsStore()

    with pytest.raises(DeprecationWarning):
        "c" in components

    with pytest.warns(DeprecationWarning):
        for c in components:
            pass

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Convert warnings to errors

        with pypsa.option_context("warnings.components_store_iter", False):
            for c in components:
                pass
