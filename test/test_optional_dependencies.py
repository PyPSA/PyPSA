import builtins

import pytest


@pytest.fixture
def hide_pytables(monkeypatch):
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == "tables":
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)


@pytest.mark.usefixtures("hide_pytables")
def test_message(ac_dc_network, hide_pytables):
    n = ac_dc_network

    with pytest.raises(
        ImportError,
        match=r"Missing optional dependencies to use HDF5 files\.",
    ):
        n.export_to_hdf5("test.h5")
