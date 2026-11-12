# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest

import pypsa
from pypsa.optimization.spec._render import formulated
from pypsa.optimization.spec.build import SPEC
from pypsa.optimization.spec.sources import _SOURCES, _providers, source


def test_every_declared_trait_has_one_file():
    declared = {t for c in pypsa.Network().components.values() for t in c._formulation}
    stated = {path.name.removesuffix(".yaml") for path in (SPEC / "traits").iterdir()}
    assert declared == stated


@pytest.mark.parametrize("key", ["snapshot_period", "${C}_active"])
def test_a_source_key_takes_one_derivation(key):
    with pytest.raises(ValueError, match="already"):
        source(key)(lambda s, name: None)


def test_a_source_name_takes_one_derivation(monkeypatch):
    monkeypatch.setitem(_SOURCES, "Generator_active", lambda s, name: None)
    with pytest.raises(ValueError, match="Generator_active has two derivations"):
        _providers(formulated(pypsa.Network()))
