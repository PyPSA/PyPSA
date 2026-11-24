# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import logging

import pytest
import yaml
from mathspec.errors import LanguageError

import pypsa
from pypsa.optimization.spec import _cache
from pypsa.optimization.spec.build import _formulation, spec

HALF = {
    "dims": ["Generator", "scenario", "snapshot"],
    "where": "Generator_p_nom_extendable AND not Generator_committable AND Generator_active",
    "expression": "Generator_p <= 0.5 * Generator_p_max_pu * Generator_p_nom_ext",
}
OVERRIDE = {"constraints": {"Generator_ext_p_upper": None, "Generator_p_half": HALF}}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(_cache, "directory", lambda: tmp_path)
    monkeypatch.setattr(_cache, "_MEMORY", {})
    return tmp_path


def _as(form, tmp_path):
    if form == "mapping":
        return OVERRIDE
    if form == "text":
        return yaml.safe_dump(OVERRIDE)
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(OVERRIDE))
    return str(path) if form == "str" else path


@pytest.mark.parametrize("form", ["mapping", "text", "str", "path"])
def test_an_override_removes_and_adds_constraints(ac_dc_network, tmp_path, form):
    m = ac_dc_network.optimize.create_model(overrides=[_as(form, tmp_path)])
    assert "Generator-ext-p-upper" not in m.constraints
    assert "Generator_p_half" in m.constraints


def test_an_override_changes_the_optimum(ac_dc_network):
    n = ac_dc_network
    n.optimize()
    base = n.objective
    n.optimize(overrides=[OVERRIDE])
    assert n.objective > base


def test_an_added_parameter_reads_a_static_column(ac_dc_network):
    n = ac_dc_network
    n.c.generators.static["half"] = 0.5
    expression = "Generator_p <= Generator_half * Generator_p_nom_ext"
    override = {
        "parameters": {"Generator_half": {"dims": ["Generator"]}},
        "constraints": {"Generator_p_half": HALF | {"expression": expression}},
    }
    m = n.optimize.create_model(overrides=[override])
    assert "Generator_p_half" in m.constraints


def test_an_override_of_an_undeclared_entry_fails(ac_dc_network):
    with pytest.raises(LanguageError, match="Generator_nonexistent"):
        ac_dc_network.optimize.create_model(
            overrides=[{"constraints": {"Generator_nonexistent": None}}]
        )


@pytest.mark.parametrize("override", [42, ["constraints"]])
def test_an_override_of_another_type_fails(ac_dc_network, override):
    with pytest.raises(TypeError, match="YAML path, YAML text or a mapping"):
        ac_dc_network.optimize.create_model(overrides=[override])


def test_the_cache_key_follows_the_content():
    assert _cache.digest([OVERRIDE]) == _cache.digest([dict(OVERRIDE)])
    assert _cache.digest([OVERRIDE]) != _cache.digest([{"constraints": {}}])


def test_a_cached_value_is_read_from_disk_by_a_new_process(cache, monkeypatch):
    calls = []
    assert _cache.cached("k", lambda: calls.append(1) or "v") == "v"
    monkeypatch.setattr(_cache, "_MEMORY", {})
    assert _cache.cached("k", lambda: calls.append(1) or "w") == "v"
    assert calls == [1]
    assert [p.name for p in cache.iterdir()] == ["k.pkl"]


def test_an_unreadable_cached_value_is_made_again(cache, caplog):
    (cache / "k.pkl").write_bytes(b"not a pickle")
    with caplog.at_level(logging.WARNING):
        assert _cache.cached("k", lambda: "v") == "v"
    assert "unreadable" in caplog.text
    assert _cache._read(cache / "k.pkl") == "v"


def test_the_cache_keeps_the_most_recently_used_files(cache, monkeypatch):
    monkeypatch.setattr(_cache, "KEEP", 2)
    for key in "abc":
        _cache.cached(key, lambda: key)
    assert sorted(p.name for p in cache.iterdir()) == ["b.pkl", "c.pkl"]


def test_the_cache_stays_off_disk_when_disabled(cache):
    with pypsa.option_context("optimization.formulation_cache", False):
        _cache.cached("k", lambda: "v")
    assert not list(cache.iterdir())


def test_a_cached_formulation_equals_the_checked_one(cache, monkeypatch):
    checked = _formulation(overrides=[OVERRIDE])
    monkeypatch.setattr(_cache, "_MEMORY", {})
    read = _formulation(overrides=[OVERRIDE])
    assert read == checked
    assert len(list(cache.iterdir())) == 2
    assert "Generator_p_half" in spec(overrides=[OVERRIDE]).constraints
