# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""What the composite types and the stored overrides of a network share."""

import json
from collections.abc import Mapping
from importlib.util import find_spec

import pytest
import xarray as xr
from mathspec.errors import LanguageError

import pypsa


class Types:
    noun = "composite type"

    def __init__(self, n):
        self.n = n
        self.registry = n.composites

    @staticmethod
    def content(name, on=None, changed=False, wrong=False):
        up = "top" if changed else "up"
        tie = f"at(Generator_p_nom_effective, by={name}_members[{up}]) == at(Generator_p_nom_effective, by={name}_members[down])"
        expression = "Generator_typo == 0" if wrong else tie
        constraints = {
            f"{name}_tie": {"dims": ["scenario", name], "expression": expression}
        }
        if on:
            constraints[f"{name}_floor"] = {
                "dims": ["scenario", on],
                "expression": f"at(Generator_p_nom_effective, by={on}_members[up]) >= 0",
            }
        return {
            "name": name,
            "description": "",
            "parameters": {},
            "components": {"Generator": {up: {}, "down": {}}},
            "math": {"constraints": constraints},
        }

    def register(self, name, overwrite=False, **how):
        return self.registry.register(self.content(name, **how), overwrite=overwrite)

    def read(self, name):
        return self.registry[name].definition

    def stale(self, name):
        constraints = self.n._registries["composites"][name]["math"]["constraints"]
        constraints[f"{name}_tie"]["expression"] = "Generator_gone == 0"


class Stored:
    noun = "override"

    def __init__(self, n):
        self.n = n
        self.registry = n.optimize.overrides

    @staticmethod
    def content(name, on=None, changed=False, wrong=False):
        if wrong:
            return {"constraints": {"Generator_typo": None}}
        key = f"Generator_{name}_changed" if changed else f"Generator_{name}"
        patch = {"parameters": {key: {"dims": ["Generator"]}}}
        if on:
            patch["constraints"] = {
                f"Generator_{name}_floor": {
                    "dims": ["Generator"],
                    "where": "Generator_p_nom_extendable",
                    "expression": f"Generator_p_nom_ext >= Generator_{on}",
                }
            }
        return patch

    def register(self, name, overwrite=False, **how):
        return self.registry.register(
            self.content(name, **how), name=name, overwrite=overwrite
        )

    def read(self, name):
        return self.registry[name]

    def stale(self, name):
        stored = self.n._registries["overrides"][name]
        stored["edits"] = {"constraints.Generator_gone": "0"}


@pytest.fixture
def n():
    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Bus", ["north", "south"])
    n.add("Generator", "gas", bus="north", p_nom=10, marginal_cost=1)
    n.add("Load", "load", bus="north", p_set=1)
    return n


@pytest.fixture(params=[Types, Stored], ids=["composites", "overrides"])
def kind(request, n):
    return request.param(n)


def _state(kind):
    return [(name, kind.read(name)) for name in kind.registry]


def test_a_registry_reads_as_a_mapping(kind):
    assert kind.register("a") is None
    kind.register("b")
    registry = kind.registry
    assert isinstance(registry, Mapping)
    assert list(registry) == list(registry.keys()) == ["a", "b"]
    assert len(registry) == len(registry.values()) == 2
    assert "a" in registry
    assert "c" not in registry
    assert registry.get("c") is None
    assert dict(_state(kind)) == {"a": kind.content("a"), "b": kind.content("b")}
    assert repr(registry) == f"{type(registry).__name__} ['a', 'b']"
    with pytest.raises(KeyError, match=f"No {kind.noun} 'c' is registered"):
        registry["c"]


def test_an_entry_is_read_by_key_only(kind):
    kind.register("a")
    with pytest.raises(AttributeError):
        kind.registry.a
    kind.register("clear")
    assert list(kind.registry) == ["a", "clear"]
    kind.registry.clear()
    assert list(kind.registry) == []


def test_a_read_entry_is_a_copy(kind):
    kind.register("a")
    kind.read("a").clear()
    assert kind.read("a") == kind.content("a")


def test_a_name_registers_once(kind):
    kind.register("a")
    match = f"The {kind.noun} 'a' is registered already. Pass `overwrite=True`"
    for how in ({}, {"changed": True}):
        with pytest.raises(ValueError, match=match):
            kind.register("a", **how)
    assert _state(kind) == [("a", kind.content("a"))]


def test_overwrite_replaces_an_entry_in_its_place(kind):
    kind.register("a")
    kind.register("b")
    kind.register("a", changed=True, overwrite=True)
    kind.register("c", overwrite=True)
    assert list(kind.registry) == ["a", "b", "c"]
    assert kind.read("a") == kind.content("a", changed=True)


def test_unregister_and_clear_drop_entries(kind):
    for name in "abc":
        kind.register(name)
    with pytest.raises(KeyError, match=f"No {kind.noun} 'd' is registered"):
        kind.registry.unregister("d")
    assert kind.registry.unregister("b") is None
    assert list(kind.registry) == ["a", "c"]
    assert kind.registry.clear() is None
    assert list(kind.registry) == []
    kind.registry.clear()
    assert kind.n._registries == {}


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        (lambda k: k.register("c", wrong=True), LanguageError, "Generator_typo"),
        (
            lambda k: k.register("a", wrong=True, overwrite=True),
            LanguageError,
            "Generator_typo",
        ),
        (
            lambda k: k.register("a", changed=True, overwrite=True),
            LanguageError,
            "b_floor",
        ),
        (lambda k: k.registry.unregister("a"), ValueError, r"\['b'\] build on 'a'"),
    ],
    ids=[
        "register",
        "overwrite",
        "overwrite under another",
        "unregister under another",
    ],
)
def test_a_refused_change_leaves_the_registry(kind, change, error, match):
    kind.register("a")
    kind.register("b", on="a")
    registries = json.dumps(kind.n._registries)
    with pytest.raises(error, match=match):
        change(kind)
    assert json.dumps(kind.n._registries) == registries
    kind.n.optimize.create_model()


def test_an_entry_others_build_on_is_kept_beside_a_stale_one(kind):
    kind.register("a")
    kind.register("b", on="a")
    kind.register("c")
    kind.stale("c")
    with pytest.raises(ValueError, match=r"\['b'\] build on 'a'; unregister them"):
        kind.registry.unregister("a")
    for name in "cba":
        kind.registry.unregister(name)
    assert list(kind.registry) == []


@pytest.mark.parametrize(
    "drop",
    [
        lambda registry: registry.unregister("a"),
        lambda registry: registry.unregister("b"),
        lambda registry: registry.clear(),
    ],
    ids=["unregister it", "unregister another", "clear"],
)
def test_unregister_and_clear_drop_beside_a_stale_entry(kind, drop):
    kind.register("a")
    kind.register("b")
    kind.stale("a")
    with pytest.raises(ValueError, match="Generator_gone"):
        kind.n.optimize.create_model()
    with pytest.raises(ValueError, match="Generator_gone"):
        kind.register("c")
    drop(kind.registry)
    kind.registry.clear()
    kind.n.optimize.create_model()


def test_overwrite_repairs_a_stale_entry_in_its_place(kind):
    kind.register("a")
    kind.register("b")
    kind.stale("a")
    kind.register("a", overwrite=True)
    assert list(kind.registry) == ["a", "b"]
    kind.n.optimize.create_model()


FORMS = [
    "copy",
    "snapshots",
    "slice",
    "cluster",
    "merge",
    "netcdf",
    "csv_folder",
    pytest.param(
        "hdf5",
        marks=pytest.mark.skipif(
            find_spec("tables") is None, reason="PyTables not installed"
        ),
    ),
    pytest.param(
        "excel",
        marks=pytest.mark.skipif(
            find_spec("openpyxl") is None, reason="openpyxl not installed"
        ),
    ),
]


def _built(n, form, tmp_path):
    if form == "copy":
        return n.copy()
    if form == "snapshots":
        return n.copy(snapshots=n.snapshots[:1])
    if form == "slice":
        return n.slice_network(["north"])
    if form == "cluster":
        busmap = n.c.buses.static.index.to_series()
        return n.cluster.spatial.cluster_by_busmap(busmap)
    if form == "merge":
        other = pypsa.Network()
        other.set_snapshots(n.snapshots)
        return other.merge(n)
    path = tmp_path / {"netcdf": "n.nc", "hdf5": "n.h5", "excel": "n.xlsx"}.get(
        form, "n"
    )
    getattr(n, f"export_to_{form}")(path)
    return pypsa.Network(path)


@pytest.mark.parametrize("form", FORMS)
def test_a_built_or_loaded_network_keeps_its_own_registries(kind, form, tmp_path):
    kind.register("a")
    kind.register("b", on="a")
    other = type(kind)(_built(kind.n, form, tmp_path))
    assert other.registry == kind.registry
    assert _state(other) == _state(kind)
    other.registry.clear()
    assert list(kind.registry) == ["a", "b"]
    assert other.registry != kind.registry


def test_merge_unites_the_entries_of_both_networks(kind):
    other = type(kind)(pypsa.Network())
    other.n.set_snapshots(kind.n.snapshots)
    kind.register("a")
    kind.register("b")
    other.register("c")
    other.register("a")
    assert list(type(kind)(kind.n.merge(other.n)).registry) == ["a", "b", "c"]
    other.register("a", changed=True, overwrite=True)
    match = rf"different {kind.noun} under the names \['a'\]"
    with pytest.raises(ValueError, match=match):
        kind.n.merge(other.n, inplace=True)
    assert list(kind.registry) == ["a", "b"]


def test_equals_follows_the_order_of_the_entries(kind):
    other = type(kind)(kind.n.copy())
    for name in "ab":
        kind.register(name)
    for name in "ba":
        other.register(name)
    assert dict(_state(other)) == dict(_state(kind))
    assert other.registry != kind.registry
    assert not kind.n.equals(other.n)
    assert kind.n.equals(kind.n.copy())


def _both(n):
    Types(n).register("a")
    Stored(n).register("a")
    return n


def _registered(n):
    return list(n.composites), list(n.optimize.overrides)


@pytest.mark.parametrize(
    "change",
    [lambda meta: meta.clear(), lambda meta: meta.update(_pypsa={"overrides": {}})],
    ids=["clear", "update"],
)
def test_changing_meta_keeps_the_registries(n, change):
    _both(n).meta["wildcards"] = {"clusters": 5}
    change(n.meta)
    assert _registered(n) == (["a"], ["a"])
    n.meta = {}
    assert _registered(n) == (["a"], ["a"])
    assert n.meta == {}


def test_meta_keys_of_the_user_stay_theirs(n):
    n.meta |= {"overrides": {"solver": {"threads": 4}}, "composites": "mine"}
    _both(n).optimize.create_model()
    assert n.equals(n.copy())
    assert n.meta == {"overrides": {"solver": {"threads": 4}}, "composites": "mine"}


def test_loading_a_plain_file_drops_the_registries(n, tmp_path):
    path = tmp_path / "plain.nc"
    pypsa.Network().export_to_netcdf(path)
    _both(n).meta["wildcards"] = {"clusters": 5}
    n.import_from_netcdf(path)
    assert _registered(n) == ([], [])
    assert n.meta == {}


def _raw_meta(path):
    if path.is_dir():
        return json.loads((path / "meta.json").read_text())
    with xr.open_dataset(path) as ds:
        return json.loads(ds.attrs["meta"])


@pytest.mark.parametrize("form", ["netcdf", "csv_folder"])
def test_a_saved_network_holds_the_registries_under_the_meta_key(n, tmp_path, form):
    _both(n).meta["wildcards"] = {"clusters": 5}
    path = tmp_path / ("net.nc" if form == "netcdf" else "net")
    getattr(n, f"export_to_{form}")(path)
    assert pypsa.Network(path).meta == n.meta
    raw = _raw_meta(path)
    assert raw.pop("_pypsa").keys() == {"composites", "overrides"}
    assert raw == n.meta


@pytest.mark.parametrize(
    "form",
    [
        "netcdf",
        "csv_folder",
        pytest.param(
            "excel",
            marks=pytest.mark.skipif(
                find_spec("openpyxl") is None, reason="openpyxl not installed"
            ),
        ),
    ],
)
def test_export_refuses_the_reserved_meta_key(n, tmp_path, form):
    n.meta["_pypsa"] = {}
    with pytest.raises(ValueError, match="meta key '_pypsa' is reserved"):
        getattr(n, f"export_to_{form}")(tmp_path / f"net.{form}")


@pytest.mark.parametrize(
    "root",
    ["garbage", {"composites": "garbage"}, {"overrides": {"old": "garbage"}}],
)
def test_malformed_registries_of_a_saved_network_fail(n, root):
    n._registries = root
    with pytest.raises(ValueError, match="maps each name to a mapping, not"):
        n.optimize.create_model()
