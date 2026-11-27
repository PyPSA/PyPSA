# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import copy
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
import yaml
from mathspec.errors import LanguageError

import pypsa

BATTERY = yaml.safe_load("""
name: battery
description: a battery whose charger and discharger share one inverter
parameters:
  bus: null
  efficiency: 0.95
  max_hours: 8
  availability: 1.0
  capital_cost_power: 5
  capital_cost_energy: 1
  shared_inverter: true
components:
  Bus:
    dc: {carrier: battery}
  Link:
    charger: {bus0: $bus, bus1: "@dc", efficiency: $efficiency, p_nom_extendable: true, capital_cost: $capital_cost_power}
    discharger: {bus0: "@dc", bus1: $bus, efficiency: $efficiency, p_nom_extendable: true, p_max_pu: $availability}
  Store:
    store: {bus: "@dc", e_nom_extendable: true, e_cyclic: true, capital_cost: $capital_cost_energy}
math:
  expressions:
    battery_power:
      dims: [scenario, snapshot, battery]
      expression: >-
        at(Link_p, by=battery_members[discharger])
        - at(Link_p, by=battery_members[charger])
  constraints:
    battery_inverter:
      description: where `battery_efficiency` is high, the charger and the discharger share one rating
      dims: [scenario, battery]
      where: battery_shared_inverter
      expression: >-
        at(Link_p_nom_effective, by=battery_members[charger])
        == at(Link_p_nom_effective, by=battery_members[discharger])
    battery_duration:
      dims: [scenario, battery]
      expression: >-
        at(Store_e_nom_effective, by=battery_members[store])
        <= battery_max_hours * at(Link_p_nom_effective, by=battery_members[discharger])
""")
CELL = yaml.safe_load(yaml.safe_dump(BATTERY).replace("battery", "cell"))
INSTANCES = pd.Index(["bat north", "bat south"])
FLOOR = {
    "dims": ["scenario", "battery"],
    "expression": "at(Store_e_nom_effective, by=battery_members[store]) >= battery_max_hours",
}
CELL_ON_BATTERY = copy.deepcopy(CELL)
CELL_ON_BATTERY["math"]["constraints"]["cell_floor"] = FLOOR
DRIFTED = {
    "patch": {"constraints": {"Generator_p_ramp_limit_up": None}},
    "edits": {"constraints.Generator_p_ramp_limit_up": "0"},
    "creates": [],
}


@pytest.fixture
def n():
    n = pypsa.Network()
    n.set_snapshots(range(24))
    n.add("Bus", ["north", "south"])
    sun = np.clip(np.sin(np.linspace(0, 2 * np.pi, 24)), 0, None)
    n.add(
        "Generator",
        ["north pv", "south pv"],
        bus=["north", "south"],
        p_nom_extendable=True,
        p_max_pu=np.column_stack([sun, sun]),
        capital_cost=50,
    )
    n.add(
        "Generator",
        ["north gas", "south gas"],
        bus=["north", "south"],
        p_nom=100,
        marginal_cost=80,
    )
    n.add("Load", ["north load", "south load"], bus=["north", "south"], p_set=10)
    n.composites.register(BATTERY)
    return n


def _add(n):
    n.add("battery", INSTANCES, bus=["north", "south"], max_hours=[2, 6])


@pytest.mark.parametrize("form", ["mapping", "text", "path"])
def test_a_type_registers_from_each_form(form, tmp_path):
    n = pypsa.Network()
    definition = {"mapping": BATTERY, "text": yaml.safe_dump(BATTERY)}.get(form)
    if form == "path":
        definition = tmp_path / "battery.yaml"
        definition.write_text(yaml.safe_dump(BATTERY))
    battery = n.composites.register(definition)
    assert n.composites.battery.name == battery.name == "battery"
    assert list(n.composites) == ["battery"]
    assert (
        n.meta["_pypsa"]["composites"]["battery"]["components"] == BATTERY["components"]
    )


def test_add_expands_an_instance_into_tagged_members(n):
    names = n.add("battery", INSTANCES, bus=["north", "south"], return_names=True)
    assert names.equals(INSTANCES)
    links = n.c.links.static
    assert list(links.index) == [
        "bat north charger",
        "bat south charger",
        "bat north discharger",
        "bat south discharger",
    ]
    assert list(links.bus0) == ["north", "south", "bat north dc", "bat south dc"]
    assert list(links.composite) == list(INSTANCES) * 2
    assert list(links.composite_member) == ["charger"] * 2 + ["discharger"] * 2
    assert (n.c.stores.static.composite_type == "battery").all()
    assert n.composites.battery.members.loc["bat south", "store"] == "bat south store"


def test_add_lays_values_over_the_instances(n):
    availability = pd.DataFrame({"bat south": 0.5, "bat north": 1.0}, index=n.snapshots)
    n.composites.battery.add(
        INSTANCES,
        bus=pd.Series({"bat south": "south", "bat north": "north"}),
        efficiency=[0.9, 0.8],
        availability=availability,
    )
    static = n.composites.battery.static
    assert list(static.bus) == ["north", "south"]
    assert list(static.efficiency) == [0.9, 0.8]
    assert list(static.max_hours) == [8, 8]
    n.add("battery", "solo", bus="north", availability=pd.Series(0.25, n.snapshots))
    p_max_pu = n.c.links.dynamic.p_max_pu
    assert (p_max_pu["bat south discharger"] == 0.5).all()
    assert (p_max_pu["solo discharger"] == 0.25).all()


@pytest.mark.parametrize(
    ("names", "kwargs", "match"),
    [
        ("x", {"bus": "north", "volume": 3}, r"unknown \['volume'\]"),
        ("x", {}, r"missing \['bus'\]"),
        ("x", {"bus": "north", "max_hours": pd.DataFrame({"x": [1.0]})}, "time series"),
        (["x", "z"], {"bus": pd.Series({"x": "north", "y": "south"})}, "lacks"),
        (["x", "z"], {"bus": pd.Series(["north", "south"])}, "lacks"),
        (["x", "z"], {"bus": "north", "efficiency": [0.9, 0.8, 0.7]}, "length 3"),
        (
            ["x", "z"],
            {"bus": "north", "availability": pd.DataFrame({"x": [1.0], "z": [1.0]})},
            "align",
        ),
        ("bat north", {"bus": "north"}, "are taken"),
        ("", {"bus": "north"}, "empty"),
        ("y", {"bus": "north"}, "exist"),
    ],
)
def test_a_refused_add_leaves_the_network_unchanged(n, names, kwargs, match):
    _add(n)
    n.add("Link", "y discharger", bus0="north", bus1="south")
    sizes = {c.name: len(c.static) for c in n.components}
    with pytest.raises(ValueError, match=match):
        n.add("battery", names, **kwargs)
    assert {c.name: len(c.static) for c in n.components} == sizes
    n.add("battery", "w", bus="north")


def test_a_refused_first_add_adds_no_columns(n):
    columns = {c.name: list(c.static.columns) for c in n.components}
    with pytest.raises(ValueError, match="length 3"):
        n.add("battery", ["x", "z"], bus="north", efficiency=[0.9, 0.8, 0.7])
    assert {c.name: list(c.static.columns) for c in n.components} == columns


def test_add_takes_a_suffix_and_refuses_overwrite(n):
    names = n.add("battery", "bat", suffix=[" a", " b"], bus="north", return_names=True)
    assert list(names) == list(n.composites.battery.members.index) == ["bat a", "bat b"]
    with pytest.raises(ValueError, match="overwrite"):
        n.add("battery", "bat a", overwrite=True, bus="north")


@pytest.mark.parametrize("index", [["a", "b"], ["a x", "b x"]])
def test_a_value_by_instance_takes_the_suffix(n, index):
    bus = pd.Series(["north", "south"], index)
    availability = pd.DataFrame(dict(zip(index, [0.5, 1.0])), index=n.snapshots)
    n.add("battery", ["a", "b"], suffix=" x", bus=bus, availability=availability)
    assert list(n.composites.battery.static.bus) == ["north", "south"]
    assert (n.c.links.dynamic.p_max_pu["a x discharger"] == 0.5).all()


def test_remove_drops_every_member_of_an_instance(n):
    _add(n)
    n.remove("battery", "bat north")
    assert list(n.composites.battery.members.index) == ["bat south"]
    assert not n.c.links.static.index.str.startswith("bat north").any()
    with pytest.raises(ValueError, match="no battery instances"):
        n.remove("battery", "bat north")


def test_instances_remove_where_their_type_no_longer_reads(n):
    _add(n)
    n.meta["_pypsa"]["composites"]["battery"]["parameters"]["legacy"] = 1
    n.remove("battery", INSTANCES)
    assert n.composites.instances.empty


def test_the_math_ties_the_members(n):
    _add(n)
    links = n.c.links.static
    links.loc["bat south charger", ["p_nom", "p_nom_extendable"]] = [7.0, False]
    n.optimize()
    p_nom = links.p_nom_opt
    e_nom = n.c.stores.static.e_nom_opt
    assert p_nom["bat north charger"] == pytest.approx(p_nom["bat north discharger"])
    assert p_nom["bat south discharger"] == pytest.approx(7)
    assert e_nom["bat south store"] <= 6 * 7 + 1e-6
    assert e_nom["bat north store"] == pytest.approx(2 * p_nom["bat north discharger"])


@pytest.mark.parametrize("anchor", ["Bus", "Link"])
def test_a_bool_parameter_masks_rows(anchor):
    n = pypsa.Network()
    n.add("Bus", ["north", "south"])
    components = BATTERY["components"]
    n.composites.register(_with(components={anchor: components[anchor], **components}))
    n.add("battery", INSTANCES, bus=["north", "south"], shared_inverter=[True, False])
    m = n.optimize.create_model()
    rows = (m.constraints["battery_inverter"].labels != -1).any("scenario")
    assert rows.to_series().to_dict() == {"bat north": True, "bat south": False}


def test_a_math_parameter_varies_by_scenario(n):
    _add(n)
    n.set_scenarios({"low": 0.5, "high": 0.5})
    n.c.buses.static.loc[("high", "bat north dc"), "battery_max_hours"] = 0.5
    n.optimize()
    p_nom = n.c.links.static.p_nom_opt.xs("low", level="scenario")
    e_nom = n.c.stores.static.e_nom_opt.xs("low", level="scenario")
    assert e_nom["bat north store"] <= 0.5 * p_nom["bat north discharger"] + 1e-6


def test_instances_run_over_investment_periods(n):
    _add(n)
    n.snapshots = pd.MultiIndex.from_product([[2030, 2040], range(12)])
    n.investment_periods = [2030, 2040]
    n.c.links.static.loc["bat north discharger", "build_year"] = 2040
    n.optimize(multi_investment_periods=True)
    assert n.model.constraints["battery_duration"].sizes["battery"] == 2


def _replaced(*pairs):
    text = yaml.safe_dump(BATTERY)
    for old, new in pairs:
        text = text.replace(old, new)
    return yaml.safe_load(text)


def test_two_types_build_side_by_side(n):
    _add(n)
    n.composites.register(CELL)
    n.add("cell", "c", bus="north")
    m = n.optimize.create_model()
    assert {"battery_inverter", "cell_inverter"} <= set(m.constraints)


def test_the_math_reads_its_own_entries_by_name():
    definition = copy.deepcopy(BATTERY)
    definition["parameters"]["power"] = 0.0
    definition["components"]["Store"]["store"]["e_initial"] = "$power"
    definition["math"]["constraints"]["battery_cap"] = {
        "dims": ["scenario", "snapshot", "battery"],
        "expression": "battery_power <= 100",
    }
    n = pypsa.Network()
    n.add("Bus", "north")
    n.composites.register(definition)
    n.add("battery", "bat", bus="north", power=3)
    assert n.c.stores.static.e_initial["bat store"] == 3


def test_a_description_is_no_parameter_reference(n):
    _add(n)
    assert "battery_shared_inverter" in n.c.buses.static
    assert "battery_efficiency" not in n.c.buses.static


def test_statistics_group_by_instance(n):
    _add(n)
    n.optimize()
    balance = n.statistics.energy_balance(groupby="composite")
    assert set(balance.index.get_level_values("composite")) == {"", *INSTANCES}


def test_an_expression_reads_each_instance(n):
    _add(n)
    n.optimize(expressions=True)
    power = n.model.expressions["battery_power"].solution
    power = power.squeeze("scenario", drop=True).to_pandas().T
    p = n.c.links.dynamic.p
    discharge = p[INSTANCES + " discharger"].set_axis(INSTANCES, axis=1)
    charge = p[INSTANCES + " charger"].set_axis(INSTANCES, axis=1)
    pd.testing.assert_frame_equal(
        power, discharge - charge, check_names=False, check_column_type=False
    )


def test_a_type_without_instances_adds_no_rows(n):
    n.optimize()
    assert "battery_inverter" not in n.model.constraints


@pytest.mark.parametrize(
    "form",
    [
        "copy",
        "slice",
        "netcdf",
        pytest.param(
            "hdf5",
            marks=pytest.mark.skipif(
                find_spec("tables") is None, reason="PyTables not installed"
            ),
        ),
    ],
)
def test_composites_survive_copy_and_io(n, tmp_path, form):
    _add(n)
    n.add("Bus", "west")
    path = tmp_path / {"netcdf": "net.nc", "hdf5": "net.h5"}.get(form, "")
    if form in ("netcdf", "hdf5"):
        getattr(n, f"export_to_{form}")(path)
    other = {"copy": n.copy, "slice": lambda: n.slice_network(n.c.buses.names)}.get(
        form, lambda: pypsa.Network(path)
    )()
    n.optimize()
    assert other.composites.battery.members.equals(n.composites.battery.members)
    pd.testing.assert_frame_equal(
        other.composites.battery.static, n.composites.battery.static, check_dtype=False
    )
    other.optimize()
    assert other.objective == pytest.approx(n.objective)


def test_merge_unites_composite_types(n):
    _add(n)
    other = pypsa.Network()
    other.set_snapshots(n.snapshots)
    other.add("Bus", "east")
    merged = other.merge(n)
    assert list(merged.composites) == ["battery"]
    assert "battery_inverter" in merged.optimize.create_model().constraints
    other.composites.register(_with(description="other"))
    with pytest.raises(ValueError, match=r"different composite type .*\['battery'\]"):
        other.merge(n)


def test_spatial_clustering_keeps_composite_types(n):
    _add(n)
    n.optimize.overrides.register(
        {"constraints": {"battery_duration": None}}, name="free"
    )
    busmap = n.c.buses.static.index.to_series().replace("south", "north")
    clustered = n.cluster.spatial.cluster_by_busmap(busmap)
    assert clustered.meta["_pypsa"] == n.meta["_pypsa"]
    clustered.optimize()
    assert clustered.composites.battery.static.index.equals(INSTANCES)


def test_equals_follows_the_order_of_types(n):
    n.composites.register(CELL)
    other = n.copy()
    assert n.equals(other)
    types = other.meta["_pypsa"]["composites"]
    other.meta["_pypsa"]["composites"] = dict(reversed(types.items()))
    assert not n.equals(other)


def test_a_stored_override_edits_a_type(n):
    _add(n)
    n.optimize.overrides.register(
        {"constraints": {"battery_duration": None}}, name="free"
    )
    assert "battery_duration" not in n.optimize.create_model().constraints


def test_a_type_refuses_an_entry_a_stored_override_creates():
    n = pypsa.Network()
    inverter = {
        "dims": ["Generator"],
        "where": "Generator_p_nom_extendable",
        "expression": "Generator_p_nom_ext >= 0",
    }
    n.optimize.overrides.register(
        {"constraints": {"battery_inverter": inverter}}, name="own"
    )
    with pytest.raises(
        ValueError, match=r"'own' creates \['constraints.battery_inverter'\]"
    ):
        n.composites.register(BATTERY)
    assert list(n.composites) == []


def test_a_type_is_checked_with_the_default_formulation(n):
    definition = copy.deepcopy(CELL)
    definition["math"]["constraints"]["cell_duration"]["where"] = "Generator_tightened"
    with pytest.raises(LanguageError, match="Generator_tightened"):
        n.composites.register(definition)
    assert list(n.composites) == ["battery"]


def test_plain_buses_cluster_beside_instances(n):
    _add(n)
    n.add("Bus", "west")
    buses = n.c.buses.static.index
    busmap = pd.Series(buses, buses).where(buses.str.startswith("bat"), "west")
    clustered = n.cluster.spatial.cluster_by_busmap(busmap)
    max_hours = clustered.c.buses.static.battery_max_hours
    assert max_hours.dropna().to_dict() == {"bat north dc": 2, "bat south dc": 6}


def test_an_incomplete_instance_fails(n):
    _add(n)
    n.remove("Store", "bat north store")
    with pytest.raises(ValueError, match=r"\['bat north'\] lack members"):
        n.optimize.create_model()
    n.remove("battery", "bat north")
    n.optimize.create_model()


def test_tags_without_their_type_fail_until_registered_again(n):
    _add(n)
    n.composites.register(CELL)
    n.add("cell", "c", bus="north")
    del n.meta["_pypsa"]
    with pytest.raises(
        ValueError, match=r"types \['battery', 'cell'\], which are not registered"
    ):
        n.optimize.create_model()
    n.composites.register(CELL)
    n.composites.register(BATTERY)
    n.optimize.create_model()


def test_an_instance_is_built_in_full_or_not_at_all(n):
    _add(n)
    links = n.c.links.static
    links.loc[["bat north charger", "bat north discharger"], "active"] = False
    n.c.stores.static.loc["bat north store", "active"] = False
    m = n.optimize.create_model()
    assert list(m.constraints["battery_inverter"].indexes["battery"]) == ["bat south"]
    n.c.stores.static.loc["bat north store", "active"] = True
    with pytest.raises(ValueError, match="outside the build"):
        n.optimize.create_model()


def _with(**changes):
    return {**BATTERY, **changes}


@pytest.mark.parametrize(
    ("definition", "match"),
    [
        (_with(name="Link"), "named by an identifier"),
        (_with(name="bus"), "named by an identifier"),
        (_with(name="register"), "named by an identifier"),
        (_with(name="n"), "named by an identifier"),
        (_with(volume=1), "sections"),
        (_with(components={"Battery": {"x": {}}}), "by component class"),
        (_with(components={"Carrier": {"x": {}}}), "by component class"),
        (_with(parameters={**BATTERY["parameters"], "name": 1}), "cannot name"),
        (
            _with(parameters={**BATTERY["parameters"], "note": 1}),
            r"uses the parameters \['note'\]",
        ),
        (
            _with(parameters={**BATTERY["parameters"], "max_hours": None}),
            "take a default",
        ),
        (_with(components={"Link": {"a": {"bus0": "@b"}}}), r"refers to \['@b'\]"),
        (_with(components={"Link": {"a": {"bus0": "$b"}}}), r"refers to \['\$b'\]"),
        (
            _with(
                components={
                    **BATTERY["components"],
                    "Store": {"store": {"bus": "@dc", "e_nom": "$max_hours"}},
                }
            ),
            r"sets \['max_hours'\]",
        ),
        (_replaced(("battery", "v"), ("max_hours", "nom")), r"columns \['v_nom'\]"),
        (
            _replaced(("battery", "composite"), ("max_hours", "type")),
            r"columns \['composite_type'\]",
        ),
        (_with(math={"objective": {}}), "has the sections"),
        (_with(math={"constraints": {"inverter": {}}}), "cannot add 'inverter'"),
        (
            _with(math={"constraints": {"Generator_ext_p_upper": None}}),
            "cannot add 'Generator_ext_p_upper'",
        ),
    ],
)
def test_a_malformed_definition_is_refused(definition, match):
    with pytest.raises(ValueError, match=match):
        pypsa.Network().composites.register(definition)


@pytest.mark.parametrize(
    ("typo", "match"),
    [
        (("Store_e_nom_effective", "Store_e_nom_eff"), "'Store_e_nom_eff' not found"),
        (("battery_members[store]", "battery_members[stor]"), "no column"),
    ],
)
def test_wrong_math_is_refused_at_register(typo, match):
    definition = _replaced(typo)
    with pytest.raises(LanguageError, match=match):
        pypsa.Network().composites.register(definition)


@pytest.mark.parametrize("first", [True, False])
def test_a_type_cannot_declare_the_names_of_another(first):
    other = _replaced(("battery", "battery_power"))
    n = pypsa.Network()
    n.composites.register(BATTERY if first else other)
    with pytest.raises(ValueError, match=r"declares \['battery_power'\]"):
        n.composites.register(other if first else BATTERY)


def test_editing_a_read_type_leaves_the_stored_type(n):
    n.composites.battery.type.templates["charger"]["efficiency"] = 0.1
    assert n.composites.battery.type.templates["charger"]["efficiency"] == "$efficiency"


def test_a_stale_type_blocks_registration_until_removed(n):
    n.meta["_pypsa"]["composites"]["battery"]["parameters"]["legacy"] = 1
    with pytest.raises(ValueError, match="declares"):
        n.composites.register(_replaced(("battery", "battery_power")))
    with pytest.raises(ValueError, match=r"parameters \['legacy'\]"):
        n.composites.register(_replaced(("battery", "cell")))
    n.composites.remove("battery")
    n.composites.register(_replaced(("battery", "cell")))


def test_another_type_of_the_same_name_is_refused(n):
    n.composites.register(BATTERY)
    with pytest.raises(ValueError, match="Another composite type"):
        n.composites.register(_with(description="other"))


def test_remove_unregisters_a_type_without_instances(n):
    with pytest.raises(KeyError, match="No composite type 'cell'"):
        n.composites.remove("cell")
    _add(n)
    meta = copy.deepcopy(n.meta)
    with pytest.raises(ValueError, match=r"\['bat north', 'bat south'\] exist"):
        n.composites.remove("battery")
    assert n.meta == meta
    n.remove("battery", INSTANCES)
    n.composites.remove("battery")
    assert list(n.composites) == []
    n.optimize.create_model()
    n.composites.register(_with(description="other"))


@pytest.mark.parametrize("drifted", [False, True])
@pytest.mark.parametrize(
    ("reader", "match"),
    [
        ({"constraints": {"battery_duration": None}}, r"\['own'\] build on 'battery'"),
        ({"constraints": {"floor": FLOOR}}, r"\['own'\] build on 'battery'"),
        (CELL_ON_BATTERY, r"\['cell'\] build on 'battery'"),
    ],
    ids=["override edits", "override reads", "type reads"],
)
def test_a_type_others_build_on_is_kept(n, reader, match, drifted):
    if "name" in reader:
        n.composites.register(reader)
    else:
        n.optimize.overrides.register(reader, name="own")
    if drifted:
        n.meta["_pypsa"].setdefault("overrides", {})["other"] = copy.deepcopy(DRIFTED)
    meta = copy.deepcopy(n.meta)
    with pytest.raises(ValueError, match=match):
        n.composites.remove("battery")
    assert n.meta == meta


@pytest.mark.parametrize("own", [False, True])
def test_remove_drops_a_type_that_no_longer_builds(n, own):
    if own:
        n.optimize.overrides.register({"constraints": {"floor": FLOOR}}, name="own")
    constraints = n.meta["_pypsa"]["composites"]["battery"]["math"]["constraints"]
    duration = constraints["battery_duration"]
    duration["expression"] = duration["expression"].replace("e_nom", "e_gone")
    with pytest.raises(LanguageError, match="Store_e_gone"):
        n.optimize.create_model()
    n.composites.remove("battery")
    assert list(n.composites) == []
    if own:
        n.optimize.overrides.remove("own")
    n.optimize.create_model()
