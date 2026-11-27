# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import importlib
import warnings

import pytest


def test_statistics_deprecated_kwargs(ac_dc_solved):
    """Test that old parameter names raise deprecation warnings for all statistics methods."""
    from pypsa.statistics.expressions import StatisticsAccessor

    n = ac_dc_solved

    # Test each method for deprecated 'comps' parameter
    for method_name in StatisticsAccessor._methods:
        if method_name in ["prices"]:
            continue
        with pytest.raises(DeprecationWarning) as excinfo:
            getattr(n.statistics, method_name)(comps="Generator")
        assert "`comps` is deprecated" in str(excinfo.value)
        assert "use `components` instead" in str(excinfo.value)

    # Test a few methods for deprecated 'aggregate_groups' parameter
    for method_name in ["supply", "energy_balance", "capex"]:
        with pytest.raises(DeprecationWarning) as excinfo:
            getattr(n.statistics, method_name)(aggregate_groups="sum")
        assert "`aggregate_groups` is deprecated" in str(excinfo.value)
        assert "use `groupby_method` instead" in str(excinfo.value)

    # Test a few methods for deprecated 'aggregate_time' parameter
    for method_name in ["supply", "withdrawal", "transmission"]:
        with pytest.raises(DeprecationWarning) as excinfo:
            getattr(n.statistics, method_name)(aggregate_time="mean")
        assert "`aggregate_time` is deprecated" in str(excinfo.value)
        assert "use `groupby_time` instead" in str(excinfo.value)

    # Test the __call__ method also has deprecated parameters
    with pytest.raises(DeprecationWarning) as excinfo:
        n.statistics(comps="Generator")
    assert "`comps` is deprecated" in str(excinfo.value)

    with pytest.raises(DeprecationWarning) as excinfo:
        n.statistics(aggregate_groups="sum")
    assert "`aggregate_groups` is deprecated" in str(excinfo.value)

    with pytest.raises(DeprecationWarning) as excinfo:
        n.statistics(aggregate_time="mean")
    assert "`aggregate_time` is deprecated" in str(excinfo.value)

    # Test that both old and new parameters raise error
    with pytest.raises(DeprecationWarning) as excinfo:
        n.statistics.installed_capacity(comps="Generator", components="Generator")
    assert "received both comps and components" in str(excinfo.value)

    # Test equivalence: old and new params produce same results
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_result = n.statistics.installed_capacity(comps="Generator")
    new_result = n.statistics.installed_capacity(components="Generator")
    assert old_result.equals(new_result)


def test_deprecation_details_are_plain_text():
    """Deprecation messages reach users as warnings, so they carry no doc markup."""
    import ast
    from pathlib import Path

    root = Path(__file__).parent.parent / "pypsa"
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if func != "deprecated":
                continue
            for kw in node.keywords:
                if kw.arg == "details" and isinstance(kw.value, ast.Constant):
                    details = kw.value.value
                    where = f"{path.name}:{node.lineno}"
                    assert "<!--" not in details, where
                    assert "!!!" not in details, where
                    assert details.count("`") % 2 == 0, where


def test_deprecated_sub_network_accessors_suggest_working_code(ac_dc_network):
    """Every suggested replacement resolves on the sub-network it points at."""
    import re
    from operator import attrgetter

    n = ac_dc_network.copy()
    n.determine_network_topology()
    sub_network = n.c.sub_networks.static.obj.iloc[0]

    for name in [
        "buses_i",
        "lines_i",
        "generators_i",
        "loads_i",
        "stores_i",
        "storage_units_i",
        "buses",
        "generators",
        "loads",
        "stores",
        "storage_units",
    ]:
        with pytest.warns(DeprecationWarning, match="deprecated") as record:
            getattr(sub_network, name)()
        suggestion = re.search(r"`([^`]+)`", str(record[0].message)).group(1)
        assert suggestion.startswith("sub_network."), (name, suggestion)
        attrgetter(suggestion.removeprefix("sub_network."))(sub_network)


def test_aggregate_across_components_warning_is_readable(ac_dc_network):
    with pytest.warns(
        DeprecationWarning, match="aggregate_across_components"
    ) as record:
        ac_dc_network.statistics.installed_capacity(aggregate_across_components=True)
    message = str(record[0].message)
    assert "<!--" not in message
    assert message.count("`") % 2 == 0


def test_aggregate_multi_graph_deprecated(ac_dc_network):
    from pypsa.network.power_flow import aggregate_multi_graph

    n = ac_dc_network.copy()
    n.determine_network_topology()
    sub_network = n.c.sub_networks.static.obj.iloc[0]
    with pytest.warns(DeprecationWarning, match="removed without replacement"):
        aggregate_multi_graph(sub_network)


def test_network_iplot_deprecated(ac_dc_network):
    with pytest.warns(DeprecationWarning, match="explore"):
        ac_dc_network.plot.iplot(iplot=False)


@pytest.mark.parametrize(
    "access", ["from pypsa.optimization.{} import {}", "pypsa.optimization.{}.{}"]
)
@pytest.mark.parametrize(
    ("module", "name", "hint"),
    [
        ("constraints", "define_nodal_balance_constraints", "Bus_nodal_balance"),
        ("constraints", "DEG2RAD", "no replacement"),
        ("variables", "define_nominal_variables", "Generator_p_nom_ext"),
        ("global_constraints", "define_growth_limit", "Carrier_growth_limit"),
        ("piecewise", "define_piecewise", "piecewise_options"),
        ("optimize", "define_objective", "total_cost"),
    ],
)
def test_removed_optimization_names(module, name, hint, access):
    namespace = {"pypsa": __import__(f"pypsa.optimization.{module}")}
    with pytest.raises(ImportError, match=rf"(?s)removed in PyPSA 1\.5\.0.*{hint}"):
        exec(access.format(module, name), namespace)


@pytest.mark.parametrize("module", ["constraints", "variables", "piecewise"])
def test_unknown_optimization_name_raises_attribute_error(module):
    module = importlib.import_module(f"pypsa.optimization.{module}")
    assert not hasattr(module, "no_such_name")
