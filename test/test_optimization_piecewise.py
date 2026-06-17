# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Unit tests for helpers in ``pypsa.optimization.piecewise``."""

from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace

import linopy
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pypsa.constants import PIECEWISE_ATTRS
from pypsa.optimization.piecewise import (
    PiecewiseOptions,
    _create_y_var,
    _get_breakpoints,
    _normalize_breakpoints,
    _to_da,
    define_piecewise,
    get_piecewise_names,
)


@pytest.fixture
def gen_marginal_cost_attrs() -> pd.Series:
    return PIECEWISE_ATTRS.query(
        "component == 'Generator' and y == 'marginal_cost'"
    ).squeeze()


def _piecewise_df(
    curves: dict[str, list[tuple[float, float]]],
    x_attr: str = "p_pu",
    y_attr: str = "marginal_cost",
) -> pd.DataFrame:
    """Build a (name, attribute)-columned DataFrame from {name: [(x, y), ...]}."""
    frames = {
        n: pd.DataFrame(rows, columns=[x_attr, y_attr]) for n, rows in curves.items()
    }
    return pd.concat(frames, axis=1, names=["name", "attribute"]).rename_axis(
        index="breakpoint"
    )


class TestNormalizeBreakpoints:
    def test_sorts_unsorted_rows(self, gen_marginal_cost_attrs: pd.Series) -> None:
        df = _piecewise_df({"gen": [(1.0, 40.0), (0.0, 10.0), (0.5, 20.0)]})
        result = _normalize_breakpoints(df, gen_marginal_cost_attrs)
        assert result["gen"]["p_pu"].tolist() == [0.0, 0.5, 1.0]
        assert result["gen"]["marginal_cost"].tolist() == [10.0, 20.0, 40.0]
        assert result.index.name == "breakpoint"

    def test_ragged_curves_aligned_with_trailing_nan(
        self, gen_marginal_cost_attrs: pd.Series
    ) -> None:
        df = _piecewise_df(
            {
                "gen0": [(0.0, 10.0), (0.5, 20.0), (1.0, 40.0)],
                "gen1": [(0.0, 5.0), (1.0, 25.0)],
            }
        )
        result = _normalize_breakpoints(df, gen_marginal_cost_attrs)
        assert len(result) == 3
        assert result["gen0"]["p_pu"].tolist() == [0.0, 0.5, 1.0]
        assert result["gen1"]["p_pu"].iloc[:2].tolist() == [0.0, 1.0]
        assert np.isnan(result["gen1"]["p_pu"].iloc[2])

    def test_idempotent(self, gen_marginal_cost_attrs: pd.Series) -> None:
        df = _piecewise_df({"gen": [(1.0, 40.0), (0.0, 10.0), (0.5, 20.0)]})
        once = _normalize_breakpoints(df, gen_marginal_cost_attrs)
        twice = _normalize_breakpoints(once, gen_marginal_cost_attrs)
        pd.testing.assert_frame_equal(once, twice)

    @pytest.mark.parametrize(
        ("curves", "match"),
        [
            pytest.param(
                {"gen": [(0.0, 10.0), (float("nan"), float("nan")), (1.0, 40.0)]},
                "non-trailing missing breakpoint",
                id="interior-nan-row",
            ),
            pytest.param(
                {"gen": [(0.0, 10.0), (0.5, float("nan")), (1.0, 40.0)]},
                "incomplete breakpoint data",
                id="missing-y",
            ),
            pytest.param(
                {"gen": [(0.0, 10.0), (float("nan"), 20.0), (1.0, 40.0)]},
                "incomplete breakpoint data",
                id="missing-x",
            ),
        ],
    )
    def test_invalid_breakpoints_raise(
        self,
        gen_marginal_cost_attrs: pd.Series,
        curves: dict[str, list[tuple[float, float]]],
        match: str,
    ) -> None:
        df = _piecewise_df(curves)
        with pytest.raises(ValueError, match=match):
            _normalize_breakpoints(df, gen_marginal_cost_attrs)


class TestGetPiecewiseNames:
    def test_missing_attribute_returns_empty(self) -> None:
        c = SimpleNamespace(
            piecewise={},
            _piecewise_attrs=PIECEWISE_ATTRS.query("component == 'Generator'"),
        )
        result = get_piecewise_names(c, "marginal_cost", pd.Index(["gen"], name="name"))
        assert result.empty
        assert result.name == "name"

    def test_all_nan_columns_dropped(self) -> None:
        df = _piecewise_df(
            {
                "gen0": [(0.0, 10.0), (1.0, 40.0)],
                "gen1": [(float("nan"), float("nan")), (float("nan"), float("nan"))],
            }
        )
        c = SimpleNamespace(
            piecewise={"marginal_cost": df},
            _piecewise_attrs=PIECEWISE_ATTRS.query("component == 'Generator'"),
        )
        active = pd.Index(["gen0", "gen1"], name="name")
        result = get_piecewise_names(c, "marginal_cost", active)
        assert result.tolist() == ["gen0"]

    def test_intersects_with_active_names(self) -> None:
        df = _piecewise_df(
            {
                "gen0": [(0.0, 10.0), (1.0, 40.0)],
                "gen1": [(0.0, 5.0), (1.0, 25.0)],
                "gen2": [(0.0, 3.0), (1.0, 15.0)],
            }
        )
        c = SimpleNamespace(
            piecewise={"marginal_cost": df},
            _piecewise_attrs=PIECEWISE_ATTRS.query("component == 'Generator'"),
        )
        active = pd.Index(["gen0", "gen2"], name="name")
        result = get_piecewise_names(c, "marginal_cost", active)
        assert sorted(result.tolist()) == ["gen0", "gen2"]

    @pytest.mark.parametrize(
        ("piecewise", "active"),
        [
            pytest.param({}, pd.Index([], name="name"), id="empty"),
            pytest.param(
                {"marginal_cost": _piecewise_df({"gen": [(0.0, 10.0), (1.0, 40.0)]})},
                pd.Index(["gen"], name="name"),
                id="populated",
            ),
        ],
    )
    def test_result_is_named_name_index(
        self, piecewise: dict, active: pd.Index
    ) -> None:
        c = SimpleNamespace(
            piecewise=piecewise,
            _piecewise_attrs=PIECEWISE_ATTRS.query("component == 'Generator'"),
        )
        result = get_piecewise_names(c, "marginal_cost", active)
        assert result.name == "name"


class TestGetBreakpoints:
    @pytest.fixture
    def component(self) -> SimpleNamespace:
        return SimpleNamespace(
            piecewise={
                y: _piecewise_df(
                    {"gen": [(0.0, 10.0), (0.5, 20.0), (1.0, 40.0)]}, x_attr=x, y_attr=y
                )
                for x, y in [("p_pu", "marginal_cost"), ("p_nom", "capital_cost")]
            },
            static=pd.DataFrame(
                index=["gen"], data={"p_nom": [100.0], "p_nom_extendable": [False]}
            ),
            name="Generator",
            extendables=pd.Index([], name="name"),
            _piecewise_attrs=PIECEWISE_ATTRS.query("component == 'Generator'"),
        )

    @pytest.fixture
    def component_extendable(self, component):
        component.extendables = pd.Index(["gen"], name="name")
        return component

    @pytest.fixture
    def pw_names(self) -> pd.Series:
        return pd.Index(["gen"], name="name")

    @pytest.mark.parametrize("cumulative_attr", [True, False])
    @pytest.mark.parametrize("invert_attr", [True, False])
    def test_allow_extendables_x(
        self, component_extendable, pw_names, cumulative_attr, invert_attr
    ) -> None:
        """Test expected x breakpoints are returned for extendable components when x_attr is nominal."""
        x_breakpoints, _ = _get_breakpoints(
            component_extendable, "capital_cost", pw_names, cumulative_attr, invert_attr
        )
        expected_x = [0, 0.5, 1.0]
        assert np.allclose(x_breakpoints.to_series().values, expected_x)

    @pytest.mark.parametrize("cumulative_attr", [True, False])
    @pytest.mark.parametrize("invert_attr", [True, False])
    def test_not_allow_extendables_x(
        self, component, pw_names, cumulative_attr, invert_attr
    ) -> None:
        """Test expected x breakpoints are returned for non-extendable components when x_attr is per-unit."""
        x_breakpoints, _ = _get_breakpoints(
            component, "marginal_cost", pw_names, cumulative_attr, invert_attr
        )
        expected_x = [0, 50, 100]
        assert np.allclose(x_breakpoints.to_series().values, expected_x)

    def test_cumulative_attr(self, component, pw_names) -> None:
        """Test expected y breakpoints are returned when cumulative_attr is True."""
        _, y_breakpoints = _get_breakpoints(
            component,
            "marginal_cost",
            pw_names,
            cumulative_attr=True,
            invert_attr=False,
        )
        assert np.allclose(y_breakpoints.sel(name="gen"), [0, 1000, 3000])

    def test_not_cumulative_attr(self, component, pw_names) -> None:
        """Test expected y breakpoints are returned when cumulative_attr is False."""
        _, y_breakpoints = _get_breakpoints(
            component,
            "marginal_cost",
            pw_names,
            cumulative_attr=False,
            invert_attr=False,
        )
        assert np.allclose(y_breakpoints.sel(name="gen"), [0, 1000, 4000])

    def test_invert_attr(self, component, pw_names) -> None:
        """Test expected y breakpoints are returned when invert_attr is True."""
        _, y_breakpoints = _get_breakpoints(
            component,
            "marginal_cost",
            pw_names,
            cumulative_attr=False,
            invert_attr=True,
        )
        assert np.allclose(y_breakpoints.sel(name="gen"), [0, 2.5, 2.5])

    def test_invert_cumulative_attr(self, component, pw_names) -> None:
        """Test expected y breakpoints are returned when invert_attr and cumulative_attr are True."""
        _, y_breakpoints = _get_breakpoints(
            component, "marginal_cost", pw_names, cumulative_attr=True, invert_attr=True
        )
        assert np.allclose(y_breakpoints.sel(name="gen"), [0, 2.5, 3.75])

    def test_not_allow_extendables_extendable(
        self, component_extendable, pw_names
    ) -> None:
        """Test that ValueError is raised when x_attr is per-unit for extendable components."""
        expected = r"Piecewise 'marginal_cost' breakpoints on a per-unit x-axis are not supported for extendable components (fixed p_nom required). Extendable components: ['gen']."
        with pytest.raises(ValueError) as excinfo:
            _get_breakpoints(
                component_extendable,
                "marginal_cost",
                pw_names,
                cumulative_attr=False,
                invert_attr=False,
            )
        assert str(excinfo.value) == expected

    @pytest.mark.parametrize("p_nom", [np.nan, np.inf, 0.0])
    def test_not_allow_extendables_missing_nom(
        self, component, pw_names, p_nom
    ) -> None:
        """Test that ValueError is raised when x_attr is per-unit for components with badly formatted nominal attribute."""
        expected = r"Piecewise 'marginal_cost' breakpoints on a per-unit x-axis cannot be scaled to absolute values for components with non-positive, non-finite or missing p_nom. Problematic components: ['gen']."
        component.static["p_nom"] = p_nom
        with pytest.raises(ValueError) as excinfo:
            _get_breakpoints(
                component,
                "marginal_cost",
                pw_names,
                cumulative_attr=False,
                invert_attr=False,
            )
        assert str(excinfo.value) == expected


class TestCreateYVar:
    @pytest.fixture(scope="class")
    def pw_names(self) -> pd.Series:
        return pd.Index(["gen1"], name="name")

    @pytest.fixture(scope="class")
    def snapshots(self):
        return pd.Index([1, 2, 3], name="snapshot")

    @pytest.fixture(scope="class")
    def linopy_model(self, snapshots):
        m = linopy.Model()
        m.add_variables(
            name="x_static", coords=[pd.Index(["gen1", "gen2"], name="name")]
        )
        m.add_variables(
            name="x_dynamic",
            coords=[pd.Index(["gen1", "gen2"], name="name"), snapshots],
        )
        return m

    @pytest.fixture(scope="class")
    def y_var_static(self, linopy_model, pw_names) -> linopy.Variable:
        return _create_y_var(
            linopy_model, linopy_model["x_static"], pw_names, "foo-static"
        )

    @pytest.fixture(scope="class")
    def y_var_dynamic(self, linopy_model, pw_names) -> linopy.Variable:
        return _create_y_var(
            linopy_model, linopy_model["x_dynamic"], pw_names, "foo-dynamic"
        )

    @pytest.mark.parametrize("y_var_name", ["y_var_static", "y_var_dynamic"])
    def test_y_var_name(self, request, y_var_name):
        """Test that created y variable has expected name."""
        y_var = request.getfixturevalue(y_var_name)
        assert y_var.name == f"foo-{y_var_name.removeprefix('y_var_')}"

    @pytest.mark.parametrize("y_var_name", ["y_var_static", "y_var_dynamic"])
    def test_y_var_lb(self, request, y_var_name):
        """Test that created y variable is unbounded below."""
        y_var = request.getfixturevalue(y_var_name)
        assert (y_var.lower == -float("inf")).all()

    def test_y_var_dims_static(self, y_var_static, pw_names):
        """Test that static y variable has expected coords."""
        assert y_var_static.coords.equals(xr.Coordinates(coords={"name": pw_names}))

    def test_y_var_dims_dynamic(self, y_var_dynamic, pw_names, snapshots):
        """Test that dynamic y variable has expected coords."""
        assert y_var_dynamic.coords.equals(
            xr.Coordinates(coords={"name": pw_names, "snapshot": snapshots})
        )


class TestToDa:
    @pytest.fixture
    def da(self) -> pd.DataFrame:
        df = _piecewise_df({"gen1": [(0.0, 10.0), (0.5, 20.0), (1.0, 40.0)]})
        da = _to_da(df, "marginal_cost")
        return da

    def test_to_da(self, da):
        """Test that DataFrame is converted to DataArray with expected dims."""
        assert set(da.dims) == {linopy.constants.BREAKPOINT_DIM, "name"}


class TestDefinePiecewise:
    @pytest.fixture(scope="class")
    def snapshots(self):
        return pd.Index([1, 2, 3], name="snapshot")

    @pytest.fixture
    def component(self) -> SimpleNamespace:
        component = SimpleNamespace(
            piecewise={
                "marginal_cost": _piecewise_df(
                    {
                        f"gen{i}": [(0.0, 10.0), (0.5, 20.0), (1.0, 40.0)]
                        for i in range(4)
                    },
                    x_attr="p_pu",
                    y_attr="marginal_cost",
                ),
                "capital_cost": _piecewise_df(
                    {"gen_extendable": [(0.0, 100.0), (0.5, 200.0), (1.0, 400.0)]},
                    x_attr="p_nom",
                    y_attr="capital_cost",
                ),
            },
            static=pd.DataFrame(
                index=[f"gen{i}" for i in range(4)], data={"p_nom": 100.0}
            ),
            name="Generator",
            extendables=pd.Index(["gen_extendable"], name="name"),
            _piecewise_attrs=PIECEWISE_ATTRS.query("component == 'Generator'"),
        )
        component.piecewise["efficiency"] = pd.DataFrame()
        return component

    @pytest.fixture
    def linopy_model(self, snapshots):
        m = linopy.Model()
        idx = pd.Index([f"gen{i}" for i in range(4)] + ["gen_extendable"], name="name")
        m.add_variables(name="x_static", coords=[idx])
        m.add_variables(name="y_static", coords=[idx])
        m.add_variables(name="x_dynamic", coords=[idx, snapshots])
        return m

    @pytest.fixture(scope="class")
    def piecewise_options(self):
        options = PiecewiseOptions(
            component="Generator",
            attribute="marginal_cost",
            sign="=",
        )
        return options

    @pytest.fixture
    def default_kwargs(self, piecewise_options, linopy_model, component):
        return {
            "m": linopy_model,
            "c": component,
            "sign": piecewise_options.sign,
            "cumulative_attr": piecewise_options.cumulative_attr,
            "method": piecewise_options.method,
            "extra_options": [],
        }

    @pytest.fixture
    def constraint_keys(self, piecewise_options, default_kwargs, linopy_model):
        """Helper fixture to get constraint keys created by define_piecewise with different extra options."""

        def _get_constraint_keys(
            replace_options: list[dict], active_names: list[str]
        ) -> pd.Series:
            extra_options = [
                replace(piecewise_options, **opts) for opts in replace_options
            ]
            kwargs = {**default_kwargs, "extra_options": extra_options}
            _ = define_piecewise(
                x_var=linopy_model["x_static"],
                pw_attr="marginal_cost",
                aux_var_name="foo_static",
                active_names=pd.Index(active_names, name="name"),
                **kwargs,
            )
            keys = pd.Series(linopy_model.constraints)
            assert len(keys) > 0
            return keys

        return _get_constraint_keys

    @pytest.mark.parametrize(
        "pw_names", [pd.Index([], name="name"), pd.Index(["store"], name="name")]
    )
    @pytest.mark.parametrize("pw_attr", ["marginal_cost", "capital_cost"])
    def test_pw_names_empty_no_active(
        self, caplog, default_kwargs, linopy_model, pw_names, pw_attr
    ):
        """Test that no piecewise variables or constraints are created when pw_names is empty due t o no active names."""
        caplog.set_level(logging.DEBUG, logger="pypsa.optimization.piecewise")
        result = define_piecewise(
            x_var=linopy_model["x_static"],
            pw_attr=pw_attr,
            aux_var_name="foo_static",
            active_names=pw_names,
            **default_kwargs,
        )
        assert result is None
        assert "Skipping piecewise constraint." in caplog.text
        assert "foo_static" not in linopy_model.variables

    @pytest.mark.parametrize("pw_attr", ["foo", "efficiency"])
    def test_pw_names_empty_no_pw(self, caplog, default_kwargs, linopy_model, pw_attr):
        """Test that no piecewise variables or constraints are created when pw_names is empty due to no piecewise constraints."""
        caplog.set_level(logging.DEBUG, logger="pypsa.optimization.piecewise")
        result = define_piecewise(
            x_var=linopy_model["x_static"],
            pw_attr=pw_attr,
            aux_var_name="foo_static",
            active_names=pd.Index(["gen1"], name="name"),
            **default_kwargs,
        )
        assert result is None
        assert "Skipping piecewise constraint." in caplog.text
        assert "foo_static" not in linopy_model.variables

    def test_provide_y_var(self, default_kwargs, linopy_model):
        """Test that provided y variable is used in piecewise constraint."""
        result = define_piecewise(
            x_var=linopy_model["x_static"],
            y_var=linopy_model["y_static"],
            pw_attr="marginal_cost",
            aux_var_name="foo_static",
            active_names=pd.Index(["gen1"], name="name"),
            **default_kwargs,
        )
        assert result == linopy_model["y_static"]
        assert "foo_static" not in linopy_model.variables

    @pytest.mark.parametrize(
        ("name", "active_names", "expected_suffix"),
        [
            ("gen2", ["gen2"], "gen2"),
            (["gen1"], ["gen1"], "gen1"),
            (["gen1", "gen2"], ["gen1"], "gen1"),
            (
                ["gen0", "gen1", "gen2", "gen3"],
                ["gen0", "gen1", "gen2", "gen3"],
                "gen0_..._gen3",
            ),
            (["gen0", "gen1", "gen2", "gen3"], ["gen1", "gen3"], "gen1_gen3"),
        ],
    )
    def test_named_option_string_all_as_options(
        self,
        constraint_keys,
        name,
        active_names,
        expected_suffix,
    ):
        """Test that extra option filtering on a name is used in piecewise constraint (all names covered by extra options)."""
        keys = constraint_keys([{"name": name}], active_names)
        assert keys.str.startswith(f"foo_static_{expected_suffix}").all()

    @pytest.mark.parametrize(
        ("name", "active_names", "expected_suffix"),
        [
            ("gen0", ["gen0", "gen1", "gen2", "gen3", "gen3"], "gen0"),
            (["gen1", "gen2"], ["gen0", "gen1", "gen2", "gen3"], "gen1_gen2"),
            (
                ["gen1", "gen2", "gen3"],
                ["gen0", "gen1", "gen2", "gen3"],
                "gen1_gen2_gen3",
            ),
        ],
    )
    def test_named_option_string_some_as_options(
        self,
        constraint_keys,
        name,
        active_names,
        expected_suffix,
    ):
        """Test that extra option filtering on a name is used in piecewise constraint (some names not covered by extra options)."""
        keys = constraint_keys([{"name": name}], active_names)
        n_w_suffix = keys.str.startswith(f"foo_static_{expected_suffix}").sum()
        n_all = keys.str.startswith("foo_static").sum()
        assert n_w_suffix == n_all / 2

    def test_unnamed_option(self, constraint_keys):
        """Test that extra option filtering without a name."""
        keys = constraint_keys([{"sign": ">="}], ["gen0", "gen1"])
        assert not keys.str.contains("gen0").any()
        assert keys.str.startswith("foo_static").all()

    @pytest.mark.parametrize(
        "names", [[["gen0", "gen1"], ["gen2"]], [["gen0"], ["gen1", "gen2"]]]
    )
    def test_named_then_named_option(self, constraint_keys, names):
        """Test that extra option filtering with multiple extra options (all with name filtering)."""
        keys = constraint_keys(
            [{"name": names[0]}, {"name": names[1]}], ["gen0", "gen1", "gen2"]
        )
        n_0 = keys.str.startswith(f"foo_static_{'_'.join(names[0])}").sum()
        n_1 = keys.str.startswith(f"foo_static_{'_'.join(names[1])}").sum()
        assert n_0 > 0
        assert n_0 == n_1

    @pytest.mark.parametrize("names", [["gen1", None], [None, "gen1"]])
    def test_unnamed_then_named_option(self, constraint_keys, names):
        """Test that extra option filtering with multiple extra options (one without name filtering)."""
        keys = constraint_keys(
            [{"name": names[0]}, {"name": names[1]}], ["gen0", "gen1"]
        )
        n_w_suffix = keys.str.startswith("foo_static_gen1").sum()
        n_all = keys.str.startswith("foo_static").sum()
        assert n_w_suffix == n_all / 2
