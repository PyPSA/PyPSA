# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for piecewise breakpoint data handling in ``n.add``.

Coverage
--------
* Components are split into those that define a single ``bus`` and those
  that define ``bus0`` / ``bus1``.
* Multi-port attributes for Link (``efficiency2``) and Process (``rate2``).
* All three user-facing input formats: plain dict, two-column DataFrame, and
  MultiIndex-columned DataFrame.
* Error paths: dict passed for a non-piecewise attribute; piecewise *per-unit*
  attributes combined with extendable capacity.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

import pypsa
from pypsa.constants import piecewise_attrs, piecewise_schema
from pypsa.descriptors import nominal_attrs
from pypsa.network.io import NetworkIOMixin

# ---------------------------------------------------------------------------
# Component catalogue
# ---------------------------------------------------------------------------

# Minimum required kwargs to add a valid instance of each component.
# One-bus components connect to a single ``bus=`` keyword.
BASE_KWARGS: dict[str, dict] = {
    "Generator": {"bus": "bus_ac", "p_nom": 100},
    "StorageUnit": {"bus": "bus_ac", "p_nom": 100},
    "Store": {"bus": "bus_ac", "e_nom": 100},
    "Line": {"bus0": "bus_ac", "bus1": "bus_ac2", "x": 0.1, "r": 0.01},
    "Link": {"bus0": "bus_ac", "bus1": "bus_ac2", "p_nom": 100},
    "Process": {"bus0": "bus_ac", "bus1": "bus_ac2", "p_nom": 100},
    "Transformer": {"bus0": "bus_ac", "bus1": "bus_ac2", "x": 0.1, "r": 0.01},
}

# ---------------------------------------------------------------------------
# Parametrize-list builders
# ---------------------------------------------------------------------------

CURVE_DICT: dict[float, float] = {0.0: 1.0, 0.5: 0.7, 1.0: 0.4}


def _build_params(base_kwargs_by_comp: dict[str, dict]) -> list:
    """Build ``(comp, base_kwargs, attr, x_attr)`` pytest.param entries.

    Attributes and their x-axis coordinate are read from
    the piecewise schema.
    """
    params = []
    for comp, base_kw in base_kwargs_by_comp.items():
        for _, attr in piecewise_attrs(comp).iterrows():
            if comp == "Process" and attr.y == "rate":
                y = "rate1"
            else:
                y = attr.y
            to_append = pytest.param(comp, base_kw, y, attr.x, id=f"{comp}-{y}")
            params.append(to_append)
    return params


ALL_PARAMS = _build_params(BASE_KWARGS)
ALL_PU_PARAMS = [p for p in ALL_PARAMS if p.values[3] != nominal_attrs[p.values[0]]]


def _build_extendable_params(base_kwargs_by_comp: dict[str, dict]) -> tuple[list, list]:
    """Split CSV piecewise attrs into per-unit (raises) and nom-based (allowed) sets.

    Returns
    -------
    pu_raises : list of pytest.param
        ``(comp, kwargs_with_extendable, attr)`` where the x-axis attribute is
        per-unit (``p_pu``, ``e_pu``, etc.).  Piecewise breakpoints on such attrs
        must be rejected when the nominal capacity is extendable.
    nom_allowed : list of pytest.param
        Same shape but for attrs whose x-axis is the nominal capacity itself
        (``p_nom``, ``s_nom``, ``e_nom``).  These are always permitted.
    """
    pu_raises: list = []
    nom_allowed: list = []
    for comp, base_kw in base_kwargs_by_comp.items():
        nom = nominal_attrs[comp]
        kw = {**base_kw, f"{nom}_extendable": True}
        for _, attr in piecewise_attrs(comp).iterrows():
            p = pytest.param(comp, kw, attr.y, id=f"{comp}-{attr.y}")
            if attr.x != nom:
                pu_raises.append(p)
            else:
                nom_allowed.append(p)
    return pu_raises, nom_allowed


EXTENDABLE_PU_RAISES_PARAMS, EXTENDABLE_NOM_ALLOWED_PARAMS = _build_extendable_params(
    BASE_KWARGS
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_plain_df(x_attr: str, y_attr: str) -> pd.DataFrame:
    """Two-column DataFrame ``[x_attr, y_attr]`` from CURVE_DICT."""
    return pd.DataFrame({x_attr: list(CURVE_DICT), y_attr: list(CURVE_DICT.values())})


def _make_multiindex_df(x_attr: str, y_attr: str, names: list[str]) -> pd.DataFrame:
    """MultiIndex-columned piecewise DataFrame for the given component names."""
    plain = {name: _make_plain_df(x_attr, y_attr) for name in names}
    mi = pd.concat(plain.values(), keys=plain.keys(), axis=1).rename_axis(
        columns=["name", "attribute"], index="breakpoint"
    )
    return mi


def _assert_piecewise_stored(
    pw_df: pd.DataFrame,
    expected_names: list[str],
    x_attr: str,
    y_attr: str,
) -> None:
    expected = _make_multiindex_df(x_attr, y_attr, expected_names)
    pd.testing.assert_frame_equal(pw_df[expected_names], expected)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_network() -> pypsa.Network:
    """Minimal network with buses required by all component tests."""
    n = pypsa.Network()
    n.add("Bus", ["bus_ac", "bus_ac2", "bus_dc"], carrier=["AC", "AC", "DC"])
    return n


# ===========================================================================
# 1. All piecewise-capable components
# ===========================================================================


@pytest.mark.parametrize(("comp", "base_kwargs", "attr", "x_attr"), ALL_PARAMS)
class TestPiecewise:
    """Piecewise attrs from the piecewise schema; parametrised over all components."""

    def test_dict_input(self, base_network, comp, base_kwargs, attr, x_attr):
        n = base_network
        n.add(comp, "c1", **base_kwargs, **{attr: CURVE_DICT})
        _assert_piecewise_stored(
            n.components[comp].piecewise[attr], ["c1"], x_attr, attr
        )

    def test_plain_df_input(self, base_network, comp, base_kwargs, attr, x_attr):
        n = base_network
        n.add(comp, "c1", **base_kwargs, **{attr: _make_plain_df(x_attr, attr)})
        _assert_piecewise_stored(
            n.components[comp].piecewise[attr], ["c1"], x_attr, attr
        )

    def test_multiindex_df_input(self, base_network, comp, base_kwargs, attr, x_attr):
        n = base_network
        mi_df = _make_multiindex_df(x_attr, attr, ["c1"])
        n.add(comp, "c1", **base_kwargs, **{attr: mi_df})
        _assert_piecewise_stored(
            n.components[comp].piecewise[attr], ["c1"], x_attr, attr
        )

    def test_dict_broadcast_to_multiple_components(
        self, base_network, comp, base_kwargs, attr, x_attr
    ):
        """A dict is broadcast identically to all named components."""
        n = base_network
        n.add(comp, ["c1", "c2"], **base_kwargs, **{attr: CURVE_DICT})
        seg = n.components[comp].piecewise[attr]
        _assert_piecewise_stored(seg, ["c1", "c2"], x_attr, attr)


# ===========================================================================
# 2. Multi-port: Link efficiency2 / Process rate2
# ===========================================================================


@pytest.mark.parametrize(
    ("comp", "attr"), [("Link", "efficiency2"), ("Process", "rate2")]
)
class TestMultiportPiecewise:
    """Port-suffixed (port ≥ 2) piecewise attributes for Links and Processes."""

    @pytest.fixture
    def multiport_kwargs(self, comp) -> dict:
        return {**BASE_KWARGS[comp], "bus2": "bus_dc"}

    def test_dict_input(self, base_network, comp, attr, multiport_kwargs):
        n = base_network
        n.add(comp, "c1", **multiport_kwargs, **{attr: CURVE_DICT})
        _assert_piecewise_stored(
            n.components[comp].piecewise[attr], ["c1"], "p_pu", attr
        )

    def test_plain_df_input(self, base_network, comp, attr, multiport_kwargs):
        n = base_network
        n.add(comp, "c1", **multiport_kwargs, **{attr: _make_plain_df("p_pu", attr)})
        _assert_piecewise_stored(
            n.components[comp].piecewise[attr], ["c1"], "p_pu", attr
        )

    def test_multiindex_df_input(self, base_network, comp, attr, multiport_kwargs):
        n = base_network
        mi_df = _make_multiindex_df("p_pu", attr, ["c1"])
        n.add(comp, "c1", **multiport_kwargs, **{attr: mi_df})
        _assert_piecewise_stored(
            n.components[comp].piecewise[attr], ["c1"], "p_pu", attr
        )


# ===========================================================================
# 3. Error paths
# ===========================================================================


class TestPiecewiseErrors:
    def test_dict_for_non_piecewise_attr_raises(self, base_network):
        """A dict passed to a plain scalar attr raises ``TypeError``."""
        n = base_network
        with pytest.raises(TypeError, match="Dictionaries are not supported"):
            n.add("Generator", "gen", bus="bus_ac", p_nom={0: 100})

    def test_multiindex_df_wrong_attribute_labels_raises_x(self, base_network):
        """MultiIndex input with wrong attribute-level labels is rejected."""
        n = base_network
        mi_df = _make_multiindex_df("WRONG_X", "marginal_cost", ["gen"])
        with pytest.raises(
            ValueError, match="Piecewise marginal_cost Dataframe has attribute column"
        ):
            n.add("Generator", "gen", bus="bus_ac", p_nom=100, marginal_cost=mi_df)

    def test_multiindex_df_wrong_attribute_labels_raises_y(self, base_network):
        """MultiIndex input with wrong name-level labels is rejected."""
        n = base_network
        mi_df = _make_multiindex_df("p_pu", "WRONG_Y", ["gen"])
        with pytest.raises(
            ValueError, match="Piecewise marginal_cost Dataframe has attribute column"
        ):
            n.add("Generator", "gen", bus="bus_ac", p_nom=100, marginal_cost=mi_df)

    def test_multiindex_df_wrong_attribute_labels_raises_name(self, base_network):
        """MultiIndex input with wrong attribute-level labels is rejected."""
        n = base_network
        mi_df = _make_multiindex_df("p_pu", "marginal_cost", ["WRONG_NAME"])
        with pytest.raises(
            ValueError, match="Piecewise marginal_cost Dataframe has name column"
        ):
            n.add("Generator", "gen", bus="bus_ac", p_nom=100, marginal_cost=mi_df)

    @pytest.mark.parametrize(
        ("component", "attr"),
        [
            ("Process", "rate0"),
            ("Process", "rate1"),
            ("Link", "efficiency"),
            ("Link", "efficiency2"),
        ],
    )
    def test_multiport_pos_neg_mix_raises(self, base_network, component, attr):
        """Multi-port piecewise attrs must not mix positive and negative values."""
        n = base_network
        curve = pd.DataFrame({"p_pu": [0.0, 0.5, 1.0], attr: [0.0, -0.7, 0.4]})
        with pytest.raises(
            NotImplementedError,
            match=f"Cannot mix positive and negative values for piecewise {attr} curves",
        ):
            n.add(
                component,
                "foo",
                bus0="bus_ac",
                bus1="bus_ac2",
                bus2="bus_dc",
                p_nom=100,
                **{attr: curve},
            )

    @pytest.mark.parametrize(
        ("comp", "extendable_kwargs", "attr"), EXTENDABLE_PU_RAISES_PARAMS
    )
    def test_pu_piecewise_on_extendable_component_raises(
        self, base_network, comp, extendable_kwargs, attr
    ):
        """Per-unit piecewise attrs must be rejected when nominal capacity is extendable."""
        n = base_network
        with pytest.raises(ValueError, match="Piecewise"):
            n.add(comp, "c1", **extendable_kwargs, **{attr: CURVE_DICT})

    @pytest.mark.parametrize(
        ("comp", "extendable_kwargs", "attr"), EXTENDABLE_NOM_ALLOWED_PARAMS
    )
    def test_nom_piecewise_on_extendable_component_allowed(
        self, base_network, comp, extendable_kwargs, attr
    ):
        """Nom-based piecewise attrs (e.g. capital_cost vs s_nom) are always permitted."""
        n = base_network
        n.add(comp, "c1", **extendable_kwargs, **{attr: CURVE_DICT})
        assert not n.components[comp].piecewise[attr].empty

    @pytest.mark.parametrize(("comp", "base_kwargs", "attr", "x_attr"), ALL_PARAMS)
    def test_first_x_above_zero_raises(
        self, base_network, comp, base_kwargs, attr, x_attr
    ):
        """Test that a first x breakpoint above zero raises for cumulative curves only."""
        n = base_network
        curve = pd.DataFrame({x_attr: [0.1, 0.5, 1.0], attr: [0.0, 0.1, 0.4]})
        with pytest.raises(ValueError, match=rf"must start at {x_attr}=0"):
            n.add(comp, "c1", **base_kwargs, **{attr: curve})

    @pytest.mark.parametrize(("comp", "base_kwargs", "attr", "x_attr"), ALL_PU_PARAMS)
    def test_last_x_below_one_raises(
        self, base_network, comp, base_kwargs, attr, x_attr
    ):
        """Test that a last x breakpoint below one raises for cumulative curves only."""
        n = base_network
        curve = pd.DataFrame({x_attr: [0.0, 0.5, 0.8], attr: [0.0, 0.1, 0.4]})
        # monkeypatch the kwargs to not have the breaking extendable attr on a per-unit piecewise attr
        with pytest.raises(ValueError, match=rf"must end at {x_attr}=1"):
            n.add(comp, "c1", **base_kwargs, **{attr: curve})

    @pytest.mark.parametrize(("comp", "base_kwargs", "attr", "x_attr"), ALL_PU_PARAMS)
    def test_last_x_below_one_raises_handles_nan(
        self, base_network, comp, base_kwargs, attr, x_attr
    ):
        """Test that a last x breakpoint below one raises for cumulative curves only."""
        n = base_network
        curve1 = pd.DataFrame({x_attr: [0.0, 0.5, 1], attr: [0.0, 0.1, 0.4]})
        curve2 = pd.DataFrame({x_attr: [0.0, 0.8], attr: [0.0, 0.1]})
        # monkeypatch the kwargs to not have the breaking extendable attr on a per-unit piecewise attr
        n.add(comp, "c1", **base_kwargs, **{attr: curve1})
        with pytest.raises(ValueError, match=rf"must end at {x_attr}=1"):
            n.add(comp, "c2", **base_kwargs, **{attr: curve2})

    @pytest.mark.parametrize(("comp", "base_kwargs", "attr", "x_attr"), ALL_PARAMS)
    def test_first_y_ignored_warning(
        self, caplog, base_network, comp, base_kwargs, attr, x_attr
    ) -> None:
        """Test that ignored non-zero first y values on cumulative curves log a warning."""

        n = base_network
        curve = pd.DataFrame({x_attr: [0.0, 0.5, 1.0], attr: [1.0, 0.1, 0.4]})
        with caplog.at_level(logging.WARNING, logger="pypsa.network.io"):
            n.add(comp, "c1", **base_kwargs, **{attr: curve})
        if attr in ["marginal_cost", "capital_cost"]:
            assert (
                f"Piecewise '{attr}' values price the increment from the previous breakpoint, so the y-value at x=0 spans zero width and will be ignored."
                in caplog.text
            )
        elif attr.startswith(("rate", "efficiency")):
            assert (
                f"A non-zero y value at x=0 for piecewise '{attr}' will be ignored when the piecewise constraint is defined"
                in caplog.text
            )


class TestNormalizeBreakpoints:
    @pytest.fixture(scope="class")
    @classmethod
    def gen_marginal_cost_attrs(cls) -> pd.Series:
        return piecewise_schema("Generator", "marginal_cost")

    def _piecewise_df(
        self,
        curves: dict[str, list[tuple[float, float]]],
        x_attr: str = "p_pu",
        y_attr: str = "marginal_cost",
    ) -> pd.DataFrame:
        """Build a (name, attribute)-columned DataFrame from {name: [(x, y), ...]}."""
        frames = {
            n: pd.DataFrame(rows, columns=[x_attr, y_attr])
            for n, rows in curves.items()
        }
        return pd.concat(frames, axis=1, names=["name", "attribute"]).rename_axis(
            index="breakpoint"
        )

    def test_sorts_unsorted_rows(self, gen_marginal_cost_attrs: pd.Series) -> None:
        df = self._piecewise_df({"gen": [(1.0, 40.0), (0.0, 10.0), (0.5, 20.0)]})
        result = NetworkIOMixin._normalize_breakpoints(df, gen_marginal_cost_attrs)
        assert result["gen"]["p_pu"].tolist() == [0.0, 0.5, 1.0]
        assert result["gen"]["marginal_cost"].tolist() == [10.0, 20.0, 40.0]
        assert result.index.name == "breakpoint"

    def test_ragged_curves_aligned_with_trailing_nan(
        self, gen_marginal_cost_attrs: pd.Series
    ) -> None:
        df = self._piecewise_df(
            {
                "gen0": [(0.0, 10.0), (0.5, 20.0), (1.0, 40.0)],
                "gen1": [(0.0, 5.0), (1.0, 25.0)],
            }
        )
        result = NetworkIOMixin._normalize_breakpoints(df, gen_marginal_cost_attrs)
        assert len(result) == 3
        assert result["gen0"]["p_pu"].tolist() == [0.0, 0.5, 1.0]
        assert result["gen1"]["p_pu"].iloc[:2].tolist() == [0.0, 1.0]
        assert result["gen1"]["p_pu"].isnull().iloc[2]

    def test_idempotent(self, gen_marginal_cost_attrs: pd.Series) -> None:
        df = self._piecewise_df({"gen": [(1.0, 40.0), (0.0, 10.0), (0.5, 20.0)]})
        once = NetworkIOMixin._normalize_breakpoints(df, gen_marginal_cost_attrs)
        twice = NetworkIOMixin._normalize_breakpoints(once, gen_marginal_cost_attrs)
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
        df = self._piecewise_df(curves)
        with pytest.raises(ValueError, match=match):
            NetworkIOMixin._normalize_breakpoints(df, gen_marginal_cost_attrs)


def test_remove_drops_piecewise_data(base_network):
    """Removed components must not leave stale piecewise curves behind."""
    n = base_network
    n.add("Generator", "gen", bus="bus_ac", p_nom=100, marginal_cost=CURVE_DICT)
    n.remove("Generator", "gen")
    assert n.c.generators.piecewise["marginal_cost"].empty
    n.add("Generator", "gen", bus="bus_ac", p_nom=100, marginal_cost=5.0)
    assert n.c.generators.piecewise["marginal_cost"].empty


class TestPiecewiseHelpers:
    """Schema-vs-data primitives on the component class."""

    @staticmethod
    @pytest.fixture(scope="class")
    def base_network() -> pypsa.Network:
        """Minimal network with buses required by all component tests."""
        n = pypsa.Network()
        n.add("Bus", ["bus_ac", "bus_ac2", "bus_dc"], carrier=["AC", "AC", "DC"])
        n.add("Link", "link", bus0="bus_ac", bus1="bus_ac2", bus2="bus_dc")
        return n

    def test_has_piecewise_tracks_data_not_schema(self, base_network):
        """`has_piecewise` reflects breakpoint data, not the schema definition."""
        n = base_network
        c = n.c.generators
        assert not c.has_piecewise("marginal_cost")
        n.add("Generator", "gen", bus="bus_ac", p_nom=100, marginal_cost=CURVE_DICT)
        assert c.has_piecewise("marginal_cost")
        assert not c.has_piecewise("capital_cost")

    @pytest.mark.parametrize(
        ("comp", "attr", "expected"),
        [
            ("Generator", "marginal_cost", "Generator-marginal_cost_piecewise"),
            ("Process", "rate0", "Process-p0_piecewise"),
            ("Process", "rate1", "Process-p1_piecewise"),
            ("Link", "efficiency", "Link-p1_piecewise"),
            ("Link", "efficiency2", "Link-p2_piecewise"),
        ],
    )
    def test_aux_var_is_schema_lookup(self, base_network, comp, attr, expected):
        """`_piecewise_aux_var` resolves from the schema, independent of data."""
        assert base_network.c[comp]._piecewise_aux_var(attr) == expected

    def test_aux_var_raises_on_unknown_attr(self, base_network):
        """A non-piecewise attribute fails fast instead of returning None."""
        with pytest.raises(ValueError, match="no piecewise schema"):
            base_network.c.generators._piecewise_aux_var("not_an_attr")

    @pytest.mark.parametrize(
        ("comp", "attr", "expected"),
        [
            ("Generator", "marginal_cost", "Generator-p"),
            ("Process", "rate0", "Process-p"),
            ("Process", "rate1", "Process-p"),
            ("Link", "efficiency", "Link-p"),
            ("Link", "efficiency2", "Link-p"),
        ],
    )
    def test_x_var_is_schema_lookup(self, base_network, comp, attr, expected):
        """`_piecewise_x_var` resolves from the schema, independent of data."""
        assert base_network.c[comp]._piecewise_x_var(attr) == expected

    def test_x_var_raises_on_unknown_attr(self, base_network):
        """A non-piecewise attribute fails fast instead of returning None."""
        with pytest.raises(ValueError, match="no piecewise schema"):
            base_network.c.generators._piecewise_x_var("not_an_attr")

    @pytest.mark.parametrize(
        ("comp", "attr", "expected_y"),
        [
            ("Generator", "marginal_cost", "marginal_cost"),
            ("Link", "efficiency", "efficiency"),
            ("Link", "efficiency2", "efficiency"),
            ("Process", "rate0", "rate"),
        ],
    )
    def test_piecewise_schema(self, base_network, comp, attr, expected_y):
        """`_piecewise_schema` resolves port-suffixed attrs to their definition row."""
        schema = base_network.c[comp]._piecewise_schema(attr)
        assert isinstance(schema, pd.Series)
        assert schema.y == expected_y

    def test_piecewise_schema_empty(self, base_network):
        """`_piecewise_schema` returns an empty series when no match."""
        schema = base_network.c.generators._piecewise_schema("foo")
        assert schema.empty


class TestPiecewiseScenarios:
    def test_add_piecewise_on_stochastic_network_raises(self, base_network):
        n = base_network
        n.set_scenarios({"low": 0.5, "high": 0.5})
        with pytest.raises(NotImplementedError, match="not yet supported"):
            n.add("Generator", "gen", bus="bus_ac", p_nom=100, marginal_cost=CURVE_DICT)

    def test_set_scenarios_with_piecewise_data_raises(self, base_network):
        n = base_network
        n.add("Generator", "gen", bus="bus_ac", p_nom=100, marginal_cost=CURVE_DICT)
        with pytest.raises(NotImplementedError, match="not yet supported"):
            n.set_scenarios({"low": 0.5, "high": 0.5})
