# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for piecewise segment data handling in ``n.add``.

Coverage
--------
* Each component type for which ``segments_x`` is defined in
  ``ComponentType.segments_attrs`` (Generator, Line, Link, Process, StorageUnit,
  Store, Transformer), parametrised over both the component and the attribute.
* Components are split into those that define a single ``bus`` and those
  that define ``bus0`` / ``bus1``.
* Multi-port attributes for Link (``efficiency2``) and Process (``rate2``).
* All three user-facing input formats: plain dict, two-column DataFrame, and
  MultiIndex-columned DataFrame.
* Error paths: dict passed for a non-segment attribute; piecewise *per-unit*
  attributes combined with extendable capacity.
"""

from __future__ import annotations

import pandas as pd
import pytest

import pypsa
from pypsa.components.types import get as _get_ctype
from pypsa.descriptors import nominal_attrs

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
    ``ComponentType.segments_attrs``.
    """
    params = []
    for comp, base_kw in base_kwargs_by_comp.items():
        for attr, x_attr in _get_ctype(comp).segments_attrs.items():
            params.append(
                pytest.param(comp, base_kw, attr, x_attr, id=f"{comp}-{attr}")
            )
    return params


ALL_PARAMS = _build_params(BASE_KWARGS)


def _build_extendable_params(base_kwargs_by_comp: dict[str, dict]) -> tuple[list, list]:
    """Split CSV segment attrs into per-unit (raises) and nom-based (allowed) sets.

    Returns
    -------
    pu_raises : list of pytest.param
        ``(comp, kwargs_with_extendable, attr)`` where the x-axis attribute is
        per-unit (``p_pu``, ``e_pu``, etc.).  Piecewise segments on such attrs
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
        for attr, x_attr in _get_ctype(comp).segments_attrs.items():
            p = pytest.param(comp, kw, attr, id=f"{comp}-{attr}")
            if x_attr != nom:
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
    """MultiIndex-columned segment DataFrame for the given component names."""
    plain = {name: _make_plain_df(x_attr, y_attr) for name in names}
    mi = pd.concat(plain.values(), keys=plain.keys(), axis=1).rename_axis(
        columns=["name", "attribute"], index="segment"
    )
    return mi


def _assert_segment_stored(
    seg_df: pd.DataFrame,
    expected_names: list[str],
    x_attr: str,
    y_attr: str,
) -> None:
    expected = _make_multiindex_df(x_attr, y_attr, expected_names)
    pd.testing.assert_frame_equal(seg_df[expected_names], expected)


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
# 1. All segment-capable components
# ===========================================================================


@pytest.mark.parametrize(("comp", "base_kwargs", "attr", "x_attr"), ALL_PARAMS)
class TestSegments:
    """Segment attrs from ComponentType.segments_attrs; parametrised over all components."""

    def test_dict_input(self, base_network, comp, base_kwargs, attr, x_attr):
        n = base_network
        n.add(comp, "c1", **base_kwargs, **{attr: CURVE_DICT})
        _assert_segment_stored(n.components[comp].segments[attr], ["c1"], x_attr, attr)

    def test_plain_df_input(self, base_network, comp, base_kwargs, attr, x_attr):
        n = base_network
        n.add(comp, "c1", **base_kwargs, **{attr: _make_plain_df(x_attr, attr)})
        _assert_segment_stored(n.components[comp].segments[attr], ["c1"], x_attr, attr)

    def test_multiindex_df_input(self, base_network, comp, base_kwargs, attr, x_attr):
        n = base_network
        mi_df = _make_multiindex_df(x_attr, attr, ["c1"])
        n.add(comp, "c1", **base_kwargs, **{attr: mi_df})
        _assert_segment_stored(n.components[comp].segments[attr], ["c1"], x_attr, attr)

    def test_dict_broadcast_to_multiple_components(
        self, base_network, comp, base_kwargs, attr, x_attr
    ):
        """A dict is broadcast identically to all named components."""
        n = base_network
        n.add(comp, ["c1", "c2"], **base_kwargs, **{attr: CURVE_DICT})
        seg = n.components[comp].segments[attr]
        _assert_segment_stored(seg, ["c1", "c2"], x_attr, attr)


# ===========================================================================
# 2. Multi-port: Link efficiency2 / Process rate2
# ===========================================================================


@pytest.mark.parametrize(
    ("comp", "attr"), [("Link", "efficiency2"), ("Process", "rate2")]
)
class TestMultiportSegments:
    """Port-suffixed (port ≥ 2) segment attributes for Links and Processes."""

    @pytest.fixture
    def multiport_kwargs(self, comp) -> dict:
        return {**BASE_KWARGS[comp], "bus2": "bus_dc"}

    def test_dict_input(self, base_network, comp, attr, multiport_kwargs):
        n = base_network
        n.add(comp, "c1", **multiport_kwargs, **{attr: CURVE_DICT})
        _assert_segment_stored(n.components[comp].segments[attr], ["c1"], "p_pu", attr)

    def test_plain_df_input(self, base_network, comp, attr, multiport_kwargs):
        n = base_network
        n.add(comp, "c1", **multiport_kwargs, **{attr: _make_plain_df("p_pu", attr)})
        _assert_segment_stored(n.components[comp].segments[attr], ["c1"], "p_pu", attr)

    def test_multiindex_df_input(self, base_network, comp, attr, multiport_kwargs):
        n = base_network
        mi_df = _make_multiindex_df("p_pu", attr, ["c1"])
        n.add(comp, "c1", **multiport_kwargs, **{attr: mi_df})
        _assert_segment_stored(n.components[comp].segments[attr], ["c1"], "p_pu", attr)


# ===========================================================================
# 3. Error paths
# ===========================================================================


class TestSegmentErrors:
    def test_dict_for_non_segment_attr_raises(self, base_network):
        """A dict passed to a plain scalar attr raises ``NotImplementedError``."""
        n = base_network
        with pytest.raises(NotImplementedError, match="Dictionaries are not supported"):
            n.add("Generator", "gen", bus="bus_ac", p_nom={0: 100})

    @pytest.mark.parametrize(
        ("comp", "extendable_kwargs", "attr"), EXTENDABLE_PU_RAISES_PARAMS
    )
    def test_pu_segments_on_extendable_component_raises(
        self, base_network, comp, extendable_kwargs, attr
    ):
        """Per-unit piecewise attrs must be rejected when nominal capacity is extendable."""
        n = base_network
        with pytest.raises(ValueError, match="Piecewise"):
            n.add(comp, "c1", **extendable_kwargs, **{attr: CURVE_DICT})

    @pytest.mark.parametrize(
        ("comp", "extendable_kwargs", "attr"), EXTENDABLE_NOM_ALLOWED_PARAMS
    )
    def test_nom_segments_on_extendable_component_allowed(
        self, base_network, comp, extendable_kwargs, attr
    ):
        """Nom-based piecewise attrs (e.g. capital_cost vs s_nom) are always permitted."""
        n = base_network
        n.add(comp, "c1", **extendable_kwargs, **{attr: CURVE_DICT})
        assert not n.components[comp].segments[attr].empty
