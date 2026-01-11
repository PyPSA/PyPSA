# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for capital cost split feature (Issue 371).

Tests for the separation of capital_cost into investment costs and fom.
"""

import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.costs import annuity, annuity_factor, periodized_cost

TOLERANCE = 1e-5
SOLVER_NAME = "highs"


@pytest.fixture
def simple_network():
    """Create a simple network for testing."""
    n = pypsa.Network()
    n.add("Bus", "bus")
    n.set_snapshots([0, 1, 2])
    return n


class TestCostsModule:
    """Tests for pypsa.costs module."""

    def test_annuity_basic(self):
        """Test basic annuity calculation."""
        # 7% discount rate, 25 years
        result = annuity(0.07, 25)
        expected = 0.07 / (1 - 1 / 1.07**25)
        assert abs(result - expected) < TOLERANCE

    def test_annuity_zero_discount_rate(self):
        """Test annuity with 0% discount rate returns 1/lifetime."""
        result = annuity(0.0, 20)
        expected = 1.0 / 20  # Simple depreciation
        assert abs(result - expected) < TOLERANCE

    def test_annuity_series(self):
        """Test annuity with pandas Series."""
        rates = pd.Series([0.05, 0.07, 0.10])
        lifetimes = pd.Series([20, 25, 30])
        result = annuity(rates, lifetimes)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_annuity_series_with_zero_rate(self):
        """Test annuity Series with zero discount rate."""
        rates = pd.Series([0.0, 0.07, 0.0])
        lifetimes = pd.Series([20, 25, 30])
        result = annuity(rates, lifetimes)
        assert abs(result.iloc[0] - 1.0 / 20) < TOLERANCE
        assert abs(result.iloc[2] - 1.0 / 30) < TOLERANCE

    def test_annuity_infinite_lifetime(self):
        """Test annuity with infinite lifetime returns discount rate."""
        result = annuity(0.07, float("inf"))
        assert result == 0.07

    def test_annuity_zero_rate_infinite_lifetime(self):
        """Test annuity with 0% rate and infinite lifetime returns 0."""
        result = annuity(0.0, float("inf"))
        assert result == 0.0

    def test_annuity_negative_discount_rate(self):
        """Test annuity with negative discount rate (penalizes present)."""
        result = annuity(-0.02, 20)
        # Negative rates are allowed and produce valid results
        assert result > 0

    def test_annuity_negative_rate_infinite_lifetime(self):
        """Test annuity with negative rate and infinite lifetime returns 0."""
        result = annuity(-0.05, float("inf"))
        assert result == 0.0

    def test_annuity_non_positive_lifetime_raises(self):
        """Test that non-positive lifetime raises ValueError."""
        with pytest.raises(ValueError, match="lifetime must be positive"):
            annuity(0.07, 0)
        with pytest.raises(ValueError, match="lifetime must be positive"):
            annuity(0.07, -10)

    def test_annuity_factor_nan_discount_rate(self):
        """Test annuity_factor returns 1.0 for NaN discount rate."""
        result = annuity_factor(np.nan, 25)
        assert result == 1.0

    def test_annuity_factor_positive_discount_rate(self):
        """Test annuity_factor returns correct value for positive discount rate."""
        result = annuity_factor(0.07, 25)
        expected = annuity(0.07, 25)
        assert abs(result - expected) < TOLERANCE

    def test_annuity_factor_zero_discount_rate(self):
        """Test annuity_factor with 0% rate returns 1/lifetime."""
        result = annuity_factor(0.0, 20)
        expected = 1.0 / 20
        assert abs(result - expected) < TOLERANCE

    def test_periodized_cost_with_overnight(self):
        """Test periodized cost with overnight cost."""
        overnight = 1000  # EUR/kW overnight
        discount_rate = 0.07
        lifetime = 25
        fom = 20  # EUR/kW/year

        result = periodized_cost(
            capital_cost=0,
            overnight_cost=overnight,
            discount_rate=discount_rate,
            lifetime=lifetime,
            fom_cost=fom,
            nyears=0.5,
        )
        expected = overnight * annuity(discount_rate, lifetime) * 0.5 + fom
        assert abs(result - expected) < TOLERANCE

    def test_periodized_cost_with_capital_cost(self):
        """Test periodized cost using capital_cost (overnight is NaN)."""
        capital = 100  # pre-annuitized
        fom = 20

        result = periodized_cost(
            capital_cost=capital,
            overnight_cost=np.nan,
            discount_rate=np.nan,
            lifetime=25,
            fom_cost=fom,
            nyears=0.5,
        )
        expected = capital + fom  # capital_cost used directly
        assert abs(result - expected) < TOLERANCE

    def test_periodized_cost_zero_discount(self):
        """Test periodized cost with 0% discount rate."""
        overnight = 1000
        lifetime = 20

        result = periodized_cost(
            capital_cost=0,
            overnight_cost=overnight,
            discount_rate=0.0,
            lifetime=lifetime,
            fom_cost=0,
            nyears=2.0,
        )
        expected = overnight / lifetime * 2.0  # Simple depreciation
        assert abs(result - expected) < TOLERANCE


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing capital_cost usage."""

    def test_default_values(self, simple_network):
        """Test that default values preserve current behavior."""
        n = simple_network
        n.add(
            "Generator",
            "gen",
            bus="bus",
            p_nom_extendable=True,
            capital_cost=100,  # Pre-annuitized
            marginal_cost=10,
        )

        # overnight_cost and discount_rate should default to NaN
        assert np.isnan(n.c.generators.static.overnight_cost.iloc[0])
        assert np.isnan(n.c.generators.static.discount_rate.iloc[0])
        assert n.c.generators.static.fom_cost.iloc[0] == 0

    def test_optimization_with_defaults(self, simple_network):
        """Test that optimization works with default (capital_cost only)."""
        n = simple_network
        n.add(
            "Generator",
            "gen",
            bus="bus",
            p_nom_extendable=True,
            capital_cost=100,
            marginal_cost=10,
        )
        n.add("Load", "load", bus="bus", p_set=50)

        n.optimize(solver_name=SOLVER_NAME)

        # Should run without errors
        assert n.c.generators.static.p_nom_opt.iloc[0] >= 0


class TestOvernightCostWithDiscountRate:
    """Tests for overnight cost with discount rate."""

    def test_annuity_applied(self, simple_network):
        """Test that annuity is applied when overnight_cost is provided."""
        n = simple_network
        overnight_cost = 1000  # EUR/kW overnight
        discount_rate = 0.07
        lifetime = 25

        n.add(
            "Generator",
            "gen",
            bus="bus",
            p_nom_extendable=True,
            overnight_cost=overnight_cost,
            discount_rate=discount_rate,
            lifetime=lifetime,
            marginal_cost=10,
        )
        n.add("Load", "load", bus="bus", p_set=50)

        n.optimize(solver_name=SOLVER_NAME)

        # Verify generator is built
        assert n.c.generators.static.p_nom_opt.iloc[0] > 0

        # The objective should use annuitized cost
        ann_factor = annuity(discount_rate, lifetime)
        expected_annual_cost = overnight_cost * ann_factor * n.nyears
        # Verify via statistics
        investment = n.statistics.investment(groupby=False)
        capacity = n.c.generators.static.p_nom_opt.iloc[0]
        actual = investment.values.sum()
        assert abs(actual - capacity * expected_annual_cost) < TOLERANCE

    def test_zero_discount_rate(self, simple_network):
        """Test that 0% discount rate works correctly (simple depreciation)."""
        n = simple_network
        overnight_cost = 1000
        discount_rate = 0.0  # Valid: simple depreciation
        lifetime = 20

        n.add(
            "Generator",
            "gen",
            bus="bus",
            p_nom_extendable=True,
            overnight_cost=overnight_cost,
            discount_rate=discount_rate,
            lifetime=lifetime,
            marginal_cost=10,
        )
        n.add("Load", "load", bus="bus", p_set=50)

        n.optimize(solver_name=SOLVER_NAME)

        # Verify generator is built
        capacity = n.c.generators.static.p_nom_opt.iloc[0]
        assert capacity > 0

        # 0% rate means simple depreciation: 1000/20 = 50 per year
        expected_annual_cost = overnight_cost / lifetime * n.nyears
        investment = n.statistics.investment(groupby=False)
        actual = investment.values.sum()
        assert abs(actual - capacity * expected_annual_cost) < TOLERANCE

    def test_fom_added_separately(self, simple_network):
        """Test that fom_cost is added to annuitized investment cost."""
        n = simple_network
        overnight_cost = 1000
        discount_rate = 0.07
        lifetime = 25
        fom_cost = 20

        n.add(
            "Generator",
            "gen",
            bus="bus",
            p_nom_extendable=True,
            overnight_cost=overnight_cost,
            discount_rate=discount_rate,
            lifetime=lifetime,
            fom_cost=fom_cost,
            marginal_cost=10,
        )
        n.add("Load", "load", bus="bus", p_set=50)

        n.optimize(solver_name=SOLVER_NAME)

        # Verify generator is built
        assert n.c.generators.static.p_nom_opt.iloc[0] > 0


class TestStatisticsFunctions:
    """Tests for new statistics functions."""

    def test_investment_function_exists(self, simple_network):
        """Test that investment() function exists and is callable."""
        n = simple_network
        n.add(
            "Generator",
            "gen",
            bus="bus",
            p_nom_extendable=True,
            capital_cost=100,
            marginal_cost=10,
        )
        n.add("Load", "load", bus="bus", p_set=50)
        n.optimize(solver_name=SOLVER_NAME)

        # Should not raise an error
        result = n.statistics.investment()
        assert isinstance(result, pd.DataFrame | pd.Series)

    def test_fom_function_exists(self, simple_network):
        """Test that fom() function exists and is callable."""
        n = simple_network
        n.add(
            "Generator",
            "gen",
            bus="bus",
            p_nom_extendable=True,
            capital_cost=100,
            fom_cost=20,
            marginal_cost=10,
        )
        n.add("Load", "load", bus="bus", p_set=50)
        n.optimize(solver_name=SOLVER_NAME)

        result = n.statistics.fom()
        assert isinstance(result, pd.DataFrame | pd.Series)

    def test_investment_with_overnight_cost(self, simple_network):
        """Test investment() returns correct value with overnight cost."""
        n = simple_network
        overnight_cost = 1000
        discount_rate = 0.07
        lifetime = 25

        n.add(
            "Generator",
            "gen",
            bus="bus",
            p_nom_extendable=True,
            overnight_cost=overnight_cost,
            discount_rate=discount_rate,
            lifetime=lifetime,
            marginal_cost=10,
        )
        n.add("Load", "load", bus="bus", p_set=50)
        n.optimize(solver_name=SOLVER_NAME)

        investment = n.statistics.investment(groupby=False)
        capacity = n.c.generators.static.p_nom_opt.iloc[0]
        expected = (
            capacity * overnight_cost * annuity(discount_rate, lifetime) * n.nyears
        )

        # Check the investment cost is calculated correctly
        assert not investment.empty
        actual = investment.values.sum()
        assert abs(actual - expected) < TOLERANCE

    def test_investment_with_capital_cost(self, simple_network):
        """Test investment() returns capital_cost when overnight is not set."""
        n = simple_network
        capital_cost = 100

        n.add(
            "Generator",
            "gen",
            bus="bus",
            p_nom_extendable=True,
            capital_cost=capital_cost,
            marginal_cost=10,
        )
        n.add("Load", "load", bus="bus", p_set=50)
        n.optimize(solver_name=SOLVER_NAME)

        investment = n.statistics.investment(groupby=False)
        capacity = n.c.generators.static.p_nom_opt.iloc[0]
        expected = capacity * capital_cost

        assert not investment.empty
        actual = investment.values.sum()
        assert abs(actual - expected) < TOLERANCE

    def test_fom_calculation(self, simple_network):
        """Test fom() returns correct value."""
        n = simple_network
        fom_cost_val = 20

        n.add(
            "Generator",
            "gen",
            bus="bus",
            p_nom_extendable=True,
            capital_cost=100,
            fom_cost=fom_cost_val,
            marginal_cost=10,
        )
        n.add("Load", "load", bus="bus", p_set=50)
        n.optimize(solver_name=SOLVER_NAME)

        fom_result = n.statistics.fom(groupby=False)
        capacity = n.c.generators.static.p_nom_opt.iloc[0]
        expected = capacity * fom_cost_val

        # Check the fom cost is calculated correctly
        assert not fom_result.empty
        actual = fom_result.values.sum()
        assert abs(actual - expected) < TOLERANCE


class TestMultiComponentScenarios:
    """Test cost split with multiple component types."""

    def test_multiple_generators(self, simple_network):
        """Test with multiple generators having different cost structures."""
        n = simple_network

        # Generator 1: overnight cost with discount rate
        n.add(
            "Generator",
            "gen1",
            bus="bus",
            p_nom_extendable=True,
            overnight_cost=1000,
            discount_rate=0.07,
            lifetime=25,
            fom_cost=20,
            marginal_cost=10,
        )

        # Generator 2: pre-annuitized (default behavior)
        n.add(
            "Generator",
            "gen2",
            bus="bus",
            p_nom_extendable=True,
            capital_cost=100,  # Already annuitized
            marginal_cost=15,
        )

        n.add("Load", "load", bus="bus", p_set=100)
        n.optimize(solver_name=SOLVER_NAME)

        # Both generators should be considered
        assert n.c.generators.static.p_nom_opt.sum() > 0

    def test_links_with_overnight_cost(self, simple_network):
        """Test overnight cost works for Links."""
        n = simple_network
        n.add("Bus", "bus2")

        n.add(
            "Link",
            "link",
            bus0="bus",
            bus1="bus2",
            p_nom_extendable=True,
            overnight_cost=500,
            discount_rate=0.05,
            lifetime=40,
            fom_cost=10,
        )

        # Verify attributes exist
        assert n.c.links.static.overnight_cost.iloc[0] == 500
        assert n.c.links.static.discount_rate.iloc[0] == 0.05
        assert n.c.links.static.fom_cost.iloc[0] == 10
        assert n.c.links.static.lifetime.iloc[0] == 40


class TestDeprecation:
    """Test deprecation of pypsa.common.annuity."""

    def test_common_annuity_deprecated(self):
        """Test that pypsa.common.annuity issues deprecation warning."""
        with pytest.warns(DeprecationWarning, match="annuity"):
            pypsa.common.annuity(0.07, 25)

    def test_costs_annuity_no_warning(self):
        """Test that pypsa.costs.annuity does not issue warning."""
        # This should not raise any warnings
        result = pypsa.costs.annuity(0.07, 25)
        assert result > 0


class TestComponentProperties:
    """Test new component properties for cost calculations."""

    def test_investment_cost_with_overnight(self, simple_network):
        """Test investment_cost property with overnight_cost."""
        n = simple_network
        overnight_cost = 1000
        discount_rate = 0.07
        lifetime = 25

        n.add(
            "Generator",
            "gen",
            bus="bus",
            overnight_cost=overnight_cost,
            discount_rate=discount_rate,
            lifetime=lifetime,
        )

        inv_cost = n.c.generators.investment_cost.iloc[0]
        expected = overnight_cost * annuity(discount_rate, lifetime) * n.nyears
        assert abs(inv_cost - expected) < TOLERANCE

    def test_investment_cost_with_capital(self, simple_network):
        """Test investment_cost property with capital_cost only."""
        n = simple_network
        capital_cost = 100

        n.add("Generator", "gen", bus="bus", capital_cost=capital_cost)

        inv_cost = n.c.generators.investment_cost.iloc[0]
        assert abs(inv_cost - capital_cost) < TOLERANCE

    def test_annuity_property(self, simple_network):
        """Test annuity property returns correct factor."""
        n = simple_network
        discount_rate = 0.07
        lifetime = 25

        n.add(
            "Generator",
            "gen",
            bus="bus",
            overnight_cost=1000,
            discount_rate=discount_rate,
            lifetime=lifetime,
        )

        ann = n.c.generators.annuity.iloc[0]
        expected = annuity(discount_rate, lifetime)
        assert abs(ann - expected) < TOLERANCE

    def test_annuity_property_nan_rate(self, simple_network):
        """Test annuity property returns 1.0 for NaN discount_rate."""
        n = simple_network
        n.add("Generator", "gen", bus="bus", capital_cost=100)

        ann = n.c.generators.annuity.iloc[0]
        assert ann == 1.0


class TestConsistencyCheck:
    """Test consistency check for cost attributes."""

    def test_warns_when_both_costs_set(self, simple_network, caplog):
        """Test that a warning is logged when both overnight and capital cost are set."""
        import logging

        n = simple_network
        n.add(
            "Generator",
            "gen",
            bus="bus",
            overnight_cost=1000,
            capital_cost=100,  # Both set - should warn
            discount_rate=0.07,
            lifetime=25,
        )

        with caplog.at_level(logging.WARNING):
            n.consistency_check()

        assert any("overnight_cost" in record.message for record in caplog.records)
        assert any("capital_cost" in record.message for record in caplog.records)

    def test_no_warning_overnight_only(self, simple_network, caplog):
        """Test no warning when only overnight_cost is set."""
        import logging

        n = simple_network
        n.add(
            "Generator",
            "gen",
            bus="bus",
            overnight_cost=1000,
            discount_rate=0.07,
            lifetime=25,
        )

        with caplog.at_level(logging.WARNING):
            n.consistency_check()

        # Should not warn about cost consistency
        assert not any(
            "overnight_cost" in record.message and "capital_cost" in record.message
            for record in caplog.records
        )

    def test_no_warning_capital_only(self, simple_network, caplog):
        """Test no warning when only capital_cost is set."""
        import logging

        n = simple_network
        n.add(
            "Generator",
            "gen",
            bus="bus",
            capital_cost=100,
        )

        with caplog.at_level(logging.WARNING):
            n.consistency_check()

        # Should not warn about cost consistency
        assert not any(
            "overnight_cost" in record.message and "capital_cost" in record.message
            for record in caplog.records
        )
