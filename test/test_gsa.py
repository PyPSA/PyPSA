# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for Global Sensitivity Analysis (GSA) functionality."""

import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.collection import NetworkCollection
from pypsa.optimization.gsa import (
    _apply_parameter_sample,
    _convert_pypsa_to_salib,
    _parse_parameter_name,
    generate_gsa_samples,
)


class TestConvertPyPSAToSALib:
    """Tests for PyPSA to SALib conversion."""

    def test_simple_conversion(self):
        """Test basic conversion from PyPSA parameters to SALib format."""
        parameters = {
            "Generator": {"capital_cost": {"solar": (0.6, 1.0), "wind": (0.8, 1.0)}}
        }

        uncertainty = _convert_pypsa_to_salib(parameters, distributions="uniform")

        assert uncertainty["num_vars"] == 2
        assert "Generator-capital_cost-solar" in uncertainty["names"]
        assert "Generator-capital_cost-wind" in uncertainty["names"]
        assert uncertainty["bounds"] == [[0.6, 1.0], [0.8, 1.0]]
        assert uncertainty["dists"] == ["unif", "unif"]

    def test_multiple_components(self):
        """Test conversion with multiple component types."""
        parameters = {
            "Generator": {"capital_cost": {"solar": (0.6, 1.0)}},
            "StorageUnit": {"capital_cost": {"battery": (0.7, 0.9)}},
        }

        uncertainty = _convert_pypsa_to_salib(parameters)

        assert uncertainty["num_vars"] == 2
        assert len(uncertainty["names"]) == 2
        assert len(uncertainty["bounds"]) == 2

    def test_normal_distribution(self):
        """Test conversion with normal distribution."""
        parameters = {"Generator": {"marginal_cost": {"gas": (50, 100)}}}

        uncertainty = _convert_pypsa_to_salib(
            parameters, distributions={"Generator-marginal_cost-gas": "normal"}
        )

        assert uncertainty["dists"][0] == "norm"


class TestGenerateGSASamples:
    """Tests for generate_gsa_samples utility function."""

    def test_sobol_sampling(self):
        """Test Sobol sample generation."""
        parameters = {"Generator": {"capital_cost": {"solar": (0.6, 1.0)}}}

        uncertainty, samples = generate_gsa_samples(
            parameters, n_samples=16, method="sobol", seed=42
        )

        assert uncertainty["num_vars"] == 1
        # SALib returns N * (2^k + 2) samples where k is num_vars
        # With 1 variable: 16 * (2^1 + 2) = 16 * 4 = 64, but Saltelli might give different
        # Just check that we get samples with the right bounds and shape
        assert samples.shape[1] == 1
        assert samples.shape[0] >= 16  # At least the requested number
        assert 0.6 <= samples.min() <= 1.0
        assert 0.6 <= samples.max() <= 1.0

    def test_reproducibility(self):
        """Test that same seed produces same samples."""
        parameters = {"Generator": {"capital_cost": {"solar": (0.6, 1.0)}}}

        _, samples1 = generate_gsa_samples(
            parameters, n_samples=16, method="sobol", seed=42
        )
        _, samples2 = generate_gsa_samples(
            parameters, n_samples=16, method="sobol", seed=42
        )

        np.testing.assert_array_equal(samples1, samples2)

    def test_different_seeds(self):
        """Test that different seeds produce different samples."""
        parameters = {"Generator": {"capital_cost": {"solar": (0.6, 1.0)}}}

        _, samples1 = generate_gsa_samples(
            parameters, n_samples=16, method="sobol", seed=42
        )
        _, samples2 = generate_gsa_samples(
            parameters, n_samples=16, method="sobol", seed=43
        )

        # Different seeds should produce different samples
        assert not np.allclose(samples1, samples2)

    def test_unsupported_method(self):
        """Test that unsupported method raises error."""
        parameters = {"Generator": {"capital_cost": {"solar": (0.6, 1.0)}}}

        with pytest.raises(NotImplementedError, match="not yet supported"):
            generate_gsa_samples(parameters, n_samples=16, method="morris")


class TestParseParameterName:
    """Tests for parameter name parsing."""

    def test_full_name(self):
        """Test parsing full parameter name."""
        component, attr, name = _parse_parameter_name(
            "Generator-capital_cost-solar"
        )
        assert component == "Generator"
        assert attr == "capital_cost"
        assert name == "solar"

    def test_name_with_hyphens(self):
        """Test parsing parameter name containing hyphens."""
        component, attr, name = _parse_parameter_name(
            "StorageUnit-p_nom_max-battery-1"
        )
        assert component == "StorageUnit"
        assert attr == "p_nom_max"
        assert name == "battery-1"

    def test_component_only(self):
        """Test parsing component-attribute only."""
        component, attr, name = _parse_parameter_name("Generator-capital_cost")
        assert component == "Generator"
        assert attr == "capital_cost"
        assert name == ""


class TestApplyParameterSample:
    """Tests for applying parameter samples to networks."""

    def test_apply_to_specific_component(self):
        """Test applying parameter to specific component."""
        n = pypsa.examples.model_energy()
        original_cost = n.generators.loc["solar", "capital_cost"]

        uncertainty = {"names": ["Generator-capital_cost-solar"]}
        sample = np.array([0.8])

        _apply_parameter_sample(n, sample, uncertainty)

        assert n.generators.loc["solar", "capital_cost"] == original_cost * 0.8

    def test_apply_multiple_parameters(self):
        """Test applying multiple parameters."""
        n = pypsa.examples.model_energy()
        original_solar = n.generators.loc["solar", "capital_cost"]
        original_wind = n.generators.loc["wind", "capital_cost"]

        uncertainty = {
            "names": [
                "Generator-capital_cost-solar",
                "Generator-capital_cost-wind",
            ]
        }
        sample = np.array([0.8, 1.2])

        _apply_parameter_sample(n, sample, uncertainty)

        assert n.generators.loc["solar", "capital_cost"] == original_solar * 0.8
        assert n.generators.loc["wind", "capital_cost"] == original_wind * 1.2


class TestOptimizeGSA:
    """Tests for optimize_gsa method."""

    def test_optimize_gsa_serial(self):
        """Test GSA optimization with serial execution."""
        n = pypsa.examples.model_energy()

        # Use a small subset of snapshots for faster testing
        n.set_snapshots(n.snapshots[:24])

        uncertainty, samples = generate_gsa_samples(
            parameters={"Generator": {"capital_cost": {"solar": (0.8, 1.2)}}},
            n_samples=4,
            method="sobol",
            seed=42,
        )

        nc = n.optimize.optimize_gsa(
            samples=samples,
            uncertainty=uncertainty,
            max_parallel=1,
        )

        assert isinstance(nc, NetworkCollection)
        # SALib Sobol returns N * (2^k + 2) samples, so we get more than requested
        assert len(nc) >= 4
        assert len(nc) == samples.shape[0]  # Should match number of samples generated

    def test_optimize_gsa_parallel(self):
        """Test GSA optimization with parallel execution."""
        n = pypsa.examples.model_energy()
        n.set_snapshots(n.snapshots[:24])

        uncertainty, samples = generate_gsa_samples(
            parameters={"Generator": {"capital_cost": {"solar": (0.8, 1.2)}}},
            n_samples=4,
            method="sobol",
            seed=42,
        )

        nc = n.optimize.optimize_gsa(
            samples=samples,
            uncertainty=uncertainty,
            max_parallel=2,
        )

        assert isinstance(nc, NetworkCollection)
        assert len(nc) >= 4
        assert len(nc) == samples.shape[0]

    def test_optimize_gsa_multiple_parameters(self):
        """Test GSA with multiple uncertain parameters."""
        n = pypsa.examples.model_energy()
        n.set_snapshots(n.snapshots[:24])

        uncertainty, samples = generate_gsa_samples(
            parameters={
                "Generator": {
                    "capital_cost": {
                        "solar": (0.8, 1.2),
                        "wind": (0.8, 1.2),
                    }
                }
            },
            n_samples=4,
            method="sobol",
            seed=42,
        )

        nc = n.optimize.optimize_gsa(
            samples=samples,
            uncertainty=uncertainty,
            max_parallel=2,
        )

        assert len(nc) >= 4
        # Check that networks have different solar and wind costs
        costs_1 = nc.networks[0].generators.loc["solar", "capital_cost"]
        costs_2 = nc.networks[1].generators.loc["solar", "capital_cost"]
        assert costs_1 != costs_2


class TestNetworkCollectionAnalyzeSobol:
    """Tests for NetworkCollection.analyze_sobol method."""

    def test_analyze_sobol_single_output(self):
        """Test Sobol analysis with single output."""
        n = pypsa.examples.model_energy()
        n.set_snapshots(n.snapshots[:24])

        uncertainty, samples = generate_gsa_samples(
            parameters={
                "Generator": {
                    "capital_cost": {
                        "solar": (0.8, 1.2),
                        "wind": (0.8, 1.2),
                    }
                }
            },
            n_samples=64,
            method="sobol",
            seed=42,
        )

        nc = n.optimize.optimize_gsa(
            samples=samples,
            uncertainty=uncertainty,
            max_parallel=2,
        )

        capex = nc.statistics.capex()

        sobol_indices = nc.analyze_sobol(uncertainty, {"capex": capex})

        assert "capex" in sobol_indices
        assert isinstance(sobol_indices["capex"], pd.DataFrame)
        assert "S1" in sobol_indices["capex"].columns
        assert "S1_conf" in sobol_indices["capex"].columns
        assert "ST" in sobol_indices["capex"].columns
        assert "ST_conf" in sobol_indices["capex"].columns
        assert len(sobol_indices["capex"]) == 2  # Two parameters

    def test_analyze_sobol_multiple_outputs(self):
        """Test Sobol analysis with multiple outputs."""
        n = pypsa.examples.model_energy()
        n.set_snapshots(n.snapshots[:24])

        uncertainty, samples = generate_gsa_samples(
            parameters={
                "Generator": {
                    "capital_cost": {
                        "solar": (0.8, 1.2),
                        "wind": (0.8, 1.2),
                    }
                }
            },
            n_samples=64,
            method="sobol",
            seed=42,
        )

        nc = n.optimize.optimize_gsa(
            samples=samples,
            uncertainty=uncertainty,
            max_parallel=2,
        )

        capex = nc.statistics.capex()
        opex = nc.statistics.opex()

        sobol_indices = nc.analyze_sobol(
            uncertainty, {"capex": capex, "opex": opex}
        )

        assert len(sobol_indices) == 2
        assert "capex" in sobol_indices
        assert "opex" in sobol_indices

    def test_analyze_sobol_series_input(self):
        """Test that Series input is handled correctly."""
        n = pypsa.examples.model_energy()
        n.set_snapshots(n.snapshots[:24])

        uncertainty, samples = generate_gsa_samples(
            parameters={"Generator": {"capital_cost": {"solar": (0.8, 1.2)}}},
            n_samples=64,
            method="sobol",
            seed=42,
        )

        nc = n.optimize.optimize_gsa(
            samples=samples,
            uncertainty=uncertainty,
            max_parallel=2,
        )

        # Get statistics as Series
        capex = nc.statistics.capex()
        assert isinstance(capex, pd.Series)

        # Should handle Series input
        sobol_indices = nc.analyze_sobol(uncertainty, {"capex": capex})
        assert "capex" in sobol_indices

    def test_analyze_sobol_array_input(self):
        """Test that numpy array input is handled correctly."""
        n = pypsa.examples.model_energy()
        n.set_snapshots(n.snapshots[:24])

        uncertainty, samples = generate_gsa_samples(
            parameters={"Generator": {"capital_cost": {"solar": (0.8, 1.2)}}},
            n_samples=64,
            method="sobol",
            seed=42,
        )

        nc = n.optimize.optimize_gsa(
            samples=samples,
            uncertainty=uncertainty,
            max_parallel=2,
        )

        capex = nc.statistics.capex()
        capex_array = capex.values

        sobol_indices = nc.analyze_sobol(uncertainty, {"capex": capex_array})
        assert "capex" in sobol_indices
