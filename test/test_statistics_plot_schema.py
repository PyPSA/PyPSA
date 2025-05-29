from pypsa.plot.statistics.schema import apply_parameter_schema


class TestSchemaContext:
    """Test that schema functions work with context parameter."""

    def test_apply_parameter_schema_with_context(self):
        """Test apply_parameter_schema accepts context parameter."""
        # Test that function accepts context parameter
        kwargs = {"x": None, "y": None, "color": None}
        context = {"index_names": ["scenario"]}

        # This should not raise an error
        result = apply_parameter_schema("installed_capacity", "bar", kwargs, context)
        assert isinstance(result, dict)

    def test_apply_parameter_schema_backward_compatible(self):
        """Test apply_parameter_schema works without context parameter."""
        from pypsa.plot.statistics.schema import apply_parameter_schema

        # Test backward compatibility
        kwargs = {"x": None, "y": None, "color": None}

        # This should not raise an error
        result = apply_parameter_schema("installed_capacity", "bar", kwargs)
        assert isinstance(result, dict)
