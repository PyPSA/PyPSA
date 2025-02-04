import pytest

import pypsa


def test_getter():
    # Default init and get
    pypsa.options.warnings.components_store_iter = True
    assert pypsa.options.warnings.components_store_iter is True
    with pytest.raises(
        AttributeError,
        match="Invalid option 'invalid_option' for this category. Valid options are: *+",
    ):
        pypsa.options.warnings.invalid_option
    with pytest.raises(
        AttributeError,
        match="Invalid category 'invalid_category'. Valid categories are: *+",
    ):
        pypsa.options.invalid_category.invalid_option


def test_setter():
    pypsa.options.warnings.components_store_iter = False
    assert pypsa.options.warnings.components_store_iter is False
    with pytest.raises(
        AttributeError,
        match="Invalid option 'invalid_option' for this category. Valid options are: *+",
    ):
        pypsa.options.warnings.invalid_option = False
    with pytest.raises(
        AttributeError,
        match="Invalid category 'invalid_category'. Valid categories are: *+",
    ):
        pypsa.options.invalid_category.invalid_option = False


def test_getter_method():
    pypsa.options.warnings.components_store_iter = True
    assert pypsa.get_option("warnings.components_store_iter") is True
    pypsa.options.warnings.components_store_iter = False
    assert pypsa.get_option("warnings.components_store_iter") is False

    with pytest.raises(
        AttributeError,
        match="Invalid option 'warnings.invalid_option'. Check options via pypsa.options.describe_options()",
    ):
        pypsa.get_option("warnings.invalid_option")
    with pytest.raises(
        AttributeError,
        match="Invalid option 'invalid_warning.invalid_option'. Check options via pypsa.options.describe_options()",
    ):
        pypsa.get_option("invalid_warning.invalid_option")


def test_setter_method():
    pypsa.set_option("warnings.components_store_iter", False)
    assert pypsa.options.warnings.components_store_iter is False
    assert pypsa.get_option("warnings.components_store_iter") is False

    with pytest.raises(
        AttributeError,
        match="Invalid option 'warnings.invalid_option'. Check options via pypsa.options.describe_options()",
    ):
        pypsa.set_option("warnings.invalid_option", False)

    with pytest.raises(
        AttributeError,
        match="Invalid option 'invalid_warning.invalid_option'. Check options via pypsa.options.describe_options()",
    ):
        pypsa.set_option("invalid_warning.invalid_option", False)


def test_describe_method():
    pypsa.describe_options()
