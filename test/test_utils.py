import pytest

from pypsa.utils import list_as_string


def test_list_as_string():
    # Test comma-separated (default)
    assert list_as_string(["a", "b", "c"]) == "a, b, c"
    # Test bullet-list
    expected_bullet = "  - x\n  - y\n  - z"
    assert (
        list_as_string(["x", "y", "z"], prefix="  ", style="bullet-list")
        == expected_bullet
    )
    # Test dict input
    assert list_as_string({"a": 1, "b": 2, "c": 3}) == "a, b, c"
    # Test empty list
    assert list_as_string([]) == ""
    # Test single item
    assert list_as_string(["a"]) == "a"
    # Test invalid style
    with pytest.raises(ValueError):
        list_as_string(["a", "b"], style="invalid")
    # Test prefix
    assert list_as_string(["a", "b"], prefix="-> ") == "-> a, b"
    # Test empty lists
    assert list_as_string([]) == ""
    assert list_as_string([], style="bullet-list") == ""
