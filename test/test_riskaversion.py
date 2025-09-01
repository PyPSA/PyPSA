import pytest

import pypsa


def test_set_risk_preference_requires_scenarios():
    n = pypsa.Network()
    # No scenarios defined: should raise
    with pytest.raises(RuntimeError, match=r"set_scenarios\(\)"):
        n.set_risk_preference(alpha=0.1, omega=0.5)


def test_set_and_get_risk_preference_dict_and_flags():
    n = pypsa.Network()
    # Define scenarios
    n.set_scenarios(["s1", "s2"])  # equal weights 0.5/0.5

    # Initially, should not have risk preference
    assert n.has_risk_preference is False
    assert n.risk_preference is None

    # Set valid preferences
    n.set_risk_preference(alpha=0.2, omega=0.7)

    # Getter returns dict
    rp = n.risk_preference
    assert isinstance(rp, dict)
    assert pytest.approx(rp["alpha"]) == 0.2
    assert pytest.approx(rp["omega"]) == 0.7

    # Boolean flag flips to True
    assert n.has_risk_preference is True


@pytest.mark.parametrize(
    ("alpha", "omega", "err", "pattern"),
    [
        (-0.1, 0.5, ValueError, "Alpha must be between 0 and 1"),
        (1.0, 0.5, ValueError, "Alpha must be between 0 and 1"),
        (0.2, -0.01, ValueError, "Omega must be between 0 and 1"),
        (0.2, 1.01, ValueError, "Omega must be between 0 and 1"),
    ],
)
def test_set_risk_preference_validation(alpha, omega, err, pattern):
    n = pypsa.Network()
    n.set_scenarios({"a": 0.4, "b": 0.6})
    with pytest.raises(err, match=pattern):
        n.set_risk_preference(alpha=alpha, omega=omega)
