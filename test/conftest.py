# static command line options from
# https://docs.pytest.org/en/latest/example/simple.html

# content of conftest.py
import pytest

def pytest_addoption(parser):
    parser.addoption("--solver_name", action="store", default="glpk",
        help="insert the solver name")

@pytest.fixture
def solver_name(request):
    return request.config.getoption("--solver_name")
