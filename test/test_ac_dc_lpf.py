import pypsa
import os
import pytest
from numpy.testing import assert_array_almost_equal as equal

@pytest.fixture
def n():
    csv_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ac-dc-meshed",
        "ac-dc-data"
    )
    return pypsa.Network(csv_folder)


@pytest.fixture
def n_r():
    csv_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ac-dc-meshed",
        "ac-dc-data",
        "results-lpf"
    )
    return pypsa.Network(csv_folder)


def test_lpf(n, n_r):

    n.lpf(snapshots=n.snapshots)

    equal(
        n.generators_t.p[n.generators.index].iloc[:2],
        n_r.generators_t.p[n.generators.index].iloc[:2]
    )
    equal(
        n.lines_t.p0[n.lines.index].iloc[:2],
        n_r.lines_t.p0[n.lines.index].iloc[:2]
    )
    equal(
        n.links_t.p0[n.links.index].iloc[:2],
        n_r.links_t.p0[n.links.index].iloc[:2]
    )


def test_lpf_chunks(n, n_r):

    for snapshot in n.snapshots[:2]:
        n.lpf(snapshot)

    n.lpf(snapshots=n.snapshots)

    equal(
        n.generators_t.p[n.generators.index],n_r.generators_t.p[n.generators.index])
    equal(
        n.lines_t.p0[n.lines.index],
        n_r.lines_t.p0[n.lines.index]
    )
    equal(
        n.links_t.p0[n.links.index],
        n_r.links_t.p0[n.links.index]
    )

