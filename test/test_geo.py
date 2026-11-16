# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pypsa.geo import haversine, haversine_pts

EARTH_RADIUS = 6371.0  # km
DEGREE = EARTH_RADIUS * np.pi / 180  # km per degree of a great circle


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [
        ([10, 50], [10, 50], 0.0),
        ([0, 0], [1, 0], DEGREE),  # along the equator
        ([0, 0], [0, 1], DEGREE),  # along a meridian
        ([179.5, 0], [-179.5, 0], DEGREE),  # across the date line
        ([0, 90], [0, -90], 180 * DEGREE),  # pole to pole
        ([0, 0], [180, 0], 180 * DEGREE),  # antipodes
    ],
)
def test_haversine_pts_known_distances(a, b, distance):
    """Great-circle distances on a sphere with the mean Earth radius are symmetric."""
    assert_allclose(haversine_pts([a], [b]), [distance], atol=1e-9)
    assert_allclose(haversine_pts([b], [a]), [distance], atol=1e-9)


def test_haversine_pts_longitude_latitude_order():
    """A degree of longitude shrinks with the cosine of latitude, a degree of latitude does not."""
    d_lon, d_lat = haversine_pts([[0, 60], [0, 60]], [[1, 60], [0, 61]])
    assert d_lon == pytest.approx(DEGREE * np.cos(np.deg2rad(60)), rel=1e-4)
    assert d_lat == pytest.approx(DEGREE)


def test_haversine_matrix_matches_pairwise():
    """Entry (i, j) is the distance from a[i] to b[j]."""
    a = [[10.1, 52.6], [10.8, 52.1]]
    b = [[-34.0, 56.0], [0.0, 0.0], [13.4, 52.5]]
    expected = [[haversine_pts([p], [q])[0] for q in b] for p in a]

    assert_allclose(haversine(a, b), expected)
    assert_allclose(haversine(a[0], b), expected[:1])


def test_haversine_wrong_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        haversine([1.0, 2.0, 3.0], [[0.0, 0.0]])
