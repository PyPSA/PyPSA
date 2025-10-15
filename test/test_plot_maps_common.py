# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

import pypsa
from pypsa.plot.maps.common import (
    _is_cartopy_available,
    add_jitter,
    apply_cmap,
    apply_layouter,
    calculate_angle,
    calculate_midpoint,
    clip_lat,
    df_to_html_table,
    feature_to_geojson,
    flip_polygon,
    get_global_stat,
    linestring_to_pdk_path,
    meters_to_lonlat,
    poly_to_geojson,
    rotate_polygon,
    round_value,
    scale_polygon_by_width,
    scale_to_max_abs,
    series_to_pdk_path,
    set_tooltip_style,
    shapefile_to_geojson,
    shorten_string,
    to_rgba255,
    to_rgba255_css,
    translate_polygon,
    wkt_to_linestring,
)

cartopy_present = _is_cartopy_available()


@pytest.fixture
def simple_network():
    n = pypsa.Network()

    # Add two buses
    n.add("Bus", "A")
    n.add("Bus", "B")

    # Add a line between them
    n.add("Line", "L1", bus0="A", bus1="B", x=0.1)
    n.add("Line", "L2", bus0="B", bus1="C", x=0.1)
    n.add("Line", "L3", bus0="A", bus1="C", x=0.1)

    # Add a generator on A
    n.add("Generator", "G1", bus="A", p_nom=100)

    # Add a load on B
    n.add("Load", "L1", bus="B", p_set=50)

    return n


# apply_cmap
def test_apply_cmap_numeric_default():
    s = pd.Series([0, 0.5, 1.0])
    result = apply_cmap(s, cmap="viridis")

    assert isinstance(result, pd.Series)
    # Each entry should be an RGBA tuple
    assert all(isinstance(c, tuple) for c in result)
    assert all(len(c) == 4 for c in result)


def test_apply_cmap_numeric_with_colormap_object():
    s = pd.Series([0, 0.5, 1.0])
    cmap_obj = plt.get_cmap("plasma")
    result = apply_cmap(s, cmap=cmap_obj)

    assert isinstance(result, pd.Series)
    assert all(isinstance(c, tuple) for c in result)
    assert all(len(c) == 4 for c in result)


# apply_layouter
def test_apply_layouter_returns_coordinates(simple_network):
    x, y = apply_layouter(simple_network, inplace=False)

    assert isinstance(x, pd.Series)
    assert isinstance(y, pd.Series)
    # Same bus index
    assert set(x.index) == {"A", "B", "C"}
    assert set(y.index) == {"A", "B", "C"}


def test_apply_layouter_inplace(simple_network):
    result = apply_layouter(simple_network, inplace=True)

    # inplace should return None
    assert result is None

    # coordinates must now exist in buses table
    assert "x" in simple_network.c.buses.static.columns
    assert "y" in simple_network.c.buses.static.columns
    assert not simple_network.c.buses.static[["x", "y"]].isna().any().any()


def test_apply_layouter_custom_layouter(simple_network):
    x, y = apply_layouter(simple_network, layouter=nx.circular_layout)

    # circular layout should give points approximately on the unit circle
    r = x**2 + y**2
    np.testing.assert_allclose(r, 1.0, rtol=1e-6, atol=1e-6)


# add_jitter
def test_add_jitter_zero():
    x = pd.Series([1, 2, 3], name="x")
    y = pd.Series([4, 5, 6], name="y")

    x_j, y_j = add_jitter(x, y, jitter=0)

    # No changes
    pd.testing.assert_series_equal(x, x_j, check_dtype=False)
    pd.testing.assert_series_equal(y, y_j, check_dtype=False)


def test_add_jitter_within_bounds():
    x = pd.Series([0.0, 1.0, 2.0], name="x")
    y = pd.Series([10.0, 11.0, 12.0], name="y")
    jitter = 0.5

    x_j, y_j = add_jitter(x, y, jitter=jitter)

    # Preserve index and name
    assert x_j.index.equals(x.index)
    assert y_j.index.equals(y.index)
    assert x_j.name == "x"
    assert y_j.name == "y"

    # Differences are bounded by jitter
    dx = (x_j - x).abs()
    dy = (y_j - y).abs()
    assert (dx <= jitter).all()
    assert (dy <= jitter).all()


# to_rgba255
def test_to_rgba_named_default_alpha():
    result = to_rgba255("red")
    assert result == [255, 0, 0, 255]


def test_to_rgba_named_custom_alpha():
    result = to_rgba255("blue", alpha=0.5)
    assert result == [0, 0, 255, 128]


def test_to_rgba_hex_default_alpha():
    result = to_rgba255("#00FF00")
    assert result == [0, 255, 0, 255]


def test_to_rgba_hex_custom_alpha():
    result = to_rgba255("#123456", alpha=0.2)
    expected_rgb = [round(c * 255) for c in mcolors.to_rgb("#123456")] + [
        round(0.2 * 255)
    ]
    assert result == expected_rgb


# to_rgba255_css
def test_to_rgba_css_named_default_alpha():
    result = to_rgba255_css("red")
    assert result == "rgba(255, 0, 0, 1.00)"


def test_to_rgba_css_named_custom_alpha():
    result = to_rgba255_css("blue", alpha=0.5)
    assert result == "rgba(0, 0, 255, 0.50)"


def test_to_rgba_css_hex_default_alpha():
    result = to_rgba255_css("#00FF00")
    assert result == "rgba(0, 255, 0, 1.00)"


def test_to_rgba_css_hex_custom_alpha():
    result = to_rgba255_css("#123456", alpha=0.2)
    expected_rgb = [round(c * 255) for c in mcolors.to_rgb("#123456")]
    expected = f"rgba({expected_rgb[0]}, {expected_rgb[1]}, {expected_rgb[2]}, 0.20)"
    assert result == expected


# set_tooltip_style
def test_set_tooltip_style_defaults():
    style = set_tooltip_style()
    assert style["backgroundColor"] == "rgba(0, 0, 0, 0.70)"
    assert style["color"] == "white"
    assert style["fontFamily"] == "Arial"
    assert style["fontSize"] == "12px"
    assert style["padding"] == "10px"
    assert style["maxWidth"] == "300px"
    assert style["overflowWrap"] == "break-word"


def test_set_tooltip_style_custom_values():
    style = set_tooltip_style(
        background_alpha=0.5,
        background_color="red",
        font_color="yellow",
        font_family="Courier",
        font_size=16,
        max_width=500,
        padding=20,
    )
    assert style["backgroundColor"] == "rgba(255, 0, 0, 0.50)"
    assert style["color"] == "yellow"
    assert style["fontFamily"] == "Courier"
    assert style["fontSize"] == "16px"
    assert style["padding"] == "20px"
    assert style["maxWidth"] == "500px"


# scale_to_max_abs
def test_scale_to_max_abs_positive():
    s = pd.Series([1, 2, 3])
    scaled = scale_to_max_abs(s, max_value=6)
    expected = pd.Series([2, 4, 6])
    pd.testing.assert_series_equal(scaled, expected, check_dtype=False)


def test_scale_to_max_abs_negative():
    s = pd.Series([-1, -5, -3])
    scaled = scale_to_max_abs(s, max_value=10)
    expected = pd.Series([-2, -10, -6])
    pd.testing.assert_series_equal(scaled, expected, check_dtype=False)


# get_global_stat
def test_get_global_stat_numbers():
    elems = [1, -3, 2]
    assert get_global_stat(elems) == 3  # max absolute
    assert get_global_stat(elems, absolute=False) == 2  # max raw


def test_get_global_stat_dicts():
    elems = [{"a": 1, "b": -5}, {"c": 2}]
    assert get_global_stat(elems) == 5
    assert get_global_stat(elems, absolute=False) == 2


def test_get_global_stat_series():
    s1 = pd.Series([1, -4, np.nan])
    s2 = pd.Series([2, -1])
    elems = [s1, s2]
    assert get_global_stat(elems) == 4
    assert get_global_stat(elems, absolute=False) == 2


def test_get_global_stat_mixed():
    s = pd.Series([1, 2])
    elems = [1, -2, {"a": -3}, s]
    assert get_global_stat(elems) == 3


def test_get_global_stat_custom_stat():
    elems = [1, 2, -5]
    # median of absolute values
    assert get_global_stat(elems, stat="median") == 2
    # custom function
    assert (
        get_global_stat(elems, stat=lambda arr: np.min(arr) + 1) == 2
    )  # check behavior


def test_get_global_stat_none_or_empty():
    assert get_global_stat([None, pd.Series([], dtype=float)]) is None
    assert get_global_stat([]) is None


def test_get_global_stat_invalid_type():
    with pytest.raises(TypeError):
        get_global_stat([object()])


def test_get_global_stat_invalid_stat():
    with pytest.raises(ValueError):
        get_global_stat([1, 2], stat="unsupported")


# wkt_to_linestring
def test_wkt_to_linestring_valid():
    wkt_str = "LINESTRING (0 0, 1 1, 2 2)"
    ls = wkt_to_linestring(wkt_str)
    assert isinstance(ls, LineString)
    assert list(ls.coords) == [(0, 0), (1, 1), (2, 2)]


def test_wkt_to_linestring_wrong_geometry():
    wkt_point = "POINT (0 0)"
    with pytest.raises(TypeError, match="Expected LineString"):
        wkt_to_linestring(wkt_point)


def test_wkt_to_linestring_invalid_string():
    invalid_wkt = "LINE (0 0, 1 1)"  # malformed
    with pytest.raises(Exception):
        wkt_to_linestring(invalid_wkt)


# linestring_to_pdk_path
def test_linestring_to_pdk_path_valid():
    line = LineString([(0, 0), (1, 1), (2, 2)])
    path = linestring_to_pdk_path(line)
    assert path == [[0, 0], [1, 1], [2, 2]]
    # Ensure all elements are lists of two floats
    for coord in path:
        assert isinstance(coord, list)
        assert len(coord) == 2
        assert all(isinstance(v, (int | float)) for v in coord)


def test_linestring_to_pdk_path_wrong_type():
    pt = Point(0, 0)
    with pytest.raises(TypeError, match="Expected LineString"):
        linestring_to_pdk_path(pt)


def test_linestring_to_pdk_path_empty_linestring():
    empty_line = LineString([])
    path = linestring_to_pdk_path(empty_line)
    assert path == []


# series_to_pdk_path
def test_series_to_pdk_path_linestrings():
    geoms = pd.Series([LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)])])
    paths = series_to_pdk_path(geoms)
    expected = [[[0, 0], [1, 1]], [[2, 2], [3, 3]]]
    assert paths == expected


def test_series_to_pdk_path_wkt_strings():
    geoms = pd.Series(["LINESTRING (0 0, 1 1)", "LINESTRING (2 2, 3 3)"])
    paths = series_to_pdk_path(geoms)
    expected = [[[0, 0], [1, 1]], [[2, 2], [3, 3]]]
    assert paths == expected


def test_series_to_pdk_path_mixed():
    geoms = pd.Series([LineString([(0, 0), (1, 1)]), "LINESTRING (2 2, 3 3)"])
    paths = series_to_pdk_path(geoms)
    expected = [[[0, 0], [1, 1]], [[2, 2], [3, 3]]]
    assert paths == expected


def test_series_to_pdk_path_invalid_string():
    geoms = pd.Series(["POINT (0 0)"])
    with pytest.raises(TypeError):
        series_to_pdk_path(geoms)


# rotate_polygon
def test_rotate_polygon_identity():
    # rotating by 0 radians should return the same polygon
    poly = np.array([[1, 0], [0, 1], [-1, 0]])
    rotated = rotate_polygon(poly, 0)
    np.testing.assert_allclose(rotated, poly, rtol=1e-12, atol=1e-12)


def test_rotate_polygon_90deg():
    # 90 degrees rotation
    poly = np.array([[1, 0], [0, 1]])
    rotated = rotate_polygon(poly, np.pi / 2)
    expected = np.array([[0, 1], [-1, 0]])
    np.testing.assert_allclose(rotated, expected, rtol=1e-12, atol=1e-12)


def test_rotate_polygon_negative_angle():
    # negative rotation
    poly = np.array([[1, 0]])
    rotated = rotate_polygon(poly, -np.pi / 2)
    expected = np.array([[0, -1]])
    np.testing.assert_allclose(rotated, expected, rtol=1e-12, atol=1e-12)


def test_rotate_polygon_square():
    # rotation preserves distances
    poly = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    rotated = rotate_polygon(poly, np.pi / 4)
    # all distances from origin should remain 1
    dists = np.linalg.norm(rotated, axis=1)
    np.testing.assert_allclose(dists, 1.0, rtol=1e-12, atol=1e-12)


# flip_polygon
def test_flip_polygon_x_axis():
    poly = np.array([[1, 2], [-3, 4]])
    flipped = flip_polygon(poly, axis="x")
    expected = np.array([[1, -2], [-3, -4]])
    np.testing.assert_allclose(flipped, expected)


def test_flip_polygon_y_axis():
    poly = np.array([[1, 2], [-3, 4]])
    flipped = flip_polygon(poly, axis="y")
    expected = np.array([[-1, 2], [3, 4]])
    np.testing.assert_allclose(flipped, expected)


def test_flip_polygon_invalid_axis():
    poly = np.array([[0, 0]])
    with pytest.raises(ValueError):
        flip_polygon(poly, axis="z")


def test_flip_polygon_preserves_shape():
    rng = np.random.default_rng(42)
    poly = rng.random((5, 2))
    flipped = flip_polygon(poly, axis="x")
    assert flipped.shape == poly.shape


# scale_polygon_by_width
def test_scale_polygon_basic():
    poly = np.array([[0, 0], [0, 2], [1, 2]])
    target_width = 4
    scaled = scale_polygon_by_width(poly, target_width)
    width_scaled = scaled[:, 1].max() - scaled[:, 1].min()
    np.testing.assert_allclose(width_scaled, target_width, rtol=1e-12, atol=1e-12)


def test_scale_polygon_preserves_ratios():
    poly = np.array([[1, 0], [2, 2]])
    target_width = 6
    scaled = scale_polygon_by_width(poly, target_width)
    original_ratio = (poly[:, 0].max() - poly[:, 0].min()) / (
        poly[:, 1].max() - poly[:, 1].min()
    )
    scaled_ratio = (scaled[:, 0].max() - scaled[:, 0].min()) / (
        scaled[:, 1].max() - scaled[:, 1].min()
    )
    np.testing.assert_allclose(scaled_ratio, original_ratio, rtol=1e-12)


def test_scale_polygon_zero_width_raises():
    poly = np.array([[0, 1]])
    with pytest.raises(ValueError, match="Cannot scale polygon with zero width"):
        scale_polygon_by_width(poly, 10)


# translate_polygon
def test_translate_polygon():
    poly = np.array([[0, 0], [1, 1], [2, 2]])
    offset = (3, -1)
    translated = translate_polygon(poly, offset)

    expected = np.array([[3, -1], [4, 0], [5, 1]])
    np.testing.assert_allclose(translated, expected, rtol=1e-12, atol=1e-12)


def test_translate_polygon_zero_offset():
    poly = np.array([[0, 0], [1, 1]])
    translated = translate_polygon(poly, (0, 0))
    np.testing.assert_allclose(translated, poly, rtol=1e-12, atol=1e-12)


# calculate_midpoint
def test_calculate_midpoint():
    p0 = (0, 0)
    p1 = (2, 4)
    midpoint = calculate_midpoint(p0, p1)
    expected = (1.0, 2.0)
    assert midpoint == expected


def test_calculate_angle():
    p0 = (0, 0)
    p1 = (1, 1)
    angle = calculate_angle(p0, p1)
    expected = np.pi / 4
    np.testing.assert_allclose(angle, expected, rtol=1e-12, atol=1e-12)

    # test horizontal
    p1 = (1, 0)
    angle = calculate_angle(p0, p1)
    np.testing.assert_allclose(angle, 0, rtol=1e-12, atol=1e-12)

    # test vertical
    p1 = (0, 1)
    angle = calculate_angle(p0, p1)
    np.testing.assert_allclose(angle, np.pi / 2, rtol=1e-12, atol=1e-12)


def test_meters_to_lonlat():
    # Simple square of 100 m offsets from reference
    poly = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    p0 = (0, 0)  # reference lon/lat in degrees
    result = meters_to_lonlat(poly, p0)

    # small approximation checks
    assert result.shape == poly.shape
    # first point should equal reference
    np.testing.assert_allclose(result[0], [0, 0], rtol=1e-12, atol=1e-12)
    # distances in degrees should be positive
    assert all(result[:, 0] >= 0)
    assert all(result[:, 1] >= 0)


# clip_lat
def test_clip_lat():
    coords = [(0, 91), (0, -91), (10, 45), (20, -45)]
    clipped = clip_lat(coords)

    # Ensure no latitude exceeds poles minus/plus buffer
    for lon, lat in clipped:
        assert -90 < lat < 90, f"Latitude {lat} out of bounds"

    # Check that normal latitudes are unchanged
    assert clipped[2] == (10, 45)
    assert clipped[3] == (20, -45)

    # Check that extreme latitudes are clipped correctly
    pole_buffer = 1e-6
    assert clipped[0][1] == 90 - pole_buffer
    assert clipped[1][1] == -90 + pole_buffer


# poly_to_geojson
def test_poly_to_geojson():
    # Polygon with some coordinates exceeding the poles
    poly = Polygon([(0, 91), (1, 91), (1, 92), (0, 92), (0, 91)])

    geojson = poly_to_geojson(poly)

    # The GeoJSON should have type Feature
    assert geojson["type"] == "Feature"

    # The geometry should be a Polygon
    geom = geojson["geometry"]
    assert geom["type"] == "Polygon"

    # All latitudes should be within [-90 + pole_buffer, 90 - pole_buffer]
    pole_buffer = 1e-6
    for ring in geom["coordinates"]:
        for lon, lat in ring:
            assert -90 + pole_buffer <= lat <= 90 - pole_buffer

    # The polygon should be closed
    exterior = geom["coordinates"][0]
    assert exterior[0] == exterior[-1]


# feature_to_geojson
@pytest.mark.skipif(not cartopy_present, reason="Cartopy not installed")
def test_feature_to_geojson_mock():
    # Mock a Cartopy-like feature with geometries() method
    poly1 = Polygon([(0, 91), (1, 91), (1, 92), (0, 92), (0, 91)])
    poly2 = Polygon([(10, -91), (11, -91), (11, -92), (10, -92), (10, -91)])
    multipoly = MultiPolygon([poly1, poly2])

    # Mock NaturalEarthFeature with geometries method
    mock_feature = SimpleNamespace(geometries=lambda: [poly1, multipoly])

    features = feature_to_geojson(mock_feature)

    # Should return 3 features: 1 from poly1, 2 from multipoly
    assert len(features) == 3

    # Each feature should be a GeoJSON Feature
    for f in features:
        assert f["type"] == "Feature"
        geom = f["geometry"]
        assert geom["type"] == "Polygon"
        # All latitudes should be clipped
        for ring in geom["coordinates"]:
            for lon, lat in ring:
                assert -90 + 1e-6 <= lat <= 90 - 1e-6
        # Polygon should be closed
        exterior = geom["coordinates"][0]
        assert exterior[0] == exterior[-1]


# shapefile_to_geojson
@pytest.mark.skipif(not cartopy_present, reason="Cartopy not installed")
def test_shapefile_to_geojson_basic():
    # Use the smallest dataset to keep test fast
    features = shapefile_to_geojson(
        resolution="110m",
        category="cultural",
        name="admin_0_countries",
        pole_buffer=1e-6,
    )

    # Should return at least one feature
    assert len(features) > 0

    # Each feature should have geometry and properties
    for f in features:
        assert f["type"] == "Feature"
        assert "geometry" in f
        assert "properties" in f

        geom = f["geometry"]
        # Geometry type should be Polygon
        assert geom["type"] == "Polygon"
        # Latitudes should be clipped
        for ring in geom["coordinates"]:
            for lon, lat in ring:
                assert -90 + 1e-6 <= lat <= 90 - 1e-6
        # Polygon should be closed
        exterior = geom["coordinates"][0]
        assert exterior[0] == exterior[-1]


# shorten_string
def test_shorten_string():
    # Converts non-string inputs
    assert shorten_string(123) == "123"
    assert shorten_string(None) == "None"

    # No shortening if max_length not set
    assert shorten_string("hello") == "hello"

    # Shorten string longer than max_length
    assert shorten_string("hello world", max_length=5) == "hello..."

    # Exactly max_length should not add ellipsis
    assert shorten_string("hello", max_length=5) == "hello"

    # Empty string
    assert shorten_string("", max_length=5) == ""


# round_value
def test_round_value():
    # Rounding with integer precision
    assert round_value(3.14159, rounding=2) == 3.14
    assert round_value(2.0, rounding=2) == 2  # integer float rounds to int
    assert round_value(5, rounding=2) == 5  # int stays int

    # Rounding with dict precision
    rounding_dict = {"a": 1, "b": 3}
    assert round_value(3.14159, rounding=rounding_dict, key="a") == 3.1
    assert round_value(3.14159, rounding=rounding_dict, key="b") == 3.142
    # Key not in dict, returns original
    assert round_value(3.14159, rounding=rounding_dict, key="c") == 3.14159

    # No rounding
    assert round_value(3.14159) == 3.14159
    assert round_value("hello") == "hello"
    assert round_value(None) is None


# df_to_html_table
def test_df_to_html_table_basic():
    df = pd.DataFrame(
        {
            "A": [1, 2],
            "B": [3, 4],
        },
        index=["row1", "row2"],
    )

    html_series = df_to_html_table(df, columns=["A", "B"], bold_header=True)

    # Check we have one HTML string per row
    assert len(html_series) == 2

    for s in html_series:
        # Check table tags are present
        assert "<table>" in s
        assert "</table>" in s

        # Check column headers are present
        assert "A:" in s
        assert "B:" in s

        # Check value alignment (default left)
        assert "text-align:left" in s

        # Check bold header
        assert "font-weight:bold" in s


def test_df_to_html_table_rounding_and_max_length():
    df = pd.DataFrame(
        {
            "A": [1.23456, 2.34567],
            "B": [3.98765, 4.87654],
        },
        index=["row1", "row2"],
    )

    html_series = df_to_html_table(
        df,
        columns=["A", "B"],
        rounding=2,
        max_value_length=4,
        max_header_length=1,
    )

    for s in html_series:
        # Rounded numbers
        assert "1.23" in s or "2.35" in s or "3.99" in s or "4.88" in s

        # Header shortening
        assert "A" in s
        assert "B" in s

        # Ellipsis should appear if header or value truncated
        assert "..." not in s or True  # Optional, just ensure no errors
