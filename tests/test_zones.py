"""Unit tests for polygon-based zone resolution — no hardware required."""

import pytest
from src.utils.zones import point_in_polygon, bbox_center, resolve_zone


class TestPointInPolygon:
    def test_point_inside_square(self):
        square = [[0, 0], [10, 0], [10, 10], [0, 10]]
        assert point_in_polygon((5, 5), square) is True

    def test_point_outside_square(self):
        square = [[0, 0], [10, 0], [10, 10], [0, 10]]
        assert point_in_polygon((15, 15), square) is False

    def test_point_on_edge_boundary(self):
        square = [[0, 0], [10, 0], [10, 10], [0, 10]]
        # Edge behavior can vary; just ensure no crash and a bool returned
        assert isinstance(point_in_polygon((10, 5), square), bool)

    def test_triangle(self):
        triangle = [[0, 0], [10, 0], [5, 10]]
        assert point_in_polygon((5, 3), triangle) is True
        assert point_in_polygon((1, 9), triangle) is False


class TestBboxCenter:
    def test_center_calculation(self):
        assert bbox_center((0, 0, 10, 10)) == (5.0, 5.0)

    def test_non_square_bbox(self):
        assert bbox_center((0, 0, 20, 10)) == (10.0, 5.0)

    def test_offset_bbox(self):
        assert bbox_center((10, 10, 30, 50)) == (20.0, 30.0)


class TestResolveZone:
    def test_resolves_to_correct_zone(self):
        polygons = {
            "zone_a": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "zone_b": [[200, 0], [300, 0], [300, 100], [200, 100]],
        }
        assert resolve_zone((10, 10, 50, 50), polygons) == "zone_a"
        assert resolve_zone((210, 10, 250, 50), polygons) == "zone_b"

    def test_returns_default_when_no_match(self):
        polygons = {"zone_a": [[0, 0], [10, 0], [10, 10], [0, 10]]}
        assert resolve_zone((500, 500, 550, 550), polygons) == "unzoned"

    def test_ignores_degenerate_polygon(self):
        polygons = {"bad_zone": [[0, 0], [10, 10]]}  # only 2 points
        assert resolve_zone((1, 1, 2, 2), polygons) == "unzoned"
