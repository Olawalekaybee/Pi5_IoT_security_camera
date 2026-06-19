"""Point-in-polygon zone matching for restricted-area detection."""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """Ray-casting algorithm — standard point-in-polygon test."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def resolve_zone(
    bbox: Tuple[int, int, int, int],
    zone_polygons: Dict[str, List[List[float]]],
    default: str = "unzoned",
) -> str:
    """Return the first zone whose polygon contains the bbox center."""
    center = bbox_center(bbox)
    for zone_name, polygon in zone_polygons.items():
        if len(polygon) >= 3 and point_in_polygon(center, polygon):
            return zone_name
    return default
