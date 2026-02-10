import numpy as np
from math import sqrt


_next_id = 0


def _get_next_id():
    global _next_id
    _next_id += 1
    return _next_id


class Point:
    def __init__(self, x, y, easting=None, northing=None, label=None):
        self.id = _get_next_id()
        self.x = x
        self.y = y
        self.easting = easting
        self.northing = northing
        self.label = label

    def geo_distance_2d(self, other):
        return sqrt((self.easting - other.easting) ** 2 + (self.northing - other.northing) ** 2)

    def px_distance(self, other):
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.id == other.id

    def __hash__(self):
        return self.id

    def __repr__(self):
        return f"Point(id={self.id}, label={self.label})"


class DatumPoint(Point):
    def __init__(self, x, y, easting=None, northing=None, label=None):
        super().__init__(x, y, easting, northing, label)

    def __repr__(self):
        return f"DatumPoint(id={self.id}, label={self.label})"


class Control(Point):
    def __init__(self, x, y, easting=None, northing=None, label=None, elevation=None, points=None):
        super().__init__(x, y, easting, northing, label)
        self.points = points
        self.elevation = elevation
        if self.label is not None and self.points is None:
            self.get_points_from_label()

    def set_label(self, label):
        self.label = label
        self.get_points_from_label()

    def get_points_from_label(self):
        try:
            self.points = (int(self.label) // 10) * 10
        except (TypeError, ValueError):
            raise ValueError(f"Invalid control label for points: {self.label!r}")

    def get_elevation_difference(self, other):
        if self.elevation is None or other.elevation is None:
            return 0.0
        return abs(self.elevation - other.elevation)

    def get_climb(self, other):
        """Returns positive climb only (uphill)."""
        if self.elevation is None or other.elevation is None:
            return 0.0
        diff = other.elevation - self.elevation
        return max(0.0, diff)

    def geo_distance_3d(self, other, climb_factor=10.0):
        return sqrt(
            (self.easting - other.easting) ** 2
            + (self.northing - other.northing) ** 2
            + ((self.elevation - other.elevation) * climb_factor) ** 2
        )

    def __str__(self):
        return f"Control({self.label})"

    def __repr__(self):
        return f"Control({self.label}, pts={self.points})"


class Path:
    def __init__(self):
        self.points = 0
        self.total_cost = 0.0
        self.distance_2d = 0.0
        self.total_climb = 0.0
        self.controls = []

    def __repr__(self):
        return f"Path(pts={self.points}, dist={self.distance_2d:.0f}m, climb={self.total_climb:.0f}m, controls={len(self.controls)})"
