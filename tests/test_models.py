import pytest
from math import sqrt
from models import Point, DatumPoint, Control, Path


class TestPoint:
    def test_unique_ids(self):
        p1 = Point(0, 0)
        p2 = Point(0, 0)
        assert p1.id != p2.id

    def test_equality_by_id(self):
        p1 = Point(5, 5)
        p2 = Point(5, 5)
        assert p1 != p2

    def test_equality_same_instance(self):
        p = Point(1, 2)
        assert p == p

    def test_not_equal_to_non_point(self):
        p = Point(1, 2)
        assert p != "not a point"
        assert p != 42
        assert p != None

    def test_hash_is_id(self):
        p = Point(3, 4)
        assert hash(p) == p.id

    def test_usable_as_dict_key(self):
        p1 = Point(0, 0)
        p2 = Point(1, 1)
        d = {p1: "a", p2: "b"}
        assert d[p1] == "a"
        assert d[p2] == "b"

    def test_geo_distance_2d(self):
        p1 = Point(0, 0, easting=0, northing=0)
        p2 = Point(0, 0, easting=3, northing=4)
        assert p1.geo_distance_2d(p2) == pytest.approx(5.0)

    def test_geo_distance_2d_zero(self):
        p = Point(0, 0, easting=10, northing=20)
        assert p.geo_distance_2d(p) == pytest.approx(0.0)

    def test_px_distance(self):
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        assert p1.px_distance(p2) == pytest.approx(5.0)

    def test_px_distance_zero(self):
        p = Point(7, 8)
        assert p.px_distance(p) == pytest.approx(0.0)

    def test_repr(self):
        p = Point(1, 2, label="test")
        assert "test" in repr(p)


class TestDatumPoint:
    def test_is_point_subclass(self):
        d = DatumPoint(1, 2, easting=100, northing=200, label="dp")
        assert isinstance(d, Point)

    def test_repr(self):
        d = DatumPoint(1, 2, label="dp")
        assert "dp" in repr(d)


class TestControl:
    def test_auto_derive_points_from_label(self):
        c = Control(0, 0, label="53")
        assert c.points == 50

    def test_label_zero(self):
        c = Control(0, 0, label="0")
        assert c.points == 0

    def test_label_three_digits(self):
        c = Control(0, 0, label="104")
        assert c.points == 100

    def test_explicit_points_not_overridden(self):
        c = Control(0, 0, label="53", points=999)
        assert c.points == 999

    def test_no_label_no_points(self):
        c = Control(0, 0)
        assert c.points is None

    def test_set_label_updates_points(self):
        c = Control(0, 0, label="0")
        assert c.points == 0
        c.set_label("92")
        assert c.label == "92"
        assert c.points == 90

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="Invalid control label"):
            Control(0, 0, label="abc")

    def test_elevation_difference(self):
        c1 = Control(0, 0, label="0", elevation=100)
        c2 = Control(0, 0, label="0", elevation=150)
        assert c1.get_elevation_difference(c2) == pytest.approx(50.0)

    def test_elevation_difference_symmetric(self):
        c1 = Control(0, 0, label="0", elevation=100)
        c2 = Control(0, 0, label="0", elevation=150)
        assert c1.get_elevation_difference(c2) == c2.get_elevation_difference(c1)

    def test_elevation_difference_none(self):
        c1 = Control(0, 0, label="0", elevation=None)
        c2 = Control(0, 0, label="0", elevation=100)
        assert c1.get_elevation_difference(c2) == 0.0

    def test_get_climb_uphill(self):
        c1 = Control(0, 0, label="0", elevation=100)
        c2 = Control(0, 0, label="0", elevation=150)
        assert c1.get_climb(c2) == pytest.approx(50.0)

    def test_get_climb_downhill_is_zero(self):
        c1 = Control(0, 0, label="0", elevation=150)
        c2 = Control(0, 0, label="0", elevation=100)
        assert c1.get_climb(c2) == pytest.approx(0.0)

    def test_get_climb_none_elevation(self):
        c1 = Control(0, 0, label="0", elevation=None)
        c2 = Control(0, 0, label="0", elevation=100)
        assert c1.get_climb(c2) == 0.0

    def test_geo_distance_3d(self):
        c1 = Control(0, 0, label="0", elevation=0)
        c1.easting, c1.northing = 0, 0
        c2 = Control(0, 0, label="0", elevation=10)
        c2.easting, c2.northing = 30, 40
        dist = c1.geo_distance_3d(c2, climb_factor=1.0)
        expected = sqrt(30**2 + 40**2 + 10**2)
        assert dist == pytest.approx(expected)

    def test_str_repr(self):
        c = Control(0, 0, label="42")
        assert "42" in str(c)
        assert "42" in repr(c)


class TestPath:
    def test_defaults(self):
        p = Path()
        assert p.points == 0
        assert p.total_cost == 0.0
        assert p.distance_2d == 0.0
        assert p.total_climb == 0.0
        assert p.controls == []

    def test_repr(self):
        p = Path()
        assert "pts=0" in repr(p)
