import pytest
import numpy as np
from models import DatumPoint, Control
from georef import AffineGeoref, georeference_controls


REAL_DATUMS = [
    DatumPoint(311, 1452, easting=1579007, northing=5174415, label="Richmond Hill Corner"),
    DatumPoint(1729, 2501, easting=1579690, northing=5173184, label="Spaghetti Junction"),
    DatumPoint(1554, 966, easting=1580102, northing=5174370, label="CDS north of start"),
    DatumPoint(1506, 1224, easting=1579981, northing=5174179, label="CDS south of start"),
]


class TestAffineGeoref:
    def test_fit_with_real_datums(self):
        g = AffineGeoref()
        g.fit(REAL_DATUMS)
        assert g.easting_coeffs is not None
        assert g.northing_coeffs is not None

    def test_transform_reproduces_datums(self):
        g = AffineGeoref()
        g.fit(REAL_DATUMS)
        for d in REAL_DATUMS:
            e, n = g.transform(d.x, d.y)
            assert abs(e - d.easting) < 15.0, f"Easting error {abs(e - d.easting):.1f}m for {d.label}"
            assert abs(n - d.northing) < 15.0, f"Northing error {abs(n - d.northing):.1f}m for {d.label}"

    def test_fit_exact_with_two_datums(self):
        d1 = DatumPoint(0, 0, easting=100, northing=200, label="a")
        d2 = DatumPoint(100, 0, easting=200, northing=200, label="b")
        g = AffineGeoref()
        g.fit([d1, d2])
        e1, n1 = g.transform(0, 0)
        assert e1 == pytest.approx(100.0)
        assert n1 == pytest.approx(200.0)
        e2, n2 = g.transform(100, 0)
        assert e2 == pytest.approx(200.0)
        assert n2 == pytest.approx(200.0)

    def test_fit_too_few_datums(self):
        d = DatumPoint(0, 0, easting=100, northing=200, label="a")
        g = AffineGeoref()
        with pytest.raises(ValueError, match="at least 2"):
            g.fit([d])

    def test_transform_before_fit_raises(self):
        g = AffineGeoref()
        with pytest.raises(RuntimeError, match="not fitted"):
            g.transform(0, 0)

    def test_pure_translation(self):
        d1 = DatumPoint(0, 0, easting=1000, northing=2000, label="a")
        d2 = DatumPoint(1, 0, easting=1001, northing=2000, label="b")
        d3 = DatumPoint(0, 1, easting=1000, northing=2001, label="c")
        g = AffineGeoref()
        g.fit([d1, d2, d3])
        e, n = g.transform(5, 5)
        assert e == pytest.approx(1005.0)
        assert n == pytest.approx(2005.0)

    def test_translation_and_scale(self):
        d1 = DatumPoint(0, 0, easting=0, northing=0, label="a")
        d2 = DatumPoint(10, 0, easting=20, northing=0, label="b")
        d3 = DatumPoint(0, 10, easting=0, northing=20, label="c")
        g = AffineGeoref()
        g.fit([d1, d2, d3])
        e, n = g.transform(5, 5)
        assert e == pytest.approx(10.0)
        assert n == pytest.approx(10.0)

    def test_report_runs(self, capsys):
        g = AffineGeoref()
        g.fit(REAL_DATUMS)
        g.report()
        captured = capsys.readouterr()
        assert "Affine" in captured.out


class TestGeoreferenceControls:
    def test_sets_easting_northing(self):
        g = AffineGeoref()
        g.fit(REAL_DATUMS)
        c = Control(311, 1452, label="0")
        georeference_controls([c], g)
        assert c.easting is not None
        assert c.northing is not None
        assert abs(c.easting - 1579007) < 5.0
        assert abs(c.northing - 5174415) < 5.0

    def test_multiple_controls(self):
        g = AffineGeoref()
        g.fit(REAL_DATUMS)
        controls = [Control(100, 200, label="50"), Control(300, 400, label="30")]
        georeference_controls(controls, g)
        for c in controls:
            assert c.easting is not None
            assert c.northing is not None
