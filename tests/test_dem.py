import os
import pytest
import numpy as np
from dem import DEM
from models import DatumPoint

DEM_DIR = os.path.join(os.path.dirname(__file__), "..", "chc-dem")
HAS_DEM = os.path.isdir(DEM_DIR) and any(
    f.lower().endswith((".tif", ".tiff", ".dem")) for f in os.listdir(DEM_DIR)
)

needs_dem = pytest.mark.skipif(not HAS_DEM, reason="DEM files not available")


class TestDEMInit:
    def test_defaults(self):
        d = DEM()
        assert d.dem_mosaic is None
        assert d.gt is None
        assert d.px_p_m is None
        assert d.dem_files == []

    def test_get_elevation_before_load_raises(self):
        d = DEM()
        with pytest.raises(RuntimeError, match="DEM not loaded"):
            d.get_elevation(1579007, 5174415)


@needs_dem
class TestDEMLoad:
    @pytest.fixture(scope="class")
    def dem(self):
        d = DEM()
        d.load_dem_files(DEM_DIR)
        return d

    def test_loads_successfully(self, dem):
        assert dem.dem_mosaic is not None
        assert dem.gt is not None
        assert len(dem.dem_files) > 0

    def test_mosaic_shape(self, dem):
        assert len(dem.dem_mosaic.shape) == 2
        assert dem.dem_mosaic.shape[0] > 0
        assert dem.dem_mosaic.shape[1] > 0

    def test_no_negative_values(self, dem):
        assert dem.dem_mosaic.min() >= 0

    def test_geotransform_has_six_elements(self, dem):
        assert len(dem.gt) == 6


@needs_dem
class TestDEMElevation:
    @pytest.fixture(scope="class")
    def dem(self):
        d = DEM()
        d.load_dem_files(DEM_DIR)
        return d

    def test_known_coordinate(self, dem):
        elev = dem.get_elevation(1579007, 5174415)
        assert elev is not None
        assert isinstance(elev, float)

    def test_reasonable_range(self, dem):
        elev = dem.get_elevation(1579007, 5174415)
        assert 0 <= elev <= 2000

    def test_out_of_bounds_returns_none(self, dem):
        assert dem.get_elevation(0, 0) is None

    def test_far_out_of_bounds(self, dem):
        assert dem.get_elevation(9999999, 9999999) is None


class TestDEMLoadErrors:
    def test_nonexistent_dir(self):
        d = DEM()
        with pytest.raises(Exception):
            d.load_dem_files("/nonexistent/path/xyz")

    def test_empty_dir(self, tmp_path):
        d = DEM()
        with pytest.raises(ValueError, match="No DEM files found"):
            d.load_dem_files(str(tmp_path))


@needs_dem
class TestDEMConversion:
    def test_calc_px_p_m(self):
        d = DEM()
        d.load_dem_files(DEM_DIR)
        datums = [
            DatumPoint(311, 1452, easting=1579007, northing=5174415, label="a"),
            DatumPoint(1729, 2501, easting=1579690, northing=5173184, label="b"),
            DatumPoint(1554, 966, easting=1580102, northing=5174370, label="c"),
        ]
        result = d.calc_rogaine_map_to_dem_conversion(datums)
        assert result is not None
        assert result > 0
