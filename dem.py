import os
import numpy as np
from math import sqrt
from osgeo import gdal


class DEM:
    def __init__(self):
        self.dem_files = []
        self.dem_mosaic = None
        self.gt = None
        self.px_p_m = None

    def load_dem_files(self, folder_path):
        """Load and mosaic DEM files, caching the result."""
        gdal.UseExceptions()
        self.dem_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".tif", ".tiff", ".dem"))
        ]
        if not self.dem_files:
            raise ValueError("No DEM files found in the specified folder")

        vrt = gdal.BuildVRT("mosaic.vrt", self.dem_files)
        self.gt = vrt.GetGeoTransform()
        self.dem_mosaic = vrt.ReadAsArray()
        self.dem_mosaic = self.dem_mosaic.clip(min=0)
        self._raster_x_size = vrt.RasterXSize
        self._raster_y_size = vrt.RasterYSize
        vrt = None

    def get_elevation(self, easting, northing):
        """Look up elevation from cached DEM using geotransform."""
        if self.dem_mosaic is None or self.gt is None:
            raise RuntimeError("DEM not loaded. Call load_dem_files first.")

        gt = self.gt
        if gt[1] == 0 or gt[5] == 0:
            return None

        px = int((easting - gt[0]) / gt[1])
        py = int((northing - gt[3]) / gt[5])

        if 0 <= px < self.dem_mosaic.shape[1] and 0 <= py < self.dem_mosaic.shape[0]:
            return float(self.dem_mosaic[py, px])
        return None

    def calc_rogaine_map_to_dem_conversion(self, datum_points):
        """Calculate px/m ratio from datum points (legacy, use affine georef instead)."""
        pxs_p_ms = []
        for f in datum_points:
            for to in datum_points:
                if f != to:
                    geo_dist = f.geo_distance_2d(to)
                    img_dist = f.px_distance(to)
                    if geo_dist > 0:
                        pxs_p_ms.append(img_dist / geo_dist)
        if pxs_p_ms:
            self.px_p_m = np.mean(pxs_p_ms)
        return self.px_p_m
