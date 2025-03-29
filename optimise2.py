import cv2
import os
import numpy as np
from osgeo import gdal

img = cv2.imread('./rogaine.png')
dem_path = './chc-dem'

class DatumPoint:
    def __init__(self, x, y, easting=None, northing=None) -> None:
        self.x = x
        self.y = y
        self.easting = easting
        self.northing = northing


class Control:
    def __init__(self, x, y, easting=None, northing=None, elevation=None) -> None:
        self.x = x
        self.y = y
        self.easting = easting
        self.northing = northing
        self.elevation = elevation


class DEM:
    def __init__(self):
        self.dem_files = []
        self.dem_mosaic = None
        self.gt = None


    def get_elevation(self, easting, northing):
        ds = gdal.Open('./mosaic.vrt')
        print(f"dem file is {ds.RasterXSize} x {ds.RasterYSize}")
        # Origin = (1576723.000000000232831,5177250.000000000000000)
        # Pixel Size = (1.000000000000000,-1.000000000000000)
        # Corner Coordinates:
        # Upper Left  ( 1576723.000, 5177250.000) (172d42'42.49"E, 43d33'26.43"S)
        # Lower Left  ( 1576723.000, 5171831.000) (172d42'41.65"E, 43d36'22.08"S)
        # Upper Right ( 1584221.000, 5177250.000) (172d48'16.69"E, 43d33'27.14"S)
        # Lower Right ( 1584221.000, 5171831.000) (172d48'16.12"E, 43d36'22.79"S)
        # Center      ( 1580472.000, 5174540.500) (172d45'29.24"E, 43d34'54.64"S)
        if self.dem_mosaic is not None and self.gt is not None:
            easting_dem_space = easting - int(self.gt[0])
            northing_dem_space = int(self.gt[3]) - northing
            print(f"dem space coordinates: {easting_dem_space}, {northing_dem_space}")
            if 0 <= easting_dem_space < self.dem_mosaic.shape[1] and 0 <= northing_dem_space < self.dem_mosaic.shape[0]:
                elevation = self.dem_mosaic[northing_dem_space, easting_dem_space]
                print(f"Elevation of point at: Longitude={easting:.6f}, Latitude={northing:.6f}, Elevation={elevation:.2f} m")
                return elevation
            else:
                print("point not within DEM")
                return None


    def load_dem_files(self, folder_path):
        """Load DEM files with proper extent handling during downsampling"""
        self.dem_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.tif', '.tiff', '.dem'))]
        if not self.dem_files:
            raise ValueError("No DEM files found in the specified folder")
        # Create virtual mosaic
        vrt = gdal.BuildVRT("mosaic.vrt", self.dem_files)
        # Get ORIGINAL geotransform and extent (before any downsampling)
        self.gt = vrt.GetGeoTransform()
        print(self.gt)
        original_extent = (
            self.gt[0],  # x_min
            self.gt[0] + vrt.RasterXSize * self.gt[1],  # x_max
            self.gt[3] + vrt.RasterYSize * self.gt[5],  # y_min
            self.gt[3]   # y_max
        )
        self.dem_mosaic = vrt.ReadAsArray()
        self.dem_mosaic = self.dem_mosaic.clip(min=0)
        vrt = None  # Cleanup


def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(controls) == 0:
            set_start_point(x, y)
        else:
            add_control(x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        set_start_point(x, y)


def add_control(x, y):
        cv2.circle(img,(x,y),20,(255,255,0),2)
        cv2.circle(img,(x,y),2,(255,255,0),-1)
        controls.append(Control(x, y))


def set_start_point(x, y):
        cv2.circle(img,(x,y),20,(255,100,0),2)
        cv2.circle(img,(x,y),2,(255,100,0),-1)
        if len(controls) == 0:
            controls.append(Control(x, y))
        else:
            controls[0] = Control(x, y)


def select_datum_points(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),20,(80,255,0),2)
        cv2.circle(img,(x,y),2,(80,255,0),-1)
        point = DatumPoint(x, y)
        while point.easting == None:
            user_input = input("Input easting: ")
            try:
                point.easting = int(user_input)
                print(point.easting)
            except:
                print("input a float")
        while point.northing == None:
            user_input = input("Input northing: ")
            try:
                point.northing = int(user_input)
                print(point.northing)
            except:
                print("input a float")
        datums.append(point)


datums = []
controls = []


def main():
    dem = DEM()
    dem.load_dem_files(dem_path)

    # step 1 - set datum points
    # cv2.namedWindow('set datum points')
    # cv2.setMouseCallback('set datum points',select_datum_points)
    # while(1):
    #     cv2.imshow('set datum points',img)
    #     k = cv2.waitKey(20) & 0xFF
    #     if k == 27:
    #         break
    # cv2.destroyAllWindows()

    datums.append(DatumPoint(0, 0, 1579719, 5173145)) # EXPECTED VALUE ~ 194m

    for datum in datums:
        # MAP IS CORRECT. WE ARE INDEXING INTO THE DEM WRONG HERE
        print(f"for point {datum.easting}, {datum.northing}")
        elevation = dem.get_elevation(datum.easting, datum.northing)
        print(f"elevation: {elevation}")

    # step 2 - select controls
    # cv2.namedWindow('image')
    # cv2.setMouseCallback('image',draw_circle)
    # while(1):
    #     cv2.imshow('image',img)
    #     k = cv2.waitKey(20) & 0xFF
    #     if k == 27:
    #         break
    # print(controls)
    # for index, control in enumerate(controls):
    #     if index != 0:
    #         cv2.line(img, (controls[index-1].x, controls[index-1].y), (control.x, control.y), (255, 255, 0), 2)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)

main()
