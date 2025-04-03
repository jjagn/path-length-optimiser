import cv2
import json
import os
import numpy as np
from numpy import mean, sqrt
from osgeo import gdal
import random

img = cv2.imread('./map.png')
dem_path = './chc-dem'


class Graph:
    def __init__(self, graph: dict = {}):
        self.graph = graph

    def add_edge(self, node1, node2, weight):
        if node1 not in self.graph:
            self.graph[node1] = {}
        else:
            self.graph[node1][node2] = weight


class Point:
    def __init__(self, x, y, easting=None, northing=None) -> None:
        self.x = x
        self.y = y
        self.easting = easting
        self.northing = northing
        if easting and northing:
            self.hash = int(x * y * easting * northing)
        else:
            self.hash = int((x + x * y + y) * x * y + 86773629)  # random enough?

    def geo_distance_1D(self, other):
        return sqrt((self.easting - other.easting)**2 + (self.northing - other.northing)**2)

    def px_distance(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return self.hash


class DatumPoint(Point):
    def __init__(self, x, y, easting=None, northing=None, label=None) -> None:
        super().__init__(x, y, easting, northing)
        self.label = label


class Control(Point):
    def __init__(self, x, y, easting=None, northing=None, elevation=None, points=None) -> None:
        super().__init__(x, y, easting, northing)
        self.elevation = elevation
        self.points = points

    def get_easting_and_northing_from_datums(self, datums, px_p_m):
        eastings = []
        northings = []
        for datum in datums:
            px_x = datum.x - self.x
            px_y = datum.y - self.y
            easting = datum.easting + px_x * px_p_m
            northing = datum.northing - px_y * px_p_m
            print(f"point easting: {easting}, northing: {northing} relative to datum {datum.label}")
            eastings.append(easting)
            northings.append(northing)
        self.easting = np.mean(eastings)
        self.northing = np.mean(northings)

    def get_graph_weight(self, other):
        # return self.geo_distance_1D(other) / sqrt(other.points) #?? might work I guess
        return self.geo_distance_1D(other) # just go by distance for now


class DEM:
    def __init__(self):
        self.dem_files = []
        self.dem_mosaic = None
        self.gt = None
        self.px_p_m = None

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

    def calc_rogaine_map_to_dem_conversion(self, datum_points):
        pxs_p_ms = []
        for f in datum_points:   # can't use from so i'm calling it f
            for to in datum_points:
                if f != to:
                    geo_dist = f.geo_distance_1D(to)
                    img_dist = f.px_distance(to)
                    px_p_m = img_dist / geo_dist
                    pxs_p_ms.append(px_p_m)
                    if f.label and to.label:
                        print(f"from {f.label} to {to.label}")
                    print(f"geographic distance: {geo_dist}m")
                    print(f"pixel distance: {img_dist}px")
                    print(f"(px/m = {px_p_m}")
        self.px_p_m = np.mean(pxs_p_ms)

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
    global datum_process_index
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),20,(80,255,0),2)
        cv2.circle(img,(x,y),2,(80,255,0),-1)
        dp = datums_to_process[datum_process_index]
        e = dp['easting']
        n = dp['northing']
        label = dp['label']
        point = DatumPoint(x, y, e, n, label)
        datums.append(point)
        datum_process_index += 1


controls = []
datums = []
datums_to_process = []
datum_process_index = 0


def main():
    global datums_to_process
    gdal.UseExceptions()
    dem = DEM()
    dem.load_dem_files(dem_path)

    with open('datums2.json', 'r') as file:
        data = json.load(file)

    for datum_key_string in data.keys():
        datum = data[datum_key_string]
        d = DatumPoint(datum['x'], datum['y'], datum['easting'], datum['northing'], datum_key_string)
        datums.append(d)
    # print(data)
    for datum in datums:
        print(datum)
        x = datum.x
        y = datum.y
        cv2.circle(img,(x,y),20,(80,255,0),2)
        cv2.circle(img,(x,y),2,(80,255,0),-1)


    # datums_to_process = data['datums']
    # print(datums_to_process)

    # print("Select datums in the below order")
    # for item in datums_to_process:
    #     print(item['label'])
    #     print(item['easting'])
    #     print(item['northing'])
    #
    # # step 1 - set datum points
    # cv2.namedWindow('set datum points')
    # cv2.setMouseCallback('set datum points',select_datum_points)
    # while(1):
    #     cv2.imshow('set datum points',img)
    #     k = cv2.waitKey(20) & 0xFF
    #     if k == 27:
    #         break
    #     if datum_process_index >= len(datums_to_process):
    #         break
    # cv2.destroyAllWindows()

    datums_dict = {}

    for datum in datums:
        datums_dict[datum.label] = {'x': datum.x, 'y': datum.y, 'easting': datum.easting, 'northing': datum.northing}
    
    print("datums export")
    print(json.dumps(datums_dict))
    # richmond hill road corner 1579007, 5174415 195, 978
    # spaghetti junction 1579690, 5173184   1106, 1632
    # cul-de-sac just north of start E 1580102, N 5174370 1009, 656
    # cul-de-sac end south of start E 1579981, N 5174179 977, 814

    # datums.append(DatumPoint(195, 978, 1579007, 5174415, "richmond hill road corner"))
    # datums.append(DatumPoint(1106, 1632, 1579690, 5173184, "spaghetti junction"))
    # datums.append(DatumPoint(1009, 656, 1580102, 5174370, "c-d-s north of start"))
    # datums.append(DatumPoint(977, 814, 1579981, 5174179, "c-d-s end south of start"))

    # for datum in datums:
    #     print(f"for point {datum.easting}, {datum.northing}")
    #     elevation = dem.get_elevation(datum.easting, datum.northing)
    #     print(f"elevation: {elevation}")

    dem.calc_rogaine_map_to_dem_conversion(datums)
    print(f"calculated px/m for map = {dem.px_p_m}")

    # step 2 - select controls
    # cv2.namedWindow('image')
    # cv2.setMouseCallback('image', draw_circle)
    # while (1):
    #     cv2.imshow('image', img)
    #     k = cv2.waitKey(20) & 0xFF
    #     if k == 27:
    #         break

    # for index, control in enumerate(controls):
    #     print(f"for control {index+1}")
    #     control.get_easting_and_northing_from_datums(datums, dem.px_p_m)
    #     if index != 0:
    #         dist_x = control.x - controls[index-1].x
    #         dist_y = control.y - controls[index-1].y
    #         print(f"px dist: {dist_x}, {dist_y}")
    #         geo_x = dist_x / dem.px_p_m
    #         geo_y = dist_y / dem.px_p_m
    #         print(f"geo dist: {geo_x}, {geo_y}")

   # should add something in here that exports the controls as json that i can load for debugging purposes 

    with open('controls.json', 'r') as file:
        controls_data = json.load(file)

    for c in controls_data["controls"]:
        control = Control(c["x"], c["y"], c["easting"], c["northing"])
        controls.append(control)

    for control in controls:
        x = control.x
        y = control.y
        cv2.circle(img,(x,y),20,(0,0,255),2)
        cv2.circle(img,(x,y),2,(0,0,255),-1)

    cv2.imshow('map', img)
    # cv2.waitKey(0)

    # for index, control in enumerate(controls):
    #     if index != 0:
    #         cv2.line(img, (controls[index-1].x, controls[index-1].y), (control.x, control.y), (255, 255, 0), 2)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # construct a 2d array of distances between controls
    distances = np.zeros((len(controls), len(controls)))
    for i, f in enumerate(controls):
        for j, t in enumerate(controls):
            distances[i, j] = f.geo_distance_1D(t)

    print(distances)

    print("JSON EXPORT")
    export = {}
    export["controls"] = []
    for control in controls:
        export["controls"].append({"x": control.x, "y": control.y, "easting": control.easting, "northing": control.northing})

    print(json.dumps(export))

    G = Graph()

    max_dist = 0

    img_with_paths = img

    for orig in controls:
        for dest in controls:
            if orig != dest:
                dist = orig.get_graph_weight(dest)
                if dist > max_dist:
                    max_dist = dist
                G.add_edge(orig, dest, dist)

    print(G)
    for origin_control in G.graph.keys():
        node = G.graph[origin_control]
        print(f"node key: {origin_control}")
        for destination_control in node.keys():
            dist = node[destination_control]
            # B, G, R
            if dist > max_dist/2:
                line_color = (0, 255 * (dist/max_dist), 255)
            else:
                line_color = (0, 255, 255 * (dist/max_dist))
            cv2.line(img_with_paths, (origin_control.x, origin_control.y), (destination_control.x, destination_control.y), line_color, 2)

    cv2.imshow('paths', img_with_paths)
    cv2.waitKey(0)

    

main()
