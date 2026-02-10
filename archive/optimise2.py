import cv2
import json
import os
import numpy as np
from numpy import mean, sqrt
from osgeo import gdal
import random
import copy
import matplotlib.pyplot as plt

img = cv2.imread('./map.png')
dem_path = './chc-dem'

display_work = False


class Graph:
    def __init__(self, graph: dict = {}, sorted: dict = {}):
        self.graph = graph
        self.sorted_in_weight_order = {}

    def add_edge(self, node1, node2, weight):
        if node1 not in self.graph:
            self.graph[node1] = {}
        self.graph[node1][node2] = weight

    def __str__(self):
        lines = ["Graph (Node -> {Neighbor: Weight}):"]
        for node, neighbors in self.graph.items():
            neighbor_str = ", ".join(
                [f"{neighbor}: {weight:.2f}" for neighbor, weight in neighbors.items()])
            lines.append(f"\n{node} -> \n{{{neighbor_str}}}")
        return "\n".join(lines)  # Returns a single string for printing

    def copy(self):
        """Returns a deep copy of the graph."""
        return Graph(copy.deepcopy(self.graph), copy.deepcopy(self.sorted_in_weight_order))

    def sort_nodes_by_weight(self):
        for node, neighbours in self.graph.items():
            sorted_neighbours = sorted(neighbours.items(), key=lambda x: x[1])
            self.sorted_in_weight_order[node] = [
                neighbour for neighbour, weight in sorted_neighbours]


class Point:
    def __init__(self, x, y, easting=None, northing=None, label=None) -> None:
        self.x = x
        self.y = y
        self.easting = easting
        self.northing = northing
        if easting and northing:
            self.hash = int(x * y * easting * northing)
        else:
            self.hash = int((x + x * y + y) * x * y +
                            86773629)  # random enough?
        self.label = label

    def geo_distance_2D(self, other):
        return sqrt((self.easting - other.easting)**2 + (self.northing - other.northing)**2)

    def px_distance(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __eq__(self, other):
        if self and other:
            return self.x == other.x and self.y == other.y
        else:
            return False

    def __hash__(self):
        return self.hash


class DatumPoint(Point):
    def __init__(self, x, y, easting=None, northing=None, label=None) -> None:
        super().__init__(x, y, easting, northing, label)


class Control(Point):
    def __init__(self, x, y, easting=None, northing=None, label=None, elevation=None, points=None) -> None:
        super().__init__(x, y, easting, northing, label)
        if self.label and not self.points:
            self.get_points_from_label()
        else:
            self.points = points
        self.elevation = elevation

    def set_label(self, label):
        self.label = label
        self.get_points_from_label

    def __str__(self):
        return f"Control: {self.label}"

    def __repr__(self):
        return f"Control: {self.label}"

    def get_elevation_difference(self, other):
        return sqrt((self.elevation - other.elevation)**2)

    def geo_distance_3D(self, other):
        # multiplied by elevation scale factor to encourage keeping on the flat
        return sqrt((self.easting - other.easting)**2 + (self.northing - other.northing)**2 + ((self.elevation - other.elevation)*10)**2)

    def get_points_from_label(self):
        self.points = (int(self.label) // 10) * 10

    def get_easting_and_northing_from_datums(self, datums, px_p_m):
        eastings = []
        northings = []
        copy = img
        for datum in datums:
            px_x = self.x - datum.x
            px_y = self.y - datum.y
            easting = datum.easting + px_x / px_p_m
            northing = datum.northing - px_y / px_p_m
            print(f"point easting: {easting}, northing: {
                  northing} relative to datum {datum.label}")
            eastings.append(easting)
            northings.append(northing)
            if display_work:
                cv2.line(copy, (self.x, self.y), (self.x -
                                                  px_x, self.y - px_y), (0, 100, 255), 2)
        if display_work:
            cv2.imshow('datum', copy)
            cv2.waitKey(20)
        self.easting = np.mean(eastings)
        self.northing = np.mean(northings)
        # to reduce error i could maybe get the distance from the closest datum or something, but i don't know why it's so far off

    def get_graph_weight(self, other):
        # return self.geo_distance_1D(other) / sqrt(other.points) #?? might work I guess
        # return self.geo_distance_3D(other)  # just go by distance for now
        return self.px_distance(other)


class Path:
    def __init__(self):
        self.points = 0
        self.total_distance = 0
        self.distance_2d = 0
        self.total_elevation = 0
        self.controls = []


class DEM:
    def __init__(self):
        self.dem_files = []
        self.dem_mosaic = None
        self.gt = None
        self.px_p_m = None

    def get_elevation(self, easting, northing):
        ds = gdal.Open('./mosaic.vrt')
        # print(f"dem file is {ds.RasterXSize} x {ds.RasterYSize}")
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
            print(f"dem space coordinates: {
                  easting_dem_space}, {northing_dem_space}")
            if 0 <= easting_dem_space < self.dem_mosaic.shape[1] and 0 <= northing_dem_space < self.dem_mosaic.shape[0]:
                elevation = self.dem_mosaic[int(
                    northing_dem_space), int(easting_dem_space)]
                print(f"Elevation of point at: easting={easting:.6f}, northing={
                      northing:.6f} is {elevation:.2f} m")
                return elevation
            else:
                print("point not within DEM")
                return None

    def calc_rogaine_map_to_dem_conversion(self, datum_points):
        pxs_p_ms = []
        for f in datum_points:   # can't use from so i'm calling it f
            for to in datum_points:
                if f != to:
                    geo_dist = f.geo_distance_2D(to)
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


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(controls) == 0:
            set_start_point(x, y, img)
        else:
            add_control(x, y, img)
    elif event == cv2.EVENT_RBUTTONDOWN:
        set_start_point(x, y, img)


def add_control(x, y):
    cv2.circle(img, (x, y), 20, (255, 255, 0), 2)
    cv2.circle(img, (x, y), 2, (255, 255, 0), -1)
    controls.append(Control(x, y))


def set_start_point(x, y):
    cv2.circle(img, (x, y), 20, (255, 100, 0), 2)
    cv2.circle(img, (x, y), 2, (255, 100, 0), -1)
    if len(controls) == 0:
        controls.append(Control(x, y))
    else:
        controls[0] = Control(x, y)


def select_datum_points(event, x, y, flags, param):
    global datum_process_index
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 20, (80, 255, 0), 2)
        cv2.circle(img, (x, y), 2, (80, 255, 0), -1)
        dp = datums_to_process[datum_process_index]
        e = dp['easting']
        n = dp['northing']
        label = dp['label']
        point = DatumPoint(x, y, e, n, label)
        datums.append(point)
        datum_process_index += 1


# def get_image_location_from_map_coords(easting, northing, datums, px_p_m):
#     xs = []
#     ys = []
#     for datum in datums:
#         de = datum.easting - easting
#         dn = datum.northing - northing
#         x = datum.x + (de * px_p_m)
#
#     return(x, y)

def plot_controls_3d(controls):
    """
    Plots a list of Control objects in 3D space.

    Args:
        controls (list): List of Control objects to plot
    """
    if not controls:
        print("No controls to plot.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates and labels
    xs = [c.easting for c in controls]
    ys = [c.northing for c in controls]
    zs = [c.elevation if c.elevation is not None else 0 for c in controls]
    labels = [c.label for c in controls]

    # Plot points
    scatter = ax.scatter(xs, ys, zs, c='r', marker='o', s=50)

    # Add labels if they exist
    for i, label in enumerate(labels):
        if label:
            ax.text(xs[i], ys[i], zs[i], label)

    # Set axis labels
    ax.set_xlabel('X (Local)')
    ax.set_ylabel('Y (Local)')
    ax.set_zlabel('Elevation')

    # Set title
    ax.set_title('Control Points in 3D')

    plt.tight_layout()
    plt.show()


def update_max(old, new):
    if new > old:
        return new
    else:
        return old


def update_min(old, new):
    if new < old:
        return new
    else:
        return old


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
        d = DatumPoint(datum['x'], datum['y'], datum['easting'],
                       datum['northing'], datum_key_string)
        datums.append(d)
    # print(data)
    for datum in datums:
        print(datum)
        x = datum.x
        y = datum.y
        if display_work:
            cv2.circle(img, (x, y), 20, (80, 255, 0), 2)
            cv2.circle(img, (x, y), 2, (80, 255, 0), -1)
            cv2.imshow('datums', img)
        datum.elevation = dem.get_elevation(datum.easting, datum.northing)
        print(f"datum {datum.label}, {datum.easting:.0f}E, {
              datum.northing:.0f}N, {datum.elevation:.2f}m")
        if display_work:
            cv2.waitKey(0)

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
        datums_dict[datum.label] = {
            'x': datum.x, 'y': datum.y, 'easting': datum.easting, 'northing': datum.northing}

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

    img_backup = img.copy()
    img_control_labels = img.copy()

    # step 2 - select controls
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    for index, control in enumerate(controls):
        print(f"for control {index+1}")
        control.get_easting_and_northing_from_datums(datums, dem.px_p_m)
        control.elevation = dem.get_elevation(
            control.easting, control.northing)
        if index != 0:
            dist_x = control.x - controls[index-1].x
            dist_y = control.y - controls[index-1].y
            print(f"px dist: {dist_x}, {dist_y}")
            geo_x = dist_x / dem.px_p_m
            geo_y = dist_y / dem.px_p_m
            print(f"geo dist: {geo_x}, {geo_y}")

   # should add something in here that exports the controls as json that i can load for debugging purposes
    if len(controls) < 1:
        with open('controls_with_points.json', 'r') as file:
            controls_data = json.load(file)

        for c in controls_data["controls"]:
            control = Control(c["x"], c["y"])
            try:
                label = c["label"]
                control.label = label
                control.get_points_from_label()
            except KeyError:
                print("no label found for loaded control")
            finally:
                control.get_easting_and_northing_from_datums(
                    datums, dem.px_p_m)
                control.elevation = dem.get_elevation(
                    control.easting, control.northing)
                controls.append(control)

    max_easting = 0
    max_northing = 0
    max_elevation = 0
    min_easting = 2000000
    min_northing = 7000000

    for control in controls:
        x = control.x
        y = control.y
        if display_work:
            print(f"drawing control {control}, {control.easting:.0f}E, {
                  control.northing:.0f}N, {control.elevation:.2f}m")
            cv2.circle(img_control_labels, (x, y), int(
                control.elevation+20), (0, 50, 250), 2)
            cv2.circle(img_control_labels, (x, y), int(
                control.elevation/5+5), (0, 50, 250), -1)
            cv2.imshow('input control labels', img_control_labels)
            cv2.waitKey(1)
        if control.label is None:
            control.set_label(input("input numbered label for control: "))
            control.get_points_from_label()
        if display_work:
            print(f"control with label {control.label}, worth {
                  control.points} points")
            cv2.circle(img_control_labels, (x, y), int(
                control.elevation+20), (0, 250, 250), 2)
            cv2.circle(img_control_labels, (x, y), int(
                control.elevation/5+5), (0, 250, 250), -1)
        max_easting = update_max(max_easting, control.easting)
        min_easting = update_min(min_easting, control.easting)
        max_northing = update_max(max_northing, control.northing)
        min_northing = update_min(min_northing, control.northing)
        max_elevation = update_max(max_elevation, control.elevation)

    # for control in controls:
    #     print(f"drawing control {control}")
    #     x = control.x
    #     y = control.y
        # r = int(200 * (control.easting - min_easting) / (max_easting - min_easting)) + 55
        # g = int(200 * (control.northing - min_northing) / (max_northing- min_northing)) + 55
        # print(f"max easting: {max_easting}, max_northing: {max_northing}, easting: {control.easting}, northing: {control.northing}, r: {r}, g:{g}\n")
        # r = int(control.elevation / max_elevation * 255)

    # cv2.imshow('map', img)
    # cv2.waitKey(0)

    # construct a 2d array of distances between controls
    # distances = np.zeros((len(controls), len(controls)))
    # for i, f in enumerate(controls):
    #     for j, t in enumerate(controls):
    #         distances[i, j] = f.geo_distance_1D(t)

    # print(distances)

    print("JSON EXPORT")
    export = {}
    export["controls"] = []
    for control in controls:
        export["controls"].append({"x": control.x, "y": control.y, "easting": control.easting,
                                  "northing": control.northing, "label": control.label, "points": control.points})

    print(json.dumps(export))

    G = Graph()

    max_dist = 0

    img_with_paths = img

    # there's a bug in this - it's leaving out one control from each i think
    for orig in controls:
        # print("SEARCHME")
        unvisited_controls = controls.copy()
        unvisited_controls.remove(orig)
        for dest in controls:
            if orig != dest:
                dist = orig.get_graph_weight(dest)
                if dist > max_dist:
                    max_dist = dist
                G.add_edge(orig, dest, dist)
                unvisited_controls.remove(dest)

    print(G)
    for origin_control in G.graph.keys():
        node = G.graph[origin_control]
        print(f"node key: {origin_control}")
        for destination_control in node.keys():
            dist = node[destination_control]
            # B, G, R
            if display_work:
                if dist > max_dist/2:
                    line_color = (0, 255 * (dist/max_dist), 255)
                else:
                    line_color = (0, 255, 255 * (dist/max_dist))
                cv2.line(img_with_paths, (origin_control.x, origin_control.y),
                         (destination_control.x, destination_control.y), line_color, 2)

    G.sort_nodes_by_weight()
    print(G.sorted_in_weight_order)
    plot_controls_3d(controls)

    if display_work:
        cv2.imshow('paths', img_with_paths)
        cv2.waitKey(1)
    #
    # OPTIMISER
    desired_length_km = 10
    desired_height_m = 500

    max_efficiency = 0
    best_path = None

    paths = []

    optimise = True
    run = True

    # turn off to have the loop run significantly (>10x) faster, at the cost of not looking as cool
    display_work_final = True

    # while optimise:
    iterations = 1000
    for i in range(0, iterations):
        print(f"iteration {i} of {iterations}")
        run = True
        unvisited_controls_graph = G.copy()
        # print(unvisited_controls_graph.sorted_in_weight_order)
        unvisited_controls_graph.sort_nodes_by_weight()
        # print(unvisited_controls_graph.sorted_in_weight_order)
        unvisited_controls = unvisited_controls_graph.graph
        start_control = controls[0]
        start_control_backup = unvisited_controls[start_control]
        # initially remove the start control from the list of unvisited controls so we cannot prematurely exit the search. i don't care about a really efficient path that only gets me like 5 pointsprint(unvisited_controls)
        current_control = start_control
        next_control = None
        fresh_map = img_backup.copy()
        path = Path()
        path.controls.append(start_control)
        removed_controls = []
        while run:
            try:
                ctrls = list(unvisited_controls[current_control].keys())
                weights = list(unvisited_controls[current_control].values())
            except Exception as e:
                print(e)
                print(f"removed: {removed_controls}")
                quit()
            if path.points <= 1200:  # do not allow the algorithm to select the home control until it has gained at least 120 points
                valid_control_found = False
                i = 0
                while not valid_control_found:
                    # next_control = random.choices(ctrls, weights)[0]
                    print(f"current control: {current_control}, next control {next_control}")
                    next_control = unvisited_controls_graph.sorted_in_weight_order[current_control][i]
                    i += 1
                    if next_control.label != "0":
                        valid_control_found = True
            else:
                # next_control = random.choices(ctrls, weights)[0]
                next_control = unvisited_controls_graph.sorted_in_weight_order[current_control][0]

            path.controls.append(next_control)
            path.points += next_control.points
            path.total_distance += G.graph[current_control][next_control]
            el = current_control.get_elevation_difference(next_control)
            path.total_elevation += el
            # print(f"elevation difference between {
            #       current_control} and {next_control} is {el}")
            path.distance_2d += current_control.geo_distance_2D(next_control)
            if display_work_final:
                cv2.line(fresh_map, (current_control.x, current_control.y),
                         (next_control.x, next_control.y), (100, 0, 255), 2)
            if current_control != start_control:
                unvisited_controls_graph.sorted_in_weight_order.pop(current_control)
                for control in unvisited_controls_graph.sorted_in_weight_order.keys():
                    unvisited_controls_graph.sorted_in_weight_order[control].remove(current_control)
                # unvisited_controls.pop(current_control)
                # for control in unvisited_controls.keys():  # EW stinks
                #     unvisited_controls[control].pop(current_control)
                #     # print(f"removed path to {current_control} from {control}")
                # removed_controls.append(current_control)
                # print(f"REMOVED: {current_control}")
            current_control = next_control

            if next_control.label == "0":
                print("path finished")
                print(f"total distance: {path.total_distance}")
                print(f"points: {path.points}")
                path_efficiency = path.points / path.total_distance
                print(f"path efficiency: {path_efficiency}")
                if path_efficiency > max_efficiency:
                    max_efficiency = path_efficiency
                    print("new most efficient path found!")
                    best_path = path
                # k = cv2.waitKey(0) & 0xFF
                # if k == 27:
                #     optimise = False
                #     print("exiting")
                # else:
                paths.append(path)
                run = False
                print("resetting")
                break
        if display_work_final:
            cv2.imshow('optimiser', fresh_map)
            cv2.waitKey(1)

    best_path_img = img_backup.copy()
    previous_control = None
    path_len = len(best_path.controls)
    r = 0
    g = 255
    step = 255 / path_len
    for control in best_path.controls:
        if previous_control:
            cv2.line(best_path_img, (previous_control.x, previous_control.y),
                     (control.x, control.y), (0, g, r), 5)
            r = r + step
            g = g - step
            previous_control = control
            cv2.imshow('best path', best_path_img)
            cv2.waitKey(1)
        else:
            previous_control = control
    print("PROGRAM FINISHED")
    print("BEST PATH FOUND:")
    print(f"POINTS: {best_path.points}")
    print(f"TOTAL DIST: {best_path.total_distance}")
    print(f"TOTAL VERT: {best_path.total_elevation}m")
    print(f"TOTAL 2D DIST: {best_path.distance_2d/1000:.2f}km")
    print(f"EFFICIENCY: {best_path.points / best_path.total_distance}")
    cv2.waitKey(0)


main()
