import json
import cv2
from osgeo import gdal

import config
from models import DatumPoint, Control
from dem import DEM
from georef import AffineGeoref, georeference_controls
from graph import build_graph
from optimiser import edge_cost, optimise
from visualise import draw_controls, draw_best_path, plot_controls_3d, show_result, LiveDisplay


def load_datums(path):
    with open(path, "r") as f:
        data = json.load(f)
    datums = []
    for label, vals in data.items():
        datums.append(DatumPoint(
            vals["x"], vals["y"],
            easting=vals["easting"], northing=vals["northing"],
            label=label,
        ))
    return datums


def load_controls_from_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    controls = []
    for c in data["controls"]:
        controls.append(Control(
            c["x"], c["y"],
            label=c.get("label"),
            points=c.get("points"),
        ))
    return controls


def main():
    gdal.UseExceptions()

    # Load DEM
    print("Loading DEM...")
    dem = DEM()
    dem.load_dem_files(config.DEM_FOLDER_PATH)

    # Load datum points and fit affine georeferencing
    print("Loading datums and fitting affine transform...")
    datums = load_datums(config.DATUMS_FILE)
    georef = AffineGeoref()
    georef.fit(datums)
    georef.report()

    # Load controls
    print("Loading controls...")
    controls = load_controls_from_file(config.CONTROLS_FILE)

    # Georeference controls using affine transform
    georeference_controls(controls, georef)

    # Look up elevation for each control
    for ctrl in controls:
        ctrl.elevation = dem.get_elevation(ctrl.easting, ctrl.northing)

    # Find home control (label "0")
    home = None
    for ctrl in controls:
        if ctrl.label == "0":
            home = ctrl
            break
    if home is None:
        raise ValueError("No home control (label '0') found in controls")

    print(f"Home control: {home} at ({home.easting:.0f}E, {home.northing:.0f}N)")
    print(f"Loaded {len(controls)} controls")

    # Build graph with cost function
    def cost_fn(a, b):
        return edge_cost(a, b, config.CLIMB_COST_FACTOR)

    print("Building graph...")
    graph = build_graph(controls, cost_fn)

    # Plot controls in 3D
    if config.DISPLAY_WORK:
        plot_controls_3d(controls)

    # Set up live display callback if DISPLAY_WORK is enabled
    img = None
    live_display = None
    if config.DISPLAY_WORK:
        img = cv2.imread(config.MAP_IMAGE_PATH)
        if img is not None:
            live_display = LiveDisplay(img, controls)
        else:
            print(f"Warning: could not load {config.MAP_IMAGE_PATH} for live display")

    # Run optimiser
    max_distance_m = config.MAX_DISTANCE_KM * 1000
    mode = getattr(config, "OPTIMISE_MODE", "max_efficiency")
    print(f"\nRunning optimiser ({config.ITERATIONS} iterations, mode={mode})...")
    print(f"  Budget: {config.MAX_DISTANCE_KM} km, {config.MAX_CLIMB_M} m climb")
    print(f"  Min points before return: {config.MIN_POINTS_BEFORE_RETURN}")
    print(f"  Top-K candidates: {config.TOP_K_CANDIDATES}")

    best_path = optimise(
        graph=graph,
        home_id=home.id,
        max_distance_m=max_distance_m,
        min_points_before_return=config.MIN_POINTS_BEFORE_RETURN,
        iterations=config.ITERATIONS,
        top_k=config.TOP_K_CANDIDATES,
        climb_factor=config.CLIMB_COST_FACTOR,
        two_opt_iters=config.TWO_OPT_MAX_ITERATIONS,
        mode=mode,
        on_path=live_display,
    )

    if live_display is not None:
        live_display.close()

    if best_path is None:
        print("No valid path found.")
        return

    # Print results
    print("\n" + "=" * 50)
    print("BEST PATH FOUND:")
    print(f"  Points:     {best_path.points}")
    print(f"  Distance:   {best_path.distance_2d / 1000:.2f} km")
    print(f"  Climb:      {best_path.total_climb:.0f} m")
    print(f"  Efficiency: {best_path.points / best_path.distance_2d:.4f} pts/m")
    print(f"  Route:      {' -> '.join(str(c) for c in best_path.controls)}")
    print("=" * 50)

    # Export route
    export = {"route": []}
    for ctrl in best_path.controls:
        export["route"].append({
            "x": ctrl.x, "y": ctrl.y,
            "easting": ctrl.easting, "northing": ctrl.northing,
            "label": ctrl.label, "points": ctrl.points,
        })
    with open("best_route.json", "w") as f:
        json.dump(export, f, indent=2)
    print("Route exported to best_route.json")

    # Visualize
    if config.DISPLAY_FINAL:
        if img is None:
            img = cv2.imread(config.MAP_IMAGE_PATH)
        if img is not None:
            show_result(img, best_path)
        else:
            print(f"Could not load map image from {config.MAP_IMAGE_PATH}")


if __name__ == "__main__":
    main()
