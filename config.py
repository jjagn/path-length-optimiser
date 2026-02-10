# config.py — Central configuration for the path length optimiser

# Map and DEM paths
MAP_IMAGE_PATH = "./map.png"
DEM_FOLDER_PATH = "./chc-dem"
MOSAIC_VRT_PATH = "./mosaic.vrt"

# Datum and control data files
DATUMS_FILE = "datums2.json"
CONTROLS_FILE = "controls_with_points.json"

# Optimiser parameters
MAX_DISTANCE_KM = 10.0
MAX_CLIMB_M = 500.0
MIN_POINTS_BEFORE_RETURN = 120
ITERATIONS = 1000
TOP_K_CANDIDATES = 5
CLIMB_COST_FACTOR = 10.0  # alpha: meters-of-distance-equivalent per meter of climb

# 2-opt parameters
TWO_OPT_MAX_ITERATIONS = 100

# Visualization
DISPLAY_WORK = False
DISPLAY_FINAL = True
