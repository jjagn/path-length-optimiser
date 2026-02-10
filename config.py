# config.py — Central configuration for the path length optimiser

# Map and DEM paths
MAP_IMAGE_PATH = "./map.png"
DEM_FOLDER_PATH = "./chc-dem"
MOSAIC_VRT_PATH = "./mosaic.vrt"

# Datum and control data files
DATUMS_FILE = "datums2.json"
CONTROLS_FILE = "controls_with_points.json"

# Optimiser parameters
MAX_DISTANCE_KM = 14.0
MAX_CLIMB_M = 800.0
MIN_POINTS_BEFORE_RETURN = 500
ITERATIONS = 1000
TOP_K_CANDIDATES = 5
CLIMB_COST_FACTOR = 20.0  # alpha: meters-of-distance-equivalent per meter of climb

# Optimisation mode: "max_points" or "max_efficiency"
#   max_points    — maximise total points collected within budget
#   max_efficiency — maximise points per meter of distance
OPTIMISE_MODE = "max_points"

# 2-opt parameters
TWO_OPT_MAX_ITERATIONS = 100

# Visualization
DISPLAY_WORK = True      # show each path as it is calculated
DISPLAY_FINAL = True     # show the best path at the end
