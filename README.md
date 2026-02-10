# Path Length Optimiser

A route optimisation tool for [rogaine](https://en.wikipedia.org/wiki/Rogaining) orienteering events. Given a scanned rogaine map, DEM elevation data, and control point locations, it finds the most point-efficient route under distance and climb constraints.

## How it works

1. **Georeferencing** — Known datum points on the map are matched to real-world coordinates (NZTM). A least-squares affine transform converts all control pixel positions to easting/northing.
2. **Elevation lookup** — DEM tiles are mosaicked and each control is assigned an elevation from the cached raster.
3. **Graph construction** — A complete weighted graph is built between all controls, where edge cost = 2D distance + (climb factor × uphill climb).
4. **Optimisation** — A multi-start budget-aware greedy algorithm selects controls by points-per-cost efficiency, ensuring feasibility to return home at each step. Routes are then improved with unvisited control insertion and 2-opt edge swaps.

## Project structure

```
plo/
├── main.py              # Entry point
├── config.py            # All tuneable parameters
├── models.py            # Point, DatumPoint, Control, Path
├── graph.py             # Graph class and builder
├── dem.py               # DEM loading and elevation lookup
├── georef.py            # Affine transform georeferencing
├── optimiser.py         # Greedy + 2-opt optimiser
├── visualise.py         # OpenCV/matplotlib drawing helpers
├── tests/               # Pytest test suite (87 tests)
├── archive/             # Legacy scripts
├── chc-dem/             # DEM GeoTIFF tiles
├── controls*.json       # Control point data
├── datums*.json         # Datum reference points
├── requirements.txt     # Python dependencies
└── map.png              # Scanned rogaine map
```

## Setup

Requires Python 3.13+ and [GDAL](https://gdal.org/) installed via Homebrew:

```bash
brew install gdal
```

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Edit `config.py` to set your parameters, then run:

```bash
python main.py
```

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `MAX_DISTANCE_KM` | 10.0 | Maximum route distance budget (km) |
| `MAX_CLIMB_M` | 500.0 | Maximum cumulative climb budget (m) |
| `MIN_POINTS_BEFORE_RETURN` | 120 | Minimum points before allowing return to start |
| `ITERATIONS` | 1000 | Number of multi-start optimiser runs |
| `TOP_K_CANDIDATES` | 5 | Candidates considered at each greedy step |
| `CLIMB_COST_FACTOR` | 10.0 | Meters-of-distance-equivalent per meter of climb |

### Input data

- **Datum points** (`datums2.json`) — Map features with known pixel (x, y) and real-world (easting, northing) coordinates. At least 3 recommended for accurate affine georeferencing.
- **Controls** (`controls_with_points.json`) — Control point pixel positions and labels. Labels encode point values (e.g. label `"53"` → 50 points). The home/start control should have label `"0"`.
- **DEM tiles** (`chc-dem/`) — GeoTIFF elevation files covering the event area.

### Output

- Best route printed to console with total points, distance, climb, and efficiency
- Route exported to `best_route.json`
- Visual overlay on the map image (if `DISPLAY_FINAL = True`)

## Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```
