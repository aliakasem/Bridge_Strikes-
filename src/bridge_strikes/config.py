"""Project-wide constants and default paths."""
from pathlib import Path

# Coordinate reference systems
CRS_WGS84 = "EPSG:4326"   # lat/lon, for storage and mapping
CRS_METRIC = "EPSG:2263"  # NY State Plane Long Island (US feet), for distances

# NYC bounding box used to drop mis-geocoded records
NYC_LAT_MIN, NYC_LAT_MAX = 40.4, 40.95
NYC_LON_MIN, NYC_LON_MAX = -74.3, -73.7

# A strike within this distance of a truck-route line counts as "on route"
ON_ROUTE_FT = 50

# NYC Open Data sources. The /resource/ (SODA) endpoint returns only 1,000 rows
# unless $limit is set, so it is raised explicitly.
URL_STRIKE = "https://data.cityofnewyork.us/resource/jdn9-td9w.csv?$limit=100000"
URL_TRUCK_ROUTE = (
    "https://data.cityofnewyork.us/api/views/jjja-shxy/rows.csv"
    "?date=20240125&accessType=DOWNLOAD"
)

# Locations removed from the analysis: (main_street, cross_street) substrings.
# BELT PARKWAY & 17TH AVENUE was physically fixed in 2022, so its history
# no longer reflects current risk.
EXCLUDED_LOCATIONS = [("BELT PARKWAY", "17TH AVENUE")]

# Gamma(shape=alpha, rate=beta) prior for per-location monthly strike rates
PRIOR_ALPHA = 0.5
PRIOR_BETA = 1.0

# Default paths, resolved relative to the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
