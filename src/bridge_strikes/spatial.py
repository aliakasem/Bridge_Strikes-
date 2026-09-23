"""Spatial enrichment: distance from each strike to the nearest truck route."""
from __future__ import annotations

import geopandas as gpd

from .config import CRS_METRIC, CRS_WGS84, ON_ROUTE_FT

FT_TO_M = 0.3048


def add_distance_to_truck_route(strikes: gpd.GeoDataFrame, routes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add dist_to_route_ft, dist_to_route_m and on_truck_route_geom.

    Distances are computed in EPSG:2263 (US feet). Raises instead of guessing
    if either input is empty.
    """
    if strikes.empty or routes.empty:
        raise ValueError("Both strikes and routes must be non-empty to compute distances")

    strikes_ft = strikes.to_crs(CRS_METRIC)
    routes_ft = routes[[routes.geometry.name]].to_crs(CRS_METRIC)

    joined = gpd.sjoin_nearest(strikes_ft, routes_ft, how="left", distance_col="dist_to_route_ft")
    # Ties (equidistant routes) duplicate a strike; keep one row per strike.
    joined = joined[~joined.index.duplicated(keep="first")].drop(columns="index_right", errors="ignore")

    joined["dist_to_route_m"] = joined["dist_to_route_ft"] * FT_TO_M
    joined["on_truck_route_geom"] = joined["dist_to_route_ft"] <= ON_ROUTE_FT
    return joined.to_crs(CRS_WGS84)
