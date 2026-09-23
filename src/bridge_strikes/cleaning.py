"""Clean and validate the raw datasets."""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt

from .config import (
    CRS_WGS84,
    EXCLUDED_LOCATIONS,
    NYC_LAT_MAX,
    NYC_LAT_MIN,
    NYC_LON_MAX,
    NYC_LON_MIN,
)

log = logging.getLogger(__name__)


def _column_or_blank(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series("", index=df.index)


def yes_no(series: pd.Series) -> pd.Series:
    """Normalize free-text flags to 'YES'/'NO' (anything but 'yes' is 'NO')."""
    is_yes = series.astype(str).str.strip().str.upper().eq("YES")
    return pd.Series(np.where(is_yes, "YES", "NO"), index=series.index)


def make_location_id(df: pd.DataFrame) -> pd.Series:
    """Stable location key: bridge BIN when available, else rounded coordinates."""
    if "bin" in df.columns:
        bins = pd.to_numeric(df["bin"], errors="coerce").round().astype("Int64")
    else:
        bins = pd.Series(pd.NA, index=df.index, dtype="Int64")
    geo = "GEO_" + df["latitude"].round(4).astype(str) + "|" + df["longitude"].round(4).astype(str)
    return ("BIN_" + bins.astype(str)).where(bins.notna(), geo)


def clean_strikes(raw: pd.DataFrame) -> gpd.GeoDataFrame:
    """Normalize columns, parse dates, keep NYC coordinates, drop excluded sites."""
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_bad_dates = int(df["date"].isna().sum())
    df = df[df["date"].notna()].copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dow"] = df["date"].dt.day_name()
    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    in_nyc = df["latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX) & df["longitude"].between(NYC_LON_MIN, NYC_LON_MAX)
    log.info(
        "Strikes: %d rows with valid dates (%d dropped), %d inside NYC bounds, %d outside",
        len(df), n_bad_dates, int(in_nyc.sum()), int((~in_nyc).sum()),
    )
    df = df[in_nyc].copy()

    main = _column_or_blank(df, "main_street").astype(str).str.upper()
    cross = _column_or_blank(df, "cross_street").astype(str).str.upper()
    for main_sub, cross_sub in EXCLUDED_LOCATIONS:
        mask = main.str.contains(main_sub, regex=False) & cross.str.contains(cross_sub, regex=False)
        log.info("Excluded %s & %s: %d records", main_sub, cross_sub, int(mask.sum()))
        df, main, cross = df[~mask], main[~mask], cross[~mask]

    df = df.copy()
    df["truck_route_dataflag"] = yes_no(_column_or_blank(df, "truck_route"))
    df["struck_overpass"] = yes_no(_column_or_blank(df, "struck_overpass"))
    df["location_id"] = make_location_id(df)

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=CRS_WGS84,
    )


def _safe_wkt(value):
    try:
        return wkt.loads(value)
    except Exception:  # noqa: BLE001 - unparseable geometry becomes null
        return None


def clean_truck_routes(raw: pd.DataFrame) -> gpd.GeoDataFrame:
    """Parse WKT route geometry. Rows with unparseable geometry are dropped."""
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    geom_col = next((c for c in ("the_geom", "geometry", "geom") if c in df.columns), None)
    if geom_col is None:
        raise ValueError(f"No geometry column in truck-route data. Columns: {list(df.columns)}")

    geoms = df[geom_col].map(_safe_wkt)
    n_bad = int(geoms.isna().sum())
    if n_bad:
        log.warning("Dropped %d truck-route rows with unparseable geometry", n_bad)
    keep = geoms.notna()
    routes = gpd.GeoDataFrame(df.drop(columns=[geom_col])[keep], geometry=geoms[keep].tolist(), crs=CRS_WGS84)
    if routes.empty:
        raise ValueError("Truck-route dataset has no valid geometries")
    return routes
