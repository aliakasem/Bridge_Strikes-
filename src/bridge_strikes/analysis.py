"""Descriptive summaries and per-location next-month risk estimates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PRIOR_ALPHA, PRIOR_BETA

DESCRIPTIVE_COLS = ["borough", "main_street", "cross_street", "truck_route_dataflag", "on_truck_route_geom"]


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Inclusive count of calendar months from start to end."""
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def _strikes_last_12m(df: pd.DataFrame) -> pd.Series:
    cutoff = df["date"].max() - pd.DateOffset(years=1)
    return df[df["date"] >= cutoff].groupby("location_id").size().rename("strikes_last_12m")


def _descriptive(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DESCRIPTIVE_COLS if c in df.columns]
    # groupby().first() takes the first non-null value per column
    return df.sort_values("date").groupby("location_id")[cols].first()


def aggregate_summaries(strikes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (grouped counts by route/overpass flags, location hotspot table)."""
    df = pd.DataFrame(strikes.drop(columns="geometry", errors="ignore"))

    grouped = (
        df.groupby(["truck_route_dataflag", "struck_overpass", "on_truck_route_geom"])
        .size()
        .reset_index(name="count")
    )
    grouped["percentage"] = 100.0 * grouped["count"] / len(df)

    hotspots = pd.concat(
        [
            _descriptive(df),
            df.groupby("location_id").size().rename("total_strikes"),
            _strikes_last_12m(df),
            df.groupby("location_id")["dist_to_route_m"].median().rename("median_dist_to_route_m"),
        ],
        axis=1,
    ).fillna({"strikes_last_12m": 0})
    hotspots["strikes_last_12m"] = hotspots["strikes_last_12m"].astype(int)
    hotspots = hotspots.reset_index().sort_values(["total_strikes", "strikes_last_12m"], ascending=False)
    hotspots["rank"] = np.arange(1, len(hotspots) + 1)
    return grouped, hotspots


def estimate_next_month_probabilities(strikes: pd.DataFrame) -> pd.DataFrame:
    """Per-location Gamma-Poisson estimate of P(>=1 strike next month).

    Each location's monthly rate is the posterior mean
        lambda = (strikes + alpha) / (months_exposed + beta)
    where months_exposed is the full study window (first to last month in the
    data). Using only the months in which a location was struck would make the
    rate at least ~1 for every location and inflate the probabilities.
    """
    df = pd.DataFrame(strikes.drop(columns="geometry", errors="ignore"))
    months_exposed = _months_between(df["month_start"].min(), df["month_start"].max())

    per_loc = pd.concat(
        [
            _descriptive(df),
            df.groupby("location_id").agg(
                first_strike=("date", "min"),
                last_strike=("date", "max"),
                latitude=("latitude", "median"),
                longitude=("longitude", "median"),
                dist_to_route_m=("dist_to_route_m", "median"),
                total_strikes=("date", "size"),
                months_with_strikes=("month_start", "nunique"),
            ),
            _strikes_last_12m(df),
        ],
        axis=1,
    ).fillna({"strikes_last_12m": 0}).reset_index()
    per_loc["strikes_last_12m"] = per_loc["strikes_last_12m"].astype(int)

    per_loc["months_exposed"] = months_exposed
    per_loc["lambda_per_month"] = (per_loc["total_strikes"] + PRIOR_ALPHA) / (months_exposed + PRIOR_BETA)
    per_loc["p_next_month"] = 1.0 - np.exp(-per_loc["lambda_per_month"])

    return per_loc.sort_values(
        ["p_next_month", "strikes_last_12m", "total_strikes"], ascending=False
    ).reset_index(drop=True)
