"""Smoke tests on small synthetic data (no network needed)."""
import numpy as np
import pandas as pd
import pytest

from bridge_strikes.analysis import estimate_next_month_probabilities
from bridge_strikes.pipeline import analyze, write_outputs


@pytest.fixture
def raw_data():
    rng = np.random.default_rng(0)
    n = 300
    sites = [
        (40.70, -73.95, "BQE", "ATLANTIC AVE", 1001),
        (40.80, -73.93, "FDR DRIVE", "E 96 ST", 1002),
        (40.60, -74.00, "BELT PARKWAY", "17TH AVENUE", 1003),  # excluded site
        (40.75, -73.85, "GRAND CENTRAL PKWY", "MAIN ST", None),
    ]
    idx = rng.choice(len(sites), size=n, p=[0.5, 0.3, 0.1, 0.1])
    days = rng.integers(0, 365 * 5, size=n)
    strikes = pd.DataFrame({
        "Date": (pd.Timestamp("2019-01-01") + pd.to_timedelta(days, unit="D")).astype(str),
        "Latitude": [sites[i][0] for i in idx],
        "Longitude": [sites[i][1] for i in idx],
        "Main_Street": [sites[i][2] for i in idx],
        "Cross_Street": [sites[i][3] for i in idx],
        "BIN": [sites[i][4] for i in idx],
        "Borough": "BROOKLYN",
        "Truck_Route": rng.choice(["Yes", "No"], size=n),
        "Struck_Overpass": rng.choice(["YES", "NO"], size=n),
    })
    strikes.loc[0, "Date"] = "not a date"
    strikes.loc[1, "Latitude"] = 10.0  # outside NYC
    routes = pd.DataFrame({
        "the_geom": ["LINESTRING (-74.05 40.70, -73.80 40.70)",
                     "LINESTRING (-73.95 40.60, -73.95 40.85)",
                     "garbage"],
    })
    return strikes, routes


def test_pipeline_runs_end_to_end(raw_data, tmp_path):
    result = analyze(*raw_data)
    assert not result.strikes["main_street"].str.contains("BELT PARKWAY").any()
    assert result.strikes["dist_to_route_ft"].notna().all()
    assert result.best_model in {"Poisson", "Negative Binomial"}
    assert result.comparison["aic"].notna().all()
    assert result.location_risk["p_next_month"].between(0, 1).all()
    write_outputs(result, tmp_path)
    assert (tmp_path / "location_risk_next_month.csv").exists()
    assert (tmp_path / "figures" / "top_10_risk_locations.png").exists()


def test_single_strike_location_is_not_inflated():
    """A location struck once in a 60-month window should have low risk."""
    dates = pd.date_range("2019-01-01", periods=60, freq="MS")
    df = pd.DataFrame({
        "location_id": ["A"] * 59 + ["B"],
        "date": list(dates[:59]) + [dates[30]],
        "latitude": 40.7, "longitude": -73.9, "dist_to_route_m": 5.0,
        "borough": "X", "main_street": "M", "cross_street": "C",
        "truck_route_dataflag": "YES", "on_truck_route_geom": True,
    })
    df["month_start"] = df["date"]
    risk = estimate_next_month_probabilities(df).set_index("location_id")
    assert risk.loc["B", "p_next_month"] < 0.05
    assert risk.loc["A", "p_next_month"] > 0.5
