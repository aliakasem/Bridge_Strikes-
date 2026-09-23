"""End-to-end pipeline and command-line entry point.

Run from the repo root:
    python -m bridge_strikes.pipeline            # uses cached data if present
    python -m bridge_strikes.pipeline --refresh  # re-download from NYC Open Data
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import aggregate_summaries, estimate_next_month_probabilities  # noqa: E402
from .cleaning import clean_strikes, clean_truck_routes  # noqa: E402
from .config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, ON_ROUTE_FT  # noqa: E402
from .data import load_data  # noqa: E402
from .models import fit_count_models  # noqa: E402
from .spatial import add_distance_to_truck_route  # noqa: E402
from .viz import plot_model_comparison, plot_monthly_series, plot_top_risk_locations  # noqa: E402

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    strikes: gpd.GeoDataFrame
    grouped: pd.DataFrame
    hotspots: pd.DataFrame
    location_risk: pd.DataFrame
    comparison: pd.DataFrame
    best_model: str | None
    monthly: pd.DataFrame


def analyze(strikes_raw: pd.DataFrame, routes_raw: pd.DataFrame) -> PipelineResult:
    """Run every analysis step on already-loaded raw data (no file I/O)."""
    strikes = clean_strikes(strikes_raw)
    routes = clean_truck_routes(routes_raw)
    strikes = add_distance_to_truck_route(strikes, routes)
    grouped, hotspots = aggregate_summaries(strikes)
    location_risk = estimate_next_month_probabilities(strikes)
    comparison, best, monthly = fit_count_models(strikes)
    return PipelineResult(strikes, grouped, hotspots, location_risk, comparison, best, monthly)


def write_outputs(result: PipelineResult, output_dir: Path) -> None:
    """Write tables (for the dashboard) and figures."""
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "strikes_enriched.csv": pd.DataFrame(result.strikes.drop(columns="geometry")),
        "summary_aggregates.csv": result.grouped,
        "hotspots.csv": result.hotspots,
        "location_risk_next_month.csv": result.location_risk,
        "model_comparison.csv": result.comparison,
        "monthly_counts.csv": result.monthly,
    }
    for name, df in tables.items():
        df.to_csv(output_dir / name, index=False)
        log.info("Wrote %s", output_dir / name)

    try:
        tables["strikes_enriched.csv"].to_parquet(output_dir / "strikes_enriched.parquet", index=False)
    except ImportError:
        log.info("pyarrow not installed; skipping parquet output")
    try:
        result.strikes.to_file(output_dir / "strikes_enriched.gpkg", layer="strikes_enriched", driver="GPKG")
    except Exception as exc:  # noqa: BLE001 - optional output
        log.warning("Could not write GeoPackage: %s", exc)

    figures = {
        "model_comparison.png": plot_model_comparison(result.comparison),
        "monthly_strikes.png": plot_monthly_series(result.monthly),
        "top_10_risk_locations.png": plot_top_risk_locations(result.location_risk, top_n=10),
    }
    for name, fig in figures.items():
        fig.savefig(fig_dir / name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log.info("Wrote %s", fig_dir / name)


def print_summary(result: PipelineResult) -> None:
    s = result.strikes
    n_on_route = int(s["on_truck_route_geom"].sum())
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Strike records analysed:        {len(s)}")
    print(f"Distinct locations:             {s['location_id'].nunique()}")
    print(f"Within {ON_ROUTE_FT} ft of a truck route:  {n_on_route} ({100 * n_on_route / max(len(s), 1):.1f}%)")
    print(f"Best model (lowest AIC):        {result.best_model}")
    print("\nModel comparison:")
    cols = [c for c in ["model_type", "expected_strikes", "probability_strike", "nb_alpha", "aic", "bic"]
            if c in result.comparison]
    print(result.comparison[cols].to_string(index=False, float_format="%.3f"))
    print("\nTop 10 locations by P(strike next month):")
    cols = ["location_id", "borough", "main_street", "cross_street", "total_strikes",
            "strikes_last_12m", "p_next_month"]
    print(result.location_risk.head(10)[[c for c in cols if c in result.location_risk]]
          .to_string(index=False, float_format="%.3f"))


def run_pipeline(data_dir: Path = DEFAULT_DATA_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR,
                 refresh: bool = False, insecure: bool = False, save: bool = True) -> PipelineResult:
    """Load data, analyse it, and optionally write outputs."""
    log.info("Starting pipeline at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    strikes_raw, routes_raw = load_data(data_dir, refresh=refresh, insecure=insecure)
    result = analyze(strikes_raw, routes_raw)
    if save:
        write_outputs(result, output_dir)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="NYC bridge-strike risk pipeline")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="where raw CSVs are cached")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="where outputs are written")
    parser.add_argument("--refresh", action="store_true", help="re-download data even if cached")
    parser.add_argument("--insecure", action="store_true",
                        help="disable TLS verification (last resort for intercepting networks)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    result = run_pipeline(args.data_dir, args.output_dir, args.refresh, args.insecure)
    print_summary(result)
    print(f"\nOutputs written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
