# NYC Bridge-Strike Risk Analysis

Estimates where and how often over-height vehicles strike bridges and overpasses in New York City, using NYC Open Data. The pipeline:

1. Downloads bridge-strike records and the NYC truck-route network (cached locally after the first run).
2. Cleans the strike data: valid dates, coordinates inside NYC, and removal of locations that have since been fixed.
3. Computes each strike's distance to the nearest designated truck route.
4. Builds a location hotspot table and a per-location probability of a strike next month.
5. Fits city-wide Poisson and Negative Binomial monthly count models and compares them by AIC.
6. Writes CSV/Parquet/GeoPackage outputs for a dashboard, plus figures.

## Repository structure

```
bridge-strike-risk/
├── src/bridge_strikes/
│   ├── config.py      # constants: CRS, bounds, URLs, exclusions, priors, paths
│   ├── data.py        # download with retries + local cache
│   ├── cleaning.py    # strike and truck-route cleaning, location IDs
│   ├── spatial.py     # nearest truck-route distance
│   ├── analysis.py    # hotspots and per-location risk
│   ├── models.py      # Poisson vs Negative Binomial count models
│   ├── viz.py         # figures
│   └── pipeline.py    # end-to-end run + CLI
├── notebooks/
│   └── bridge_strike_analysis.ipynb
├── tests/
│   └── test_pipeline.py   # synthetic-data smoke tests (no network)
├── data/raw/          # cached source CSVs (git-ignored)
├── outputs/           # generated tables and figures (git-ignored)
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

This installs the package in editable mode, so the notebook and CLI both import from `src/`.

## Usage

```bash
python -m bridge_strikes.pipeline               # uses cached data if present
python -m bridge_strikes.pipeline --refresh     # re-download from NYC Open Data
python -m bridge_strikes.pipeline --output-dir path/to/outputs
pytest                                          # run tests
```

Or open `notebooks/bridge_strike_analysis.ipynb` and select the `.venv` kernel.

If a download fails (NYC Open Data's API has been moving between SODA2 and SODA3), download the CSVs manually and save them as `data/raw/bridge_strikes.csv` and `data/raw/truck_routes.csv`. `--insecure` disables TLS verification and is only meant for networks that intercept HTTPS.

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `strikes_enriched.csv` / `.parquet` / `.gpkg` | Cleaned strikes with location ID and truck-route distance |
| `summary_aggregates.csv` | Counts by truck-route flag, overpass flag, and geometric on-route status |
| `hotspots.csv` | Location ranking by total and last-12-month strikes |
| `location_risk_next_month.csv` | Per-location monthly rate λ and P(≥1 strike next month) |
| `model_comparison.csv` | Poisson vs NB forecast, AIC/BIC, trend coefficient |
| `monthly_counts.csv` | City-wide strikes per month (zero months included) |
| `figures/*.png` | Model comparison, monthly series, top-10 locations |

## Method notes

**Location risk.** Each location's monthly rate uses a Gamma(0.5, 1) prior:
λ = (strikes + 0.5) / (months in study window + 1), and P(≥1 strike) = 1 − e^(−λ). The exposure is the full study window, not only the months in which the location was struck.

**City-wide models.** Monthly counts, including zero months, are regressed on a linear time trend. Both Poisson and NB2 are fit by maximum likelihood with statsmodels. The NB dispersion α is estimated. The model with the lower AIC is reported as best. The NB next-month probability uses the NB zero probability (1 + αλ)^(−1/α).

**Exclusions.** BELT PARKWAY & 17TH AVENUE is removed because the site was fixed in 2022 and its history no longer reflects current risk. Edit `EXCLUDED_LOCATIONS` in `config.py` to change this.

## Limitations

- Only locations with at least one recorded strike appear in the risk table. Never-struck bridges are not scored.
- Every location is assumed to be exposed for the whole study window, which ignores structures built, raised, or re-signed during that period.
- Truck-route proximity is descriptive. It is not used as a model covariate.
- The linear time trend is a simple baseline; it does not model seasonality.

## Data sources

- Bridge strikes: NYC Open Data, dataset `jdn9-td9w`
- Truck routes: NYC Open Data, dataset `jjja-shxy`
