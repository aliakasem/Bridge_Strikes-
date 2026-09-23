"""Download NYC Open Data CSVs, with a local cache for reproducibility."""
from __future__ import annotations

import io
import logging
import ssl
import time
import urllib.request
from pathlib import Path

import certifi
import pandas as pd

from .config import URL_STRIKE, URL_TRUCK_ROUTE

log = logging.getLogger(__name__)

STRIKES_CACHE = "bridge_strikes.csv"
ROUTES_CACHE = "truck_routes.csv"


def fetch_csv(url: str, retries: int = 3, delay: int = 5, insecure: bool = False) -> pd.DataFrame | None:
    """Download a CSV over HTTPS. Returns None if every attempt fails.

    Certificates are verified against certifi's CA bundle. ``insecure=True``
    disables verification and should only be used as a last resort on a
    network that intercepts TLS.
    """
    ctx = ssl.create_default_context(cafile=certifi.where())
    if insecure:
        log.warning("TLS certificate verification is DISABLED for %s", url)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    request = urllib.request.Request(url, headers={"User-Agent": "bridge-strike-risk"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, context=ctx, timeout=120) as resp:
                content = resp.read()
            df = pd.read_csv(io.BytesIO(content), low_memory=False)
            log.info("Downloaded %d rows from %s", len(df), url)
            return df
        except Exception as exc:  # noqa: BLE001 - report and retry any failure
            log.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(delay)
    return None


def load_dataset(url: str, cache_path: Path, refresh: bool = False, insecure: bool = False) -> pd.DataFrame:
    """Return a dataset from the local cache, or download it and cache it.

    Order: cached file (unless ``refresh``), then download, then a stale
    cache. Raises if nothing is available, so the pipeline never runs on
    empty or made-up data.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        log.info("Using cached data: %s", cache_path)
        return pd.read_csv(cache_path, low_memory=False)

    df = fetch_csv(url, insecure=insecure)
    if df is not None and not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        return df

    if cache_path.exists():
        log.warning("Download failed; falling back to cached copy %s", cache_path)
        return pd.read_csv(cache_path, low_memory=False)

    raise RuntimeError(
        f"Could not download {url} and no cached copy exists at {cache_path}. "
        "Download the CSV manually from NYC Open Data and save it at that path."
    )


def load_data(data_dir: Path, refresh: bool = False, insecure: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the bridge-strike and truck-route datasets."""
    data_dir = Path(data_dir)
    strikes = load_dataset(URL_STRIKE, data_dir / STRIKES_CACHE, refresh, insecure)
    routes = load_dataset(URL_TRUCK_ROUTE, data_dir / ROUTES_CACHE, refresh, insecure)
    return strikes, routes
