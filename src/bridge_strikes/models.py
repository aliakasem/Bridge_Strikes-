"""City-wide monthly count models: Poisson vs Negative Binomial."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

log = logging.getLogger(__name__)


def monthly_series(strikes: pd.DataFrame) -> pd.DataFrame:
    """Monthly strike counts, including months with zero strikes.

    ``t`` is months since the first month, which keeps the trend term on a
    sensible scale.
    """
    counts = strikes.groupby("month_start").size()
    months = pd.date_range(counts.index.min(), counts.index.max(), freq="MS")
    counts = counts.reindex(months, fill_value=0)
    return pd.DataFrame({"month_start": months, "strike_count": counts.to_numpy(), "t": np.arange(len(months))})


def _p_at_least_one_poisson(lam: float) -> float:
    return 1.0 - np.exp(-lam)


def _p_at_least_one_nb(lam: float, alpha: float) -> float:
    # NB2 with mean lam and variance lam + alpha * lam**2: P(Y=0) = (1 + alpha*lam)^(-1/alpha).
    # As alpha -> 0 this converges to the Poisson case.
    if alpha < 1e-8:
        return _p_at_least_one_poisson(lam)
    return 1.0 - (1.0 + alpha * lam) ** (-1.0 / alpha)


def fit_count_models(strikes: pd.DataFrame) -> tuple[pd.DataFrame, str | None, pd.DataFrame]:
    """Fit Poisson and NB2 trend models by maximum likelihood.

    Both are fit with statsmodels so their AIC values are comparable. The NB
    dispersion parameter alpha is estimated, not fixed.

    Returns (comparison table, best model name or None, monthly series).
    """
    monthly = monthly_series(strikes)
    mean, var = monthly["strike_count"].mean(), monthly["strike_count"].var()
    log.info("Monthly counts: %d months, mean %.2f, variance %.2f (ratio %.2f)", len(monthly), mean, var, var / mean)

    next_t = int(monthly["t"].max()) + 1
    next_month = monthly["month_start"].max() + pd.offsets.MonthBegin(1)
    rows = []

    for name, fitter in [("Poisson", smf.poisson), ("Negative Binomial", smf.negativebinomial)]:
        row = {"model_type": name, "forecast_month": next_month.date(), "n_months": len(monthly)}
        if len(monthly) < 3:
            log.warning("%s: only %d months of data, skipping", name, len(monthly))
            rows.append(row)
            continue
        try:
            res = fitter("strike_count ~ t", data=monthly).fit(disp=0, maxiter=500)
            lam = float(np.asarray(res.predict(pd.DataFrame({"t": [next_t]})))[0])
            if name == "Poisson":
                p = _p_at_least_one_poisson(lam)
            else:
                row["nb_alpha"] = float(res.params["alpha"])
                p = _p_at_least_one_nb(lam, row["nb_alpha"])
            row.update(
                expected_strikes=lam,
                probability_strike=p,
                trend_coef=float(res.params["t"]),
                trend_pvalue=float(res.pvalues["t"]),
                aic=float(res.aic),
                bic=float(res.bic),
                log_likelihood=float(res.llf),
                converged=bool(res.mle_retvals.get("converged", True)),
            )
            log.info("%s: expected %.2f strikes in %s, P(>=1) = %.3f, AIC %.1f",
                     name, lam, next_month.strftime("%Y-%m"), p, res.aic)
        except Exception as exc:  # noqa: BLE001 - record failure, don't invent numbers
            log.warning("%s model failed: %s", name, exc)
        rows.append(row)

    comparison = pd.DataFrame(rows)
    for col in ("expected_strikes", "probability_strike", "aic"):
        if col not in comparison:
            comparison[col] = np.nan

    valid = comparison[comparison["aic"].notna()]
    best = comparison.loc[valid["aic"].idxmin(), "model_type"] if len(valid) else None
    comparison["is_best"] = comparison["model_type"].eq(best)
    return comparison, best, monthly
