"""Figures. Each function returns a matplotlib Figure; callers save or show it."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _location_label(row: pd.Series, max_len: int = 30) -> str:
    if pd.notna(row.get("main_street")) and pd.notna(row.get("cross_street")):
        label = f"{row['main_street']} & {row['cross_street']}"
    else:
        label = str(row["location_id"])
    return label if len(label) <= max_len else label[: max_len - 3] + "..."


def plot_model_comparison(comparison: pd.DataFrame) -> plt.Figure:
    """Side-by-side bars of expected strikes and P(>=1 strike) per model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    models = comparison["model_type"]
    edge = ["red" if best else "black" for best in comparison["is_best"]]
    width = [3 if best else 1 for best in comparison["is_best"]]
    colors = ["skyblue", "lightcoral"][: len(models)]

    for ax, col, ylabel, fmt, title in [
        (ax1, "expected_strikes", "Expected strikes next month", "{:.2f}", "Expected strike count"),
        (ax2, "probability_strike", "P(at least one strike)", "{:.3f}", "Strike probability"),
    ]:
        values = comparison[col].fillna(0)
        bars = ax.bar(models, values, color=colors, edgecolor=edge, linewidth=width, alpha=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, comparison[col]):
            txt = "n/a" if pd.isna(v) else fmt.format(v)
            ax.annotate(txt, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontweight="bold", xytext=(0, 3), textcoords="offset points")
    ax2.set_ylim(0, 1.1)
    fig.suptitle("Model comparison (red outline = lowest AIC)")
    fig.tight_layout()
    return fig


def plot_monthly_series(monthly: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly["month_start"], monthly["strike_count"], marker="o", markersize=3)
    ax.set_ylabel("Strikes per month")
    ax.set_title("City-wide bridge strikes per month", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_top_risk_locations(location_risk: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Four-panel summary of the highest-risk locations."""
    top = location_risk.head(top_n).reset_index(drop=True).copy()
    top["label"] = top.apply(_location_label, axis=1)
    n = len(top)
    x = np.arange(n)
    loc_ticks = [f"Loc {i + 1}" for i in range(n)]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(f"Top {n} bridge-strike risk locations", fontsize=16, fontweight="bold")

    bars = ax1.barh(x, top["p_next_month"], color=plt.cm.Reds(np.linspace(1, 0.6, n)))
    ax1.set_yticks(x, [f"{i + 1}. {lbl}" for i, lbl in enumerate(top["label"])])
    ax1.invert_yaxis()
    ax1.set_xlabel("P(strike next month)")
    ax1.set_title("Strike probability by location", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)
    for bar in bars:
        ax1.annotate(f"{bar.get_width():.3f}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                     xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, fontweight="bold")

    w = 0.35
    ax2.bar(x - w / 2, top["total_strikes"], w, label="Total strikes", color="steelblue", alpha=0.8)
    ax2.bar(x + w / 2, top["strikes_last_12m"], w, label="Last 12 months", color="coral", alpha=0.8)
    ax2.set_xticks(x, loc_ticks, rotation=45)
    ax2.set_ylabel("Strikes")
    ax2.set_title("Strike history", fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    bars3 = ax3.bar(x, top["dist_to_route_m"], color=plt.cm.Blues(np.linspace(0.6, 1, n)))
    ax3.set_xticks(x, loc_ticks, rotation=45)
    ax3.set_ylabel("Distance to truck route (m)")
    ax3.set_title("Proximity to truck routes", fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)
    for bar in bars3:
        ax3.annotate(f"{bar.get_height():.1f}m", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    sc = ax4.scatter(top["months_with_strikes"], top["lambda_per_month"], s=top["total_strikes"] * 20,
                     c=top["p_next_month"], cmap="Reds", edgecolors="black", linewidths=0.5, alpha=0.8)
    for i, row in top.iterrows():
        ax4.annotate(str(i + 1), (row["months_with_strikes"], row["lambda_per_month"]),
                     xytext=(5, 5), textcoords="offset points", fontsize=9, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax4.set_xlabel("Months with at least one strike")
    ax4.set_ylabel("Strike rate (λ per month)")
    ax4.set_title("Recurrence vs rate (bubble size = total strikes)", fontweight="bold")
    ax4.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax4, label="P(strike next month)")

    fig.tight_layout()
    return fig
