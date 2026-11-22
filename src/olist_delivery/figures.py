"""Static charts (PNG). One restrained palette, thin marks, no dual axes, direct labels."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LIGHT_BLUE = "#86b6ef"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlecolor": INK,
    }
)


def _save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _no_x_grid(ax):
    ax.grid(axis="x", visible=False)


def monthly_trend(monthly: pd.DataFrame) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, height_ratios=[1, 1])
    x = monthly["purchase_month"]
    top.bar(x, monthly["orders"], width=20, color=BLUE)
    top.set_title("Orders delivered, by purchase month")
    top.set_ylabel("Orders")
    _no_x_grid(top)

    bottom.plot(x, monthly["late_rate"] * 100, color=ORANGE, lw=2, marker="o", ms=5, mfc=SURFACE, mew=1.5)
    peak = monthly.loc[monthly["late_rate"].idxmax()]
    bottom.annotate(
        f"{peak['late_rate']:.0%} late",
        (peak["purchase_month"], peak["late_rate"] * 100),
        xytext=(8, 4),
        textcoords="offset points",
        color=INK2,
        fontsize=9,
    )
    bottom.set_title("Share of orders delivered after the promised date")
    bottom.set_ylabel("Late orders (%)")
    bottom.set_ylim(bottom=0)
    _no_x_grid(bottom)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, "01_monthly_orders_and_late_rate")


def review_by_delay(delay: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = [BLUE if "late" not in b else ORANGE for b in delay["delay_bucket"].astype(str)]
    bars = ax.bar(delay["delay_bucket"].astype(str), delay["avg_review"], color=colors, width=0.62)
    for bar, n, val in zip(bars, delay["orders"], delay["avg_review"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.06, f"{val:.2f}", ha="center", color=INK, fontsize=10)
        ax.text(bar.get_x() + bar.get_width() / 2, 0.12, f"n={n:,}", ha="center", color="white", fontsize=8)
    ax.set_ylim(0, 5)
    ax.set_ylabel("Average review score (1-5)")
    ax.set_title("Customers punish lateness, and reward earliness very little")
    ax.tick_params(axis="x", labelsize=8.5)
    _no_x_grid(ax)
    fig.tight_layout()
    _save(fig, "02_review_score_by_delay")


def stage_breakdown(stages: pd.DataFrame) -> None:
    stages = stages.set_index("outcome").loc[["On time", "Late"]]
    parts = [("approval_days", "Payment approval", BLUE), ("handling_days", "Seller handling", ORANGE), ("transit_days", "Carrier transit", AQUA)]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    left = np.zeros(len(stages))
    for col, label, color in parts:
        vals = stages[col].to_numpy()
        ax.barh(stages.index, vals, left=left, color=color, height=0.55, label=label, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(vals):
            if v >= 1.5:
                ax.text(left[i] + v / 2, i, f"{v:.1f}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        left += vals
    for i, total in enumerate(left):
        ax.text(total + 0.4, i, f"{total:.1f} days", va="center", color=INK2, fontsize=10)
    ax.set_xlim(0, left.max() * 1.15)
    ax.set_xlabel("Average days from purchase")
    ax.set_title("Late orders lose most of their time with the carrier")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "03_time_by_stage")


def state_late_rate(states: pd.DataFrame, national_rate: float) -> None:
    top = states.sort_values("late_rate", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["customer_state"], top["late_rate"] * 100, color=BLUE, height=0.65)
    ax.axvline(national_rate * 100, color=ORANGE, lw=1.5)
    ax.text(national_rate * 100 + 0.3, -0.9, f"platform {national_rate:.1%}", color=ORANGE, fontsize=9, va="center")
    for y, (rate, n) in enumerate(zip(top["late_rate"], top["orders"])):
        ax.text(rate * 100 + 0.3, y, f"{rate:.1%}  (n={n:,})", va="center", color=INK2, fontsize=8.5)
    ax.set_xlim(0, top["late_rate"].max() * 100 * 1.3)
    ax.set_xlabel("Late orders (%)")
    ax.set_title("Late-delivery rate by destination state (15 highest)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "04_late_rate_by_state")


def distance_effect(dist: pd.DataFrame) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(dist))
    a.plot(x, dist["median_promised_days"], color=BLUE, lw=2, marker="o", ms=5)
    a.plot(x, dist["median_delivery_days"], color=ORANGE, lw=2, marker="o", ms=5)
    a.text(x[-1], dist["median_promised_days"].iloc[-1] + 1, "promised", color=BLUE, ha="right", fontsize=9)
    a.text(x[-1], dist["median_delivery_days"].iloc[-1] - 3, "actual", color=ORANGE, ha="right", fontsize=9)
    a.set_title("Promise vs reality (median days)")
    a.set_ylim(bottom=0)
    b.bar(x, dist["late_rate"] * 100, color=BLUE, width=0.6)
    b.set_title("Late orders (%)")
    for ax in (a, b):
        ax.set_xticks(x, dist["distance_band_km"].astype(str), rotation=30, ha="right", fontsize=8.5)
        ax.set_xlabel("Seller-to-customer distance (km)")
        _no_x_grid(ax)
    fig.tight_layout()
    _save(fig, "05_distance_effect")


def repeat_purchase(by_delivery: pd.DataFrame, window_days: int = 180) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    df = by_delivery.set_index("first_delivery").loc[["On time", "Late"]]
    colors = [BLUE, ORANGE]
    rate = df["repeat_rate"] * 100
    err = np.vstack([rate - df["ci_low"] * 100, df["ci_high"] * 100 - rate])
    ax.bar(df.index, rate, color=colors, width=0.5, yerr=err, ecolor=INK2, capsize=5)
    for i, (r, n) in enumerate(zip(rate, df["customers"])):
        ax.text(i, 0.1, f"{r:.2f}%\nn={n:,}", ha="center", color="white", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(df["ci_high"] * 100) * 1.25)
    ax.set_ylabel(f"Bought again within {window_days} days (%)")
    ax.set_title("Repeat purchase by first-order delivery (95% CI)")
    _no_x_grid(ax)
    fig.tight_layout()
    _save(fig, "06_repeat_purchase_by_first_delivery")


def seller_scorecard(sc: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    styles = {"Healthy": ("#c3c2b7", 0.6), "Watch": (BLUE, 0.8), "Priority": (ORANGE, 0.95)}
    for tier, (color, alpha) in styles.items():
        part = sc[sc["tier"] == tier]
        ax.scatter(
            part["late_rate_shrunk"] * 100,
            part["avg_review"],
            s=np.clip(part["orders"] / 4, 14, 200),
            color=color,
            alpha=alpha,
            edgecolor=SURFACE,
            linewidth=0.8,
            label=f"{tier} ({len(part)})",
        )
    ax.axvline(sc["platform_late_rate"].iloc[0] * 100, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("Late-delivery rate, shrunk toward platform average (%)")
    ax.set_ylabel("Average review score")
    ax.set_title("Seller reliability scorecard (marker size = orders)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    _save(fig, "07_seller_scorecard")


def model_folds(folds: pd.DataFrame) -> None:
    pivot = folds.pivot(index="fold_start", columns="model", values="roc_auc")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(pivot))
    w = 0.38
    ax.bar(x - w / 2, pivot["lightgbm"], w, color=BLUE, label="LightGBM")
    ax.bar(x + w / 2, pivot["logistic_regression"], w, color=LIGHT_BLUE, label="Logistic regression")
    ax.axhline(0.5, color=ORANGE, lw=1.5)
    ax.text(len(pivot) - 0.55, 0.51, "no skill", color=ORANGE, fontsize=9, ha="right")
    ax.set_xticks(x, [f"{s[:7]}\n+2 months" for s in pivot.index])
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("ROC AUC (walk-forward)")
    ax.set_title("Predicting late delivery at checkout: modest, uneven signal")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    _no_x_grid(ax)
    fig.tight_layout()
    _save(fig, "08_model_walk_forward_auc")


def model_deciles(deciles: pd.DataFrame, base_rate: float) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(deciles["risk_decile"].astype(int), deciles["actual_late_rate"] * 100, color=BLUE, width=0.65)
    ax.axhline(base_rate * 100, color=ORANGE, lw=1.5)
    ax.text(10.4, base_rate * 100 + 0.3, f"average {base_rate:.1%}", color=ORANGE, ha="right", fontsize=9)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Predicted risk decile (1 = highest risk), out-of-fold")
    ax.set_ylabel("Actual late orders (%)")
    ax.set_title("Orders the model ranks as riskiest really are late more often")
    _no_x_grid(ax)
    fig.tight_layout()
    _save(fig, "09_model_risk_deciles")


def feature_importance(imp: pd.DataFrame, top: int = 12) -> None:
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(t["feature"], t["share"] * 100, color=BLUE, height=0.65)
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_title("What the model leans on")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "10_model_feature_importance")


def make_all(results: dict, model_result: dict | None) -> None:
    monthly_trend(results["monthly_kpis"])
    review_by_delay(results["delay_vs_review"])
    stage_breakdown(results["stage_breakdown"])
    state_late_rate(results["state_performance"], results["late_impact"]["late_rate"])
    distance_effect(results["distance_effect"])
    repeat_purchase(results["repeat_by_delivery"])
    seller_scorecard(results["seller_scorecard"])
    if model_result:
        model_folds(model_result["folds"])
        model_deciles(model_result["deciles"], model_result["metrics"]["pooled_out_of_fold_lightgbm"]["base_rate"])
        feature_importance(model_result["importance"])
