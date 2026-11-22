"""Descriptive analyses. Every function takes the order fact table and returns a DataFrame
(or a plain dict for single-number summaries) so results can be saved, charted and tested."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .build import delivered_orders

DELAY_BINS = [-np.inf, -8, -4, 0, 3, 7, np.inf]
DELAY_LABELS = [
    "8+ days early",
    "4-7 days early",
    "On time (0-3 early)",
    "1-3 days late",
    "4-7 days late",
    "8+ days late",
]
DISTANCE_BINS = [0, 100, 300, 600, 1000, 1500, 2500, np.inf]
DISTANCE_LABELS = ["<100", "100-300", "300-600", "600-1,000", "1,000-1,500", "1,500-2,500", "2,500+"]
# Orders in these states never became a real purchase, so they don't count as customer activity.
_DEAD_STATUSES = {"canceled", "unavailable"}


def monthly_kpis(fact: pd.DataFrame, min_orders: int = config.MIN_ORDERS_PER_MONTH) -> pd.DataFrame:
    d = delivered_orders(fact)
    m = d.groupby("purchase_month").agg(
        orders=("order_id", "count"),
        revenue=("price_total", "sum"),
        avg_delivery_days=("delivery_days", "mean"),
        avg_promised_days=("promised_days", "mean"),
        late_rate=("is_late", "mean"),
        avg_review=("review_score", "mean"),
    )
    return m[m["orders"] >= min_orders].reset_index()


def late_impact(fact: pd.DataFrame) -> dict:
    """How much worse do customers rate late orders? Welch difference of means with 95% CI."""
    d = delivered_orders(fact).dropna(subset=["review_score"])
    late = d.loc[d["is_late"] == 1, "review_score"]
    ok = d.loc[d["is_late"] == 0, "review_score"]
    diff = ok.mean() - late.mean()
    se = np.sqrt(ok.var(ddof=1) / len(ok) + late.var(ddof=1) / len(late))
    all_delivered = delivered_orders(fact)
    return {
        "delivered_orders": int(len(all_delivered)),
        "late_rate": float(all_delivered["is_late"].mean()),
        "reviewed_orders": int(len(d)),
        "avg_review_on_time": float(ok.mean()),
        "avg_review_late": float(late.mean()),
        "review_gap": float(diff),
        "review_gap_ci_low": float(diff - 1.96 * se),
        "review_gap_ci_high": float(diff + 1.96 * se),
        "one_star_share_on_time": float((ok == 1).mean()),
        "one_star_share_late": float((late == 1).mean()),
        "revenue_in_late_orders": float(all_delivered.loc[all_delivered["is_late"] == 1, "price_total"].sum()),
        "revenue_total": float(all_delivered["price_total"].sum()),
    }


def delay_vs_review(fact: pd.DataFrame) -> pd.DataFrame:
    d = delivered_orders(fact).dropna(subset=["review_score"]).copy()
    d["delay_bucket"] = pd.cut(d["delay_days"], bins=DELAY_BINS, labels=DELAY_LABELS)
    out = d.groupby("delay_bucket", observed=True).agg(
        orders=("order_id", "count"),
        avg_review=("review_score", "mean"),
        one_star_share=("review_score", lambda s: (s == 1).mean()),
        five_star_share=("review_score", lambda s: (s == 5).mean()),
    )
    return out.reset_index()


def stage_breakdown(fact: pd.DataFrame) -> pd.DataFrame:
    """Where do late orders lose their time? Mean days per stage, on-time vs late."""
    d = delivered_orders(fact).copy()
    d["approval_days"] = d["approval_hours"] / 24
    d["outcome"] = np.where(d["is_late"] == 1, "Late", "On time")
    cols = ["approval_days", "handling_days", "transit_days", "delivery_days", "promised_days"]
    return d.groupby("outcome")[cols].mean().reset_index()


def state_performance(fact: pd.DataFrame, min_orders: int = 300) -> pd.DataFrame:
    d = delivered_orders(fact)
    s = d.groupby("customer_state").agg(
        orders=("order_id", "count"),
        late_rate=("is_late", "mean"),
        median_delivery_days=("delivery_days", "median"),
        median_promised_days=("promised_days", "median"),
        avg_review=("review_score", "mean"),
        median_freight_ratio=("freight_ratio", "median"),
        median_distance_km=("distance_km", "median"),
    )
    s["promise_buffer_days"] = s["median_promised_days"] - s["median_delivery_days"]
    return s[s["orders"] >= min_orders].sort_values("late_rate", ascending=False).reset_index()


def lane_performance(fact: pd.DataFrame, min_orders: int = 300) -> pd.DataFrame:
    d = delivered_orders(fact).dropna(subset=["seller_state"])
    lanes = d.groupby(["seller_state", "customer_state"]).agg(
        orders=("order_id", "count"),
        late_rate=("is_late", "mean"),
        median_delivery_days=("delivery_days", "median"),
        avg_review=("review_score", "mean"),
    )
    return lanes[lanes["orders"] >= min_orders].sort_values("late_rate", ascending=False).reset_index()


def distance_effect(fact: pd.DataFrame) -> pd.DataFrame:
    d = delivered_orders(fact).dropna(subset=["distance_km"]).copy()
    d["distance_band_km"] = pd.cut(d["distance_km"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS, right=False)
    out = d.groupby("distance_band_km", observed=True).agg(
        orders=("order_id", "count"),
        median_delivery_days=("delivery_days", "median"),
        median_promised_days=("promised_days", "median"),
        late_rate=("is_late", "mean"),
        avg_review=("review_score", "mean"),
    )
    return out.reset_index()


def category_performance(fact: pd.DataFrame, min_orders: int = 300) -> pd.DataFrame:
    d = delivered_orders(fact)
    c = d.groupby("primary_category").agg(
        orders=("order_id", "count"),
        revenue=("price_total", "sum"),
        late_rate=("is_late", "mean"),
        avg_review=("review_score", "mean"),
        median_freight_ratio=("freight_ratio", "median"),
    )
    return c[c["orders"] >= min_orders].sort_values("revenue", ascending=False).reset_index()


def seller_scorecard(
    fact: pd.DataFrame, min_orders: int = 30, prior_strength: int = 20
) -> pd.DataFrame:
    """Rank sellers on delivery reliability without over-reacting to small samples.

    ``late_rate_shrunk`` pulls each seller's late rate toward the platform average with a
    weight equal to ``prior_strength`` orders (an empirical-Bayes style estimate), so a
    seller with 2 late orders out of 3 is not ranked above one with 40 late out of 120.

    Tiers:
      * Priority  - shrunk late rate at least 2x the platform rate
      * Watch     - shrunk late rate at least 1.25x the platform rate, or average review
                    more than half a star below the platform average
      * Healthy   - everyone else
    """
    d = delivered_orders(fact).dropna(subset=["primary_seller_id"])
    base_rate = d["is_late"].mean()
    base_review = d["review_score"].mean()

    s = d.groupby("primary_seller_id").agg(
        orders=("order_id", "count"),
        revenue=("price_total", "sum"),
        late_orders=("is_late", "sum"),
        avg_handling_days=("handling_days", "mean"),
        avg_review=("review_score", "mean"),
        seller_state=("seller_state", "first"),
    )
    s = s[s["orders"] >= min_orders].copy()
    s["late_rate"] = s["late_orders"] / s["orders"]
    s["late_rate_shrunk"] = (s["late_orders"] + prior_strength * base_rate) / (s["orders"] + prior_strength)

    priority = s["late_rate_shrunk"] >= 2 * base_rate
    watch = (s["late_rate_shrunk"] >= 1.25 * base_rate) | (s["avg_review"] <= base_review - 0.5)
    s["tier"] = np.select([priority, watch], ["Priority", "Watch"], default="Healthy")
    s["platform_late_rate"] = base_rate
    return s.sort_values("late_rate_shrunk", ascending=False).reset_index()


def tier_summary(scorecard: pd.DataFrame) -> pd.DataFrame:
    t = scorecard.groupby("tier").agg(
        sellers=("primary_seller_id", "count"),
        orders=("orders", "sum"),
        revenue=("revenue", "sum"),
        avg_late_rate=("late_rate", "mean"),
        avg_review=("avg_review", "mean"),
    )
    t["revenue_share"] = t["revenue"] / t["revenue"].sum()
    order = pd.CategoricalIndex(t.index, ["Priority", "Watch", "Healthy"], ordered=True)
    return t.set_index(order).sort_index().reset_index().rename(columns={"index": "tier"})


def _live_orders(fact: pd.DataFrame) -> pd.DataFrame:
    return fact[~fact["order_status"].isin(_DEAD_STATUSES)]


def cohort_retention(fact: pd.DataFrame, max_offset: int = 6, min_cohort: int = 800) -> pd.DataFrame:
    """Share of each first-purchase-month cohort that buys again k months later."""
    live = _live_orders(fact)[["customer_unique_id", "purchase_month"]].drop_duplicates()
    first = live.groupby("customer_unique_id")["purchase_month"].min().rename("cohort")
    live = live.join(first, on="customer_unique_id")
    live["offset"] = (
        (live["purchase_month"].dt.year - live["cohort"].dt.year) * 12
        + (live["purchase_month"].dt.month - live["cohort"].dt.month)
    )
    sizes = live[live["offset"] == 0].groupby("cohort")["customer_unique_id"].nunique()
    active = live[live["offset"].between(0, max_offset)].groupby(["cohort", "offset"])[
        "customer_unique_id"
    ].nunique()
    table = active.unstack("offset").reindex(columns=range(max_offset + 1)).fillna(0)
    table = table.div(sizes, axis=0)
    table = table[sizes >= min_cohort]
    # Hide cells whose calendar month is beyond the end of the data: they are unobserved, not zero.
    last_month = live["purchase_month"].max()
    for cohort in table.index:
        for k in table.columns:
            if cohort + pd.DateOffset(months=int(k)) > last_month:
                table.loc[cohort, k] = np.nan
    table.insert(0, "cohort_size", sizes[table.index])
    return table.reset_index()


def repeat_purchase_by_first_experience(fact: pd.DataFrame, window_days: int = 180) -> dict[str, pd.DataFrame]:
    """Does a bad first delivery cost the follow-up purchase?

    A customer counts as 'repeat' if they place another order on a later calendar day within
    ``window_days`` of their first. Customers whose first order is too recent to observe the
    full window are excluded, otherwise the rate would be biased low for late joiners.
    """
    live = _live_orders(fact).sort_values("order_purchase_timestamp")
    first = live.drop_duplicates("customer_unique_id", keep="first")
    first = first[
        ["customer_unique_id", "order_purchase_timestamp", "is_late", "review_score", "is_delivered"]
    ].rename(columns={"order_purchase_timestamp": "first_ts"})

    later = live[["customer_unique_id", "order_purchase_timestamp"]].merge(
        first[["customer_unique_id", "first_ts"]], on="customer_unique_id"
    )
    later = later[later["order_purchase_timestamp"].dt.normalize() > later["first_ts"].dt.normalize()]
    gap = (later["order_purchase_timestamp"] - later["first_ts"]).dt.days
    repeaters = set(later.loc[gap <= window_days, "customer_unique_id"])

    cutoff = live["order_purchase_timestamp"].max() - pd.Timedelta(days=window_days)
    cohort = first[(first["first_ts"] <= cutoff) & first["is_delivered"] & first["is_late"].notna()].copy()
    cohort["repeat"] = cohort["customer_unique_id"].isin(repeaters)

    cohort["first_delivery"] = np.where(cohort["is_late"] == 1, "Late", "On time")

    def summarise(frame: pd.DataFrame, key: str) -> pd.DataFrame:
        out = frame.groupby(key).agg(customers=("customer_unique_id", "count"), repeat_rate=("repeat", "mean"))
        lo, hi = wilson_interval(out["repeat_rate"] * out["customers"], out["customers"])
        out["ci_low"], out["ci_high"] = lo, hi
        return out.reset_index()

    return {
        "by_delivery": summarise(cohort, "first_delivery"),
        "by_review_score": summarise(cohort.dropna(subset=["review_score"]), "review_score"),
    }


def wilson_interval(successes, n, z: float = 1.96):
    """95% Wilson score interval for a proportion (well behaved for small rates)."""
    successes, n = np.asarray(successes, dtype=float), np.asarray(n, dtype=float)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def run_all(fact: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
    scorecard = seller_scorecard(fact)
    repeat = repeat_purchase_by_first_experience(fact)
    return {
        "monthly_kpis": monthly_kpis(fact),
        "delay_vs_review": delay_vs_review(fact),
        "stage_breakdown": stage_breakdown(fact),
        "state_performance": state_performance(fact),
        "lane_performance": lane_performance(fact),
        "distance_effect": distance_effect(fact),
        "category_performance": category_performance(fact),
        "seller_scorecard": scorecard,
        "seller_tiers": tier_summary(scorecard),
        "cohort_retention": cohort_retention(fact),
        "repeat_by_delivery": repeat["by_delivery"],
        "repeat_by_review_score": repeat["by_review_score"],
        "late_impact": late_impact(fact),
    }
