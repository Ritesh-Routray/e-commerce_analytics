"""Build the order-level fact table that every analysis and the model read from.

One row per order. Delivery timing, basket, payment, review and geography are
joined once here so downstream code never repeats the joins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .loading import (
    load_orders,
    load_reviews,
    load_zip_centroids,
    read_table,
)

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    """Great-circle distance in km. Accepts scalars or arrays."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _days(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Elapsed fractional days; negative or missing spans are treated as bad data (NaN)."""
    delta = (later - earlier).dt.total_seconds() / 86400
    return delta.where(delta >= 0)


def add_delivery_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Derive timing columns from the five order timestamps.

    ``delay_days`` compares calendar dates (the estimate is always midnight), so
    an order delivered at 23:00 on the promised day counts as on time.
    ``is_late`` is only defined for orders that actually arrived.
    """
    out = df.copy()
    purchase = out["order_purchase_timestamp"]
    out["approval_hours"] = _days(out["order_approved_at"], purchase) * 24
    out["handling_days"] = _days(out["order_delivered_carrier_date"], out["order_approved_at"])
    out["transit_days"] = _days(
        out["order_delivered_customer_date"], out["order_delivered_carrier_date"]
    )
    out["delivery_days"] = _days(out["order_delivered_customer_date"], purchase)
    out["promised_days"] = _days(out["order_estimated_delivery_date"], purchase)

    arrived = out["order_delivered_customer_date"].notna()
    delay = (
        out["order_delivered_customer_date"].dt.normalize()
        - out["order_estimated_delivery_date"].dt.normalize()
    ).dt.days
    out["delay_days"] = delay.where(arrived)
    out["is_late"] = (out["delay_days"] > 0).where(arrived).astype("float")
    return out


def _basket_features(raw_dir=None) -> pd.DataFrame:
    items = read_table("items", raw_dir)
    products = read_table("products", raw_dir)
    translation = read_table("category_translation", raw_dir)

    products = products.merge(translation, on="product_category_name", how="left")
    products["category"] = (
        products["product_category_name_english"]
        .fillna(products["product_category_name"])
        .fillna("unknown")
    )
    products["volume_cm3"] = (
        products["product_length_cm"] * products["product_height_cm"] * products["product_width_cm"]
    )

    items = items.merge(
        products[["product_id", "category", "product_weight_g", "volume_cm3"]],
        on="product_id",
        how="left",
    )

    # The "primary" seller/category is the one behind the most expensive line item;
    # ties fall back to the lowest item number so the result is deterministic.
    ranked = items.sort_values(["order_id", "price", "order_item_id"], ascending=[True, False, True])
    primary = ranked.drop_duplicates("order_id")[["order_id", "seller_id", "category"]].rename(
        columns={"seller_id": "primary_seller_id", "category": "primary_category"}
    )

    agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"),
        price_total=("price", "sum"),
        freight_total=("freight_value", "sum"),
        weight_g_total=("product_weight_g", "sum"),
        volume_cm3_total=("volume_cm3", "sum"),
    )
    return agg.join(primary.set_index("order_id")).reset_index()


def _payment_features(raw_dir=None) -> pd.DataFrame:
    pay = read_table("payments", raw_dir)
    main_type = (
        pay.sort_values(["order_id", "payment_value"], ascending=[True, False])
        .drop_duplicates("order_id")[["order_id", "payment_type"]]
    )
    agg = pay.groupby("order_id").agg(
        payment_total=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
    )
    return agg.join(main_type.set_index("order_id")).reset_index()


def _review_features(raw_dir=None) -> pd.DataFrame:
    """Keep the most recently answered review per order (a few orders have several)."""
    reviews = load_reviews(raw_dir)
    latest = reviews.sort_values(["order_id", "review_answer_timestamp"]).drop_duplicates(
        "order_id", keep="last"
    )
    latest["has_comment"] = latest["review_comment_message"].notna()
    return latest[["order_id", "review_score", "has_comment"]]


def build_order_fact(raw_dir=None) -> pd.DataFrame:
    orders = add_delivery_metrics(load_orders(raw_dir))

    customers = read_table("customers", raw_dir).rename(
        columns={
            "customer_zip_code_prefix": "customer_zip",
            "customer_state": "customer_state",
            "customer_city": "customer_city",
        }
    )
    sellers = read_table("sellers", raw_dir).rename(
        columns={
            "seller_id": "primary_seller_id",
            "seller_zip_code_prefix": "seller_zip",
            "seller_state": "seller_state",
            "seller_city": "seller_city",
        }
    )

    fact = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(_basket_features(raw_dir), on="order_id", how="left")
        .merge(sellers, on="primary_seller_id", how="left")
        .merge(_payment_features(raw_dir), on="order_id", how="left")
        .merge(_review_features(raw_dir), on="order_id", how="left")
    )

    centroids = load_zip_centroids(raw_dir)
    cust_geo = centroids.rename(columns={"lat": "cust_lat", "lng": "cust_lng"})
    sell_geo = centroids.rename(columns={"lat": "sell_lat", "lng": "sell_lng"})
    fact = fact.join(cust_geo, on="customer_zip").join(sell_geo, on="seller_zip")
    fact["distance_km"] = haversine_km(
        fact["sell_lat"], fact["sell_lng"], fact["cust_lat"], fact["cust_lng"]
    )
    fact = fact.drop(columns=["cust_lat", "cust_lng", "sell_lat", "sell_lng"])

    fact["cross_state"] = (fact["seller_state"] != fact["customer_state"]).where(
        fact["seller_state"].notna()
    )
    fact["freight_ratio"] = fact["freight_total"] / fact["price_total"].replace(0, np.nan)
    purchase = fact["order_purchase_timestamp"]
    fact["purchase_month"] = purchase.dt.to_period("M").dt.to_timestamp()
    fact["purchase_dow"] = purchase.dt.dayofweek
    fact["purchase_hour"] = purchase.dt.hour
    fact["is_delivered"] = (fact["order_status"] == "delivered") & fact[
        "order_delivered_customer_date"
    ].notna()
    return fact.sort_values("order_purchase_timestamp").reset_index(drop=True)


def delivered_orders(fact: pd.DataFrame) -> pd.DataFrame:
    """Orders that reached the customer, with a usable delivery span."""
    mask = fact["is_delivered"] & fact["delivery_days"].notna() & fact["is_late"].notna()
    return fact[mask].copy()


def save_fact(fact: pd.DataFrame) -> None:
    config.ensure_dirs()
    fact.to_parquet(config.FACT_PATH, index=False)


def load_fact() -> pd.DataFrame:
    return pd.read_parquet(config.FACT_PATH)
