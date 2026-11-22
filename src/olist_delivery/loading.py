"""Reading the raw Olist CSVs.

The public Kaggle release and some re-exports of it differ in file names and
timestamp formats, so both are handled here instead of leaking into the rest of
the code.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

# logical name -> accepted file names (first match wins)
_FILE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "orders": ("olist_orders_dataset.csv", "olist_orders.csv"),
    "items": ("olist_order_items_dataset.csv", "olist_order_items.csv"),
    "payments": ("olist_order_payments_dataset.csv", "olist_payments.csv"),
    "reviews": ("olist_order_reviews_dataset.csv", "olist_reviews.csv"),
    "customers": ("olist_customers_dataset.csv", "olist_customers.csv"),
    "sellers": ("olist_sellers_dataset.csv", "olist_sellers.csv"),
    "products": ("olist_products_dataset.csv", "olist_products.csv"),
    "geolocation": ("olist_geolocation_dataset.csv", "olist_geolocation.csv"),
    "category_translation": (
        "product_category_name_translation.csv",
    ),
}

_TS_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d-%m-%Y %H:%M",
    "%d-%m-%Y %H:%M:%S",
)


def resolve_path(name: str, raw_dir: Path | None = None) -> Path:
    raw_dir = Path(raw_dir or config.RAW_DIR)
    for candidate in _FILE_CANDIDATES[name]:
        path = raw_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find the '{name}' table in {raw_dir}. Expected one of "
        f"{_FILE_CANDIDATES[name]}. See data/README.md for download instructions."
    )


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse timestamps that may use ISO or day-first formats (even mixed in one column)."""
    text = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    pending = text.notna() & (text != "")
    for fmt in _TS_FORMATS:
        if not pending.any():
            break
        attempt = pd.to_datetime(text[pending], format=fmt, errors="coerce")
        ok = attempt.notna()
        parsed.loc[attempt.index[ok]] = attempt[ok]
        pending.loc[attempt.index[ok]] = False
    return parsed


def read_table(name: str, raw_dir: Path | None = None, **kwargs) -> pd.DataFrame:
    return pd.read_csv(resolve_path(name, raw_dir), **kwargs)


def load_orders(raw_dir: Path | None = None) -> pd.DataFrame:
    orders = read_table("orders", raw_dir)
    for col in (
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ):
        orders[col] = parse_timestamps(orders[col])
    return orders


def load_reviews(raw_dir: Path | None = None) -> pd.DataFrame:
    reviews = read_table("reviews", raw_dir)
    for col in ("review_creation_date", "review_answer_timestamp"):
        reviews[col] = parse_timestamps(reviews[col])
    reviews["review_score"] = pd.to_numeric(reviews["review_score"], errors="coerce")
    return reviews


def load_zip_centroids(raw_dir: Path | None = None) -> pd.DataFrame:
    """One (lat, lng) per zip-code prefix: the mean of all geolocation rows inside Brazil."""
    geo = read_table("geolocation", raw_dir)
    in_brazil = geo["geolocation_lat"].between(*config.BRAZIL_LAT) & geo[
        "geolocation_lng"
    ].between(*config.BRAZIL_LNG)
    geo = geo[in_brazil]
    centroids = (
        geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .mean()
        .rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng"})
    )
    centroids.index.name = "zip_prefix"
    return centroids
