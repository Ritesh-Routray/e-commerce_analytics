"""Paths and project-wide constants."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Raw Olist CSVs. Override with OLIST_RAW_DIR if you keep the data elsewhere.
RAW_DIR = Path(os.environ.get("OLIST_RAW_DIR", ROOT / "data" / "raw"))
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

FACT_PATH = PROCESSED_DIR / "order_fact.parquet"

# Months with fewer orders than this are dropped from trend charts: the first and
# last months of the dataset are partial and would distort the picture.
MIN_ORDERS_PER_MONTH = 500

# Brazil bounding box, used to discard geolocation rows with impossible coordinates.
BRAZIL_LAT = (-34.0, 6.0)
BRAZIL_LNG = (-74.0, -33.0)

# Model split: everything purchased before TEST_START is used for training/validation,
# everything from TEST_START onward is the held-out future.
VALID_START = "2018-01-01"
TEST_START = "2018-04-01"
RANDOM_STATE = 42


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
