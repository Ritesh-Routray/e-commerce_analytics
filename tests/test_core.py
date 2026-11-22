import numpy as np
import pandas as pd

from olist_delivery.analysis import wilson_interval
from olist_delivery.build import add_delivery_metrics, haversine_km
from olist_delivery.loading import parse_timestamps
from olist_delivery.model import add_asof_rates


def test_parse_timestamps_handles_iso_and_day_first():
    s = pd.Series(["2017-10-02 10:56:33", "02-10-2017 10:56", None, "garbage"])
    out = parse_timestamps(s)
    assert out.iloc[0] == pd.Timestamp("2017-10-02 10:56:33")
    assert out.iloc[1] == pd.Timestamp("2017-10-02 10:56:00")
    assert out.iloc[2:].isna().all()


def test_haversine_sao_paulo_to_rio():
    d = haversine_km(-23.5505, -46.6333, -22.9068, -43.1729)
    assert 350 < float(d) < 365


def _order(purchase, approved, carrier, delivered, estimated):
    return pd.DataFrame(
        {
            "order_purchase_timestamp": [pd.Timestamp(purchase)],
            "order_approved_at": [pd.Timestamp(approved)],
            "order_delivered_carrier_date": [pd.Timestamp(carrier)],
            "order_delivered_customer_date": [pd.Timestamp(delivered) if delivered else pd.NaT],
            "order_estimated_delivery_date": [pd.Timestamp(estimated)],
        }
    )


def test_late_is_judged_on_calendar_date():
    on_time = add_delivery_metrics(
        _order("2018-01-01 10:00", "2018-01-01 11:00", "2018-01-02 09:00", "2018-01-10 23:30", "2018-01-10")
    )
    late = add_delivery_metrics(
        _order("2018-01-01 10:00", "2018-01-01 11:00", "2018-01-02 09:00", "2018-01-11 00:10", "2018-01-10")
    )
    assert on_time["is_late"].iloc[0] == 0 and on_time["delay_days"].iloc[0] == 0
    assert late["is_late"].iloc[0] == 1 and late["delay_days"].iloc[0] == 1


def test_undelivered_order_has_no_late_flag():
    row = add_delivery_metrics(
        _order("2018-01-01 10:00", "2018-01-01 11:00", "2018-01-02 09:00", None, "2018-01-10")
    )
    assert np.isnan(row["is_late"].iloc[0])


def test_negative_durations_become_missing():
    row = add_delivery_metrics(
        _order("2018-01-05 10:00", "2018-01-04 11:00", "2018-01-02 09:00", "2018-01-09 10:00", "2018-01-10")
    )
    assert np.isnan(row["approval_hours"].iloc[0])
    assert np.isnan(row["handling_days"].iloc[0])


def test_asof_rate_never_uses_deliveries_after_purchase():
    t = pd.Timestamp
    df = pd.DataFrame(
        {
            "seller": ["a", "a", "a"],
            "order_purchase_timestamp": [t("2018-01-01"), t("2018-01-05"), t("2018-01-20")],
            "order_delivered_customer_date": [t("2018-01-10"), t("2018-01-12"), t("2018-01-25")],
            "is_late": [1.0, 0.0, 1.0],
        }
    )
    out = add_asof_rates(df, ["seller"], "s", prior_rate=0.5, prior_strength=2)
    # order 0: nothing delivered yet; order 1 (Jan 5): still nothing; order 2 (Jan 20): two deliveries seen
    assert out["s_orders"].tolist() == [0, 0, 2]
    assert out["s_late_rate"].iloc[0] == 0.5
    assert np.isclose(out["s_late_rate"].iloc[2], (1 + 2 * 0.5) / (2 + 2))


def test_wilson_interval_contains_estimate():
    lo, hi = wilson_interval(np.array([20.0]), np.array([1000.0]))
    assert lo[0] < 0.02 < hi[0] and 0 <= lo[0] and hi[0] <= 1
