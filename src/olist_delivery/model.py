"""Late-delivery risk model.

Question: at the moment a customer checks out, can we tell which orders are likely to
arrive after the promised date? Only information available at purchase time is used.

Leakage control is the main design concern:
  * No post-purchase timestamps (approval, carrier hand-over) are used as features.
  * Seller / lane / destination history features are computed "as of" each order's purchase
    time, counting only deliveries that had already happened by then.
  * Evaluation is walk-forward: each fold trains only on orders whose delivery was already
    complete before the fold starts, then scores the next two months. A single random split
    would mix future and past and flatter the model.

Late rates on Olist swing wildly by period (Black Friday 2017, a delay wave in Feb-Mar 2018,
then a calm spring), so the model is judged on how well it *ranks* orders by risk within a
period, not on absolute probabilities.
"""
from __future__ import annotations

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .build import delivered_orders

BASE_NUMERIC = [
    "price_total",
    "freight_total",
    "freight_ratio",
    "n_items",
    "n_sellers",
    "weight_g_total",
    "volume_cm3_total",
    "distance_km",
    "promised_days",
    "max_installments",
    "payment_total",
    "purchase_hour",
    "purchase_dow",
    "cross_state",
]
SELLER_HISTORY = ["seller_hist_orders", "seller_hist_late_rate"]
LANE_HISTORY = ["lane_hist_orders", "lane_hist_late_rate"]
REGIME_HISTORY = ["dest_hist_late_rate", "platform_recent_late_rate"]
CATEGORICAL = ["customer_state", "seller_state", "category_group", "payment_type"]

# Named feature sets, used both for the final model and for the ablation table.
FEATURE_SETS: dict[str, list[str]] = {
    "order_only": BASE_NUMERIC,
    "plus_seller_lane_history": BASE_NUMERIC + SELLER_HISTORY + LANE_HISTORY,
    "plus_regime_history": BASE_NUMERIC + SELLER_HISTORY + LANE_HISTORY + REGIME_HISTORY,
}
# The ablation (see outputs/tables/model_ablation.csv) shows the as-of history features lower
# walk-forward AUC: they track the period's delay wave rather than durable seller quality.
DEFAULT_FEATURE_SET = "order_only"

TOP_CATEGORIES = 25
HISTORY_PRIOR_STRENGTH = 10
PLATFORM_WINDOW = pd.Timedelta(days=28)
FOLD_STARTS = ["2017-11-01", "2018-01-01", "2018-03-01", "2018-05-01", "2018-07-01"]
FOLD_MONTHS = 2


def add_asof_rates(
    df: pd.DataFrame,
    keys: list[str] | None,
    prefix: str,
    prior_rate: float,
    window: pd.Timedelta | None = None,
    prior_strength: int = HISTORY_PRIOR_STRENGTH,
    emit_count: bool = True,
) -> pd.DataFrame:
    """Add ``{prefix}_late_rate`` (and ``{prefix}_orders``) using only deliveries completed
    strictly before each order's purchase time.

    ``keys=None`` treats the whole table as one group (platform-wide). With ``window`` only the
    most recent deliveries inside that look-back are counted. Rates are smoothed toward
    ``prior_rate`` so thin histories don't produce extreme values.
    """
    out = df.copy()
    purchase = out["order_purchase_timestamp"].to_numpy("datetime64[ns]")
    delivered = out["order_delivered_customer_date"].to_numpy("datetime64[ns]")
    late = out["is_late"].to_numpy(dtype=float)

    n_seen = np.zeros(len(out))
    n_late = np.zeros(len(out))

    if keys is None:
        groups = [np.arange(len(out))]
    else:
        groups = out.groupby(keys, observed=True).indices.values()

    for idx in groups:
        order = np.argsort(delivered[idx], kind="stable")
        d_sorted = delivered[idx][order]
        cum_late = np.concatenate([[0.0], np.cumsum(late[idx][order])])
        hi = np.searchsorted(d_sorted, purchase[idx], side="left")  # delivered before purchase
        if window is None:
            lo = np.zeros_like(hi)
        else:
            lo = np.searchsorted(d_sorted, purchase[idx] - np.timedelta64(window), side="left")
        n_seen[idx] = hi - lo
        n_late[idx] = cum_late[hi] - cum_late[lo]

    out[f"{prefix}_late_rate"] = (n_late + prior_strength * prior_rate) / (n_seen + prior_strength)
    if emit_count:
        out[f"{prefix}_orders"] = n_seen
    return out


def prepare_modeling_table(fact: pd.DataFrame) -> pd.DataFrame:
    """Delivered orders plus the as-of history features and categorical encodings."""
    d = delivered_orders(fact).reset_index(drop=True)
    d["cross_state"] = d["cross_state"].astype(float)

    # Smoothing prior = late rate among orders delivered before the first fold begins.
    first_fold = pd.Timestamp(FOLD_STARTS[0])
    prior = float(d.loc[d["order_delivered_customer_date"] < first_fold, "is_late"].mean())

    d = add_asof_rates(d, ["primary_seller_id"], "seller_hist", prior)
    d = add_asof_rates(d, ["seller_state", "customer_state"], "lane_hist", prior)
    d = add_asof_rates(d, ["customer_state"], "dest_hist", prior, emit_count=False)
    d = add_asof_rates(d, None, "platform_recent", prior, window=PLATFORM_WINDOW, emit_count=False)

    top = d["primary_category"].value_counts().head(TOP_CATEGORIES).index
    d["category_group"] = d["primary_category"].where(d["primary_category"].isin(top), "other")
    for col in CATEGORICAL:
        d[col] = d[col].astype("category")
    d.attrs["prior_rate"] = prior
    return d


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y)
    order = np.argsort(-p)
    y_sorted = y[order]
    base = float(np.mean(y))
    out = {
        "orders": int(len(y)),
        "base_rate": base,
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }
    for frac in (0.05, 0.10, 0.20):
        k = max(1, int(round(len(y) * frac)))
        caught = float(y_sorted[:k].sum())
        out[f"precision_at_{int(frac * 100)}pct"] = caught / k
        out[f"recall_at_{int(frac * 100)}pct"] = caught / float(y_sorted.sum())
        out[f"lift_at_{int(frac * 100)}pct"] = (caught / k) / base
    return out


def risk_deciles(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y), "p": p})
    frame["risk_decile"] = pd.qcut(frame["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    out = frame.groupby("risk_decile", observed=True).agg(
        orders=("y", "size"), predicted_late_rate=("p", "mean"), actual_late_rate=("y", "mean")
    )
    return out.reset_index()


def _lgbm() -> lgb.LGBMClassifier:
    # Deliberately small and heavily regularised: the signal is weak and the regime shifts,
    # so a big model mostly memorises one period's delay wave.
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=10.0,
        random_state=config.RANDOM_STATE,
        verbose=-1,
    )


def _logreg() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)
    )


def _fold_frames(d: pd.DataFrame):
    for start in FOLD_STARTS:
        lo = pd.Timestamp(start)
        hi = lo + pd.DateOffset(months=FOLD_MONTHS)
        # Train only on orders whose outcome was already known when the fold begins.
        train = d[d["order_delivered_customer_date"] < lo]
        test = d[(d["order_purchase_timestamp"] >= lo) & (d["order_purchase_timestamp"] < hi)]
        if len(test) and test["is_late"].nunique() == 2 and len(train) > 1000:
            yield start, train, test


def walk_forward(d: pd.DataFrame, numeric: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-fold metrics for each model, pooled out-of-fold LightGBM scores)."""
    cols = numeric + CATEGORICAL
    rows, pooled = [], []
    for start, train, test in _fold_frames(d):
        y_test = test["is_late"].to_numpy()

        gbm = _lgbm().fit(train[cols], train["is_late"])
        p_gbm = gbm.predict_proba(test[cols])[:, 1]
        lr = _logreg().fit(train[numeric], train["is_late"])
        p_lr = lr.predict_proba(test[numeric])[:, 1]

        for name, p in (("lightgbm", p_gbm), ("logistic_regression", p_lr)):
            rows.append({"fold_start": start, "model": name, **ranking_metrics(y_test, p)})
        pooled.append(test[["order_id", "order_purchase_timestamp", "is_late"]].assign(fold=start, risk=p_gbm))
    return pd.DataFrame(rows), pd.concat(pooled, ignore_index=True)


def ablation(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, numeric in FEATURE_SETS.items():
        folds, _ = walk_forward(d, numeric)
        g = folds[folds["model"] == "lightgbm"]
        rows.append(
            {
                "feature_set": name,
                "n_features": len(numeric) + len(CATEGORICAL),
                "mean_roc_auc": g["roc_auc"].mean(),
                "mean_pr_auc": g["pr_auc"].mean(),
                "mean_lift_at_10pct": g["lift_at_10pct"].mean(),
            }
        )
    return pd.DataFrame(rows)


def train_and_evaluate(fact: pd.DataFrame, save: bool = True) -> dict:
    d = prepare_modeling_table(fact)
    numeric = FEATURE_SETS[DEFAULT_FEATURE_SET]
    cols = numeric + CATEGORICAL

    folds, pooled = walk_forward(d, numeric)
    abl = ablation(d)

    summary = (
        folds.groupby("model")[["roc_auc", "pr_auc", "lift_at_10pct", "recall_at_10pct"]]
        .mean()
        .round(4)
        .to_dict("index")
    )
    importance_model = _lgbm().fit(d[cols], d["is_late"])
    importance = (
        pd.DataFrame({"feature": cols, "gain": importance_model.booster_.feature_importance("gain")})
        .assign(share=lambda t: t["gain"] / t["gain"].sum())
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    metrics = {
        "feature_set": DEFAULT_FEATURE_SET,
        "fold_starts": FOLD_STARTS,
        "fold_months": FOLD_MONTHS,
        "mean_over_folds": summary,
        "pooled_out_of_fold_lightgbm": ranking_metrics(
            pooled["is_late"].to_numpy(), pooled["risk"].to_numpy()
        ),
    }
    result = {
        "metrics": metrics,
        "folds": folds,
        "ablation": abl,
        "importance": importance,
        "deciles": risk_deciles(pooled["is_late"].to_numpy(), pooled["risk"].to_numpy()),
        "out_of_fold": pooled,
    }

    if save:
        config.ensure_dirs()
        joblib.dump(
            {
                "model": importance_model,
                "features": cols,
                "numeric": numeric,
                "categorical": CATEGORICAL,
                "category_levels": {c: list(d[c].cat.categories) for c in CATEGORICAL},
                "prior_rate": d.attrs["prior_rate"],
                "note": "Trained on all delivered orders; use for ranking risk, not absolute probability.",
            },
            config.MODELS_DIR / "late_delivery_lgbm.joblib",
        )
    return result
