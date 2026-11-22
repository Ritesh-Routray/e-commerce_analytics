# Olist Delivery Intelligence

Does breaking the delivery promise cost an e-commerce marketplace? This project analyses ~99k orders from
the Brazilian Olist marketplace to measure **how often deliveries are late, where the time is lost, what late
delivery does to customer reviews, which sellers are least reliable, and whether lateness can be predicted
at checkout.**

Everything is plain Python (pandas, scikit-learn, LightGBM, matplotlib), runs from one command, and every
number in the written report is regenerated from the data.

## Headline results

| Question | Answer |
|---|---|
| How common is late delivery? | 6.8% of delivered orders arrive after the promised date |
| What does it do to reviews? | 4.29 stars on time vs 2.27 late; 54% of late orders get 1 star vs 7% on time |
| Where is the time lost? | Late orders spend ~28 days in carrier transit vs ~8 for on-time orders |
| Is it geographic? | Late rate rises from 4% under 100 km to ~12% at 1,500-2,500 km; AL is worst at 21% |
| Is it steady? | No: waves in Nov 2017 and Feb-Mar 2018 (up to 19% of orders late), calm otherwise |
| Which sellers are at risk? | 30 "Priority" sellers (2x platform late rate) hold ~3% of revenue among scored sellers |
| Does it hurt repeat purchase? | Only slightly (1.6% vs 2.0% repeat in 180 days), but repeat purchase is rare overall |
| Can we predict it at checkout? | Modestly: walk-forward ROC AUC ~0.66, top-decile lift ~2.5x; a plain logistic regression does as well |

Full write-up with charts: [`reports/FINDINGS.md`](reports/FINDINGS.md).

## What is in here

```
src/olist_delivery/
  loading.py    read raw CSVs (Kaggle or re-exported names; ISO or day-first timestamps)
  build.py      one-row-per-order fact table: timing, basket, payment, review, geography
  analysis.py   KPIs, delay vs review, stage breakdown, state/lane/distance, seller scorecard,
                cohort retention, repeat purchase with confidence intervals
  model.py      late-delivery risk model with walk-forward evaluation and feature ablation
  figures.py    charts       report.py   generates reports/FINDINGS.md
scripts/run_pipeline.py     end-to-end run
tests/                      unit tests for the parts that are easy to get subtly wrong
outputs/                    tables (CSV/JSON) and figures (PNG) produced by the pipeline
```

## Method notes

- **Late** means the delivery *calendar date* is after the estimated date (the estimate is always midnight).
- **Seller scorecard** uses a shrunk late rate, pulling small samples toward the platform average, so a seller
  with 2 late orders out of 3 is not ranked above one with 40 late out of 120.
- **Repeat purchase** only counts customers whose first order is at least 180 days before the end of the data,
  to avoid right-censoring bias, and reports Wilson 95% intervals.
- **Model leakage control:** only checkout-time features; each fold trains solely on orders already delivered
  before the fold starts; evaluation is walk-forward over five two-month windows rather than a random split.
  History features (seller/lane late rate as of purchase) were tested and *removed* because they hurt
  out-of-time performance. See the ablation in `outputs/tables/model_ablation.csv`.
- The model ranks risk; it is not a calibrated probability forecaster because the base late rate swings from
  1% to 19% between months.

## Run it

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
# download the data into data/raw/ first, see data/README.md
python scripts/run_pipeline.py            # add --skip-model for a fast run
pytest
```

## Data and credit

Data: *Brazilian E-Commerce Public Dataset by Olist* (Kaggle, CC BY-NC-SA 4.0), see [`data/README.md`](data/README.md).
The data is not redistributed here. Code is released under the MIT license.

Author: Ritesh Routray
