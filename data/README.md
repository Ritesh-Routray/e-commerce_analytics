# Data

The data is the **Brazilian E-Commerce Public Dataset by Olist**, published on Kaggle under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). It is not stored in this
repository.

1. Download it from <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>.
2. Unzip the CSVs into `data/raw/` (or point the `OLIST_RAW_DIR` environment variable at them).

Expected tables (either the Kaggle names, e.g. `olist_orders_dataset.csv`, or the shortened
names `olist_orders.csv` are accepted): orders, order items, payments, reviews, customers,
sellers, products, geolocation, and the category translation file. Timestamps may be in ISO
(`2017-10-02 10:56:33`) or day-first (`02-10-2017 10:56`) format; the loader handles both.
