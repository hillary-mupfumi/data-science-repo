"""
Synthetic data generator for the RFM customer segmentation project.

Transaction-level data is generated for an online retail business, with
customers drawn from six archetypes (Champions, Loyal, New, At Risk,
Hibernating, One-Time) so that Recency, Frequency, and Monetary behaviour
varies in a realistic, clusterable way. A fixed random seed is used so the
dataset is reproducible.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 7
rng = np.random.default_rng(RANDOM_SEED)

SNAPSHOT_DATE = pd.Timestamp("2025-01-01")
WINDOW_START = pd.Timestamp("2023-01-01")

COUNTRIES = ["United Kingdom", "Germany", "France", "Ireland", "Netherlands", "Spain"]
COUNTRY_WEIGHTS = [0.55, 0.12, 0.10, 0.08, 0.08, 0.07]

PRODUCTS = [
    ("SKU-1001", "Ceramic Mug", 6.50), ("SKU-1002", "Tote Bag", 9.00),
    ("SKU-1003", "Notebook Set", 12.50), ("SKU-1004", "Desk Lamp", 24.00),
    ("SKU-1005", "Throw Blanket", 32.00), ("SKU-1006", "Candle Trio", 18.00),
    ("SKU-1007", "Wireless Mouse", 21.50), ("SKU-1008", "Water Bottle", 14.00),
    ("SKU-1009", "Photo Frame", 10.00), ("SKU-1010", "Cushion Cover", 15.50),
]

ARCHETYPES = {
    "Champion":     {"share": 0.15, "n_orders": (14, 30), "recency_days": (1, 30),   "order_value": (40, 140)},
    "Loyal":        {"share": 0.20, "n_orders": (7, 14),  "recency_days": (5, 60),   "order_value": (25, 90)},
    "New":          {"share": 0.15, "n_orders": (1, 3),   "recency_days": (1, 45),   "order_value": (20, 80)},
    "At Risk":      {"share": 0.20, "n_orders": (6, 16),  "recency_days": (150, 300),"order_value": (30, 100)},
    "Hibernating":  {"share": 0.20, "n_orders": (2, 6),   "recency_days": (250, 720),"order_value": (15, 60)},
    "One-Time":     {"share": 0.10, "n_orders": (1, 1),   "recency_days": (30, 700), "order_value": (10, 70)},
}

N_CUSTOMERS = 1200

rows = []
customer_id = 10000
invoice_no = 500000

for archetype, cfg in ARCHETYPES.items():
    n_cust = int(N_CUSTOMERS * cfg["share"])
    for _ in range(n_cust):
        customer_id += 1
        n_orders = rng.integers(cfg["n_orders"][0], cfg["n_orders"][1] + 1)
        last_purchase_recency = rng.integers(cfg["recency_days"][0], cfg["recency_days"][1] + 1)
        last_purchase_date = SNAPSHOT_DATE - pd.Timedelta(days=int(last_purchase_recency))

        # Earlier orders are spread backwards from the last purchase date, bounded
        # by the start of the observation window
        span_days = max((last_purchase_date - WINDOW_START).days, 1)
        if n_orders > 1:
            offsets = np.sort(rng.integers(0, span_days, size=n_orders - 1))
            order_dates = [WINDOW_START + pd.Timedelta(days=int(o)) for o in offsets] + [last_purchase_date]
        else:
            order_dates = [last_purchase_date]

        country = rng.choice(COUNTRIES, p=COUNTRY_WEIGHTS)

        for order_date in order_dates:
            invoice_no += 1
            target_value = rng.uniform(cfg["order_value"][0], cfg["order_value"][1])
            n_lines = rng.integers(1, 4)
            remaining_value = target_value
            for line in range(n_lines):
                sku, desc, unit_price = PRODUCTS[rng.integers(0, len(PRODUCTS))]
                qty = max(1, int(round((remaining_value / n_lines) / unit_price)))
                rows.append({
                    "InvoiceNo": invoice_no,
                    "CustomerID": customer_id,
                    "InvoiceDate": order_date,
                    "StockCode": sku,
                    "Description": desc,
                    "Quantity": qty,
                    "UnitPrice": unit_price,
                    "Country": country,
                    "Archetype": archetype,  # retained for validation only, not used as a model feature
                })

df = pd.DataFrame(rows)
df = df.sort_values(["CustomerID", "InvoiceDate"]).reset_index(drop=True)
df.to_csv("online_retail_transactions.csv", index=False)
print(f"Generated {len(df):,} transaction lines for {df['CustomerID'].nunique():,} customers")
print(df.groupby("Archetype")["CustomerID"].nunique())
