"""
Synthetic data generator for the inventory demand forecasting project.

A synthetic daily sales dataset is generated for a regional beverage
distributor carrying six SKUs across three categories. The generator
injects trend, weekly seasonality, annual seasonality, promotional spikes,
and occasional stockout days, so the resulting series behaves like a real
demand signal rather than pure noise. A fixed random seed is used so the
dataset is reproducible.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

START_DATE = "2022-01-01"
END_DATE = "2024-12-31"
dates = pd.date_range(START_DATE, END_DATE, freq="D")
n_days = len(dates)

SKUS = [
    {"sku": "CSD-COLA-500ML", "category": "Carbonated Soft Drinks", "base_demand": 420, "unit_cost": 0.85},
    {"sku": "CSD-ORANGE-500ML", "category": "Carbonated Soft Drinks", "base_demand": 260, "unit_cost": 0.85},
    {"sku": "WTR-STILL-1L", "category": "Bottled Water", "base_demand": 510, "unit_cost": 0.45},
    {"sku": "WTR-SPARK-1L", "category": "Bottled Water", "base_demand": 190, "unit_cost": 0.55},
    {"sku": "JCE-APPLE-1L", "category": "Juices", "base_demand": 150, "unit_cost": 1.10},
    {"sku": "JCE-MANGO-1L", "category": "Juices", "base_demand": 110, "unit_cost": 1.15},
]

# Annual growth rate applied as a gentle linear trend across the 3-year window
ANNUAL_GROWTH = 0.06

# Day-of-week multipliers: outlets restock ahead of the weekend, so Thu-Sat
# are the strongest days and Sunday-Monday are the weakest.
DOW_MULTIPLIER = {0: 0.90, 1: 0.88, 2: 0.95, 3: 1.08, 4: 1.18, 5: 1.20, 6: 0.85}

def month_multiplier(month, category):
    """Annual seasonality: soft drinks and water peak in warmer months,
    juices are comparatively flat with a small holiday bump in December."""
    if category in ("Carbonated Soft Drinks", "Bottled Water"):
        # Peak around June-August, trough around December-January
        return 1.0 + 0.35 * np.sin((month - 4) / 12 * 2 * np.pi)
    return 1.0 + 0.12 * np.sin((month - 4) / 12 * 2 * np.pi) + (0.10 if month == 12 else 0.0)

rows = []
for sku_info in SKUS:
    sku, category, base, unit_cost = (
        sku_info["sku"], sku_info["category"], sku_info["base_demand"], sku_info["unit_cost"]
    )

    # Promotional days: about 3% of days, randomly placed, demand lifted 30-70%
    promo_days = set(rng.choice(n_days, size=int(n_days * 0.03), replace=False))

    # Stockout days: about 1.5% of days, demand recorded as zero because no stock was available
    stockout_days = set(rng.choice(n_days, size=int(n_days * 0.015), replace=False))

    for i, date in enumerate(dates):
        trend = 1.0 + ANNUAL_GROWTH * (i / 365.0)
        dow = DOW_MULTIPLIER[date.dayofweek]
        month_mult = month_multiplier(date.month, category)
        noise = rng.normal(1.0, 0.09)

        expected = max(base * trend * dow * month_mult * noise, 5)

        if i in promo_days:
            expected *= rng.uniform(1.3, 1.7)

        units_sold = rng.poisson(expected)

        stockout_flag = i in stockout_days
        if stockout_flag:
            units_sold = 0

        rows.append(
            {
                "date": date,
                "sku": sku,
                "category": category,
                "units_sold": int(units_sold),
                "unit_cost": unit_cost,
                "is_promo_day": i in promo_days,
                "is_stockout_day": stockout_flag,
            }
        )

df = pd.DataFrame(rows)
df = df.sort_values(["sku", "date"]).reset_index(drop=True)
df.to_csv("beverage_distributor_sales.csv", index=False)
print(f"Generated {len(df):,} rows across {len(SKUS)} SKUs from {START_DATE} to {END_DATE}")
print(df.groupby("sku")["units_sold"].agg(["mean", "std"]).round(2))
