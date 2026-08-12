# Inventory Demand Forecasting

## Executive Summary

Daily unit demand for a beverage distributor's highest-volume SKU was forecast using an XGBoost regression model trained on calendar and lag features, and evaluated against a seasonal-naive baseline on a 90-day holdout period. The model reduced mean absolute error by 20.15% (53.16 units versus 66.57 units) and root mean squared error by 27.91% relative to the baseline, with a mean absolute percentage error of 11.95%.

The forecast was translated into an operational reorder point of 2,578.45 units, combining 2,133.94 units of expected lead-time demand with a 444.51-unit safety stock buffer sized to a 95% service level. Twelve of the 90 holdout days, 13.33% of the period, were flagged as carrying elevated stockout risk.

## Contents

```
Inventory Demand Forecasting/
├── README.md
├── requirements.txt
├── data/
│   ├── generate_data.py        # synthetic data generator (documented, reproducible)
│   └── beverage_distributor_sales.csv
└── notebooks/
    └── demand_forecasting.ipynb
```

## Data

The dataset is synthetically generated to reproduce realistic demand patterns for a multi-SKU beverage distributor: trend, weekly and annual seasonality, promotional spikes, and stockout days are all built into the generator with a fixed random seed for reproducibility. The generation script is included in `data/generate_data.py` so the methodology is fully transparent.

## Method

1. Calendar and lag/rolling-window features are engineered from the daily series.
2. An XGBoost regressor is trained on the first ~2.5 years of data and evaluated on a 90-day holdout.
3. Forecast accuracy is compared against a seasonal-naive baseline (same weekday, prior week).
4. The forecast is converted into a reorder point using a standard lead-time-demand-plus-safety-stock formula, and days with abnormally high forecast demand are flagged as elevated stockout risk.

## How to Run

```
pip install -r requirements.txt
python data/generate_data.py
jupyter notebook notebooks/demand_forecasting.ipynb
```
