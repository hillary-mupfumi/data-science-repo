# RFM Customer Segmentation

## Executive Summary

Two years of transaction history for 1,200 customers were used to build a Recency, Frequency, and Monetary (RFM) profile per customer, which was then segmented using both a rule-based quartile scoring method and unsupervised k-means clustering. Six rule-based segments were identified, with Loyal Customers (439 customers, 36.58% of the base) and Champions (138 customers, 11.50%) together accounting for 48.08% of customers but 86.18% of total revenue (639,400.50 of 741,973.00). Hibernating customers made up 32.58% of the base with an average recency of 437.00 days, representing the largest reactivation opportunity.

K-means clustering on standardized RFM values selected four clusters as optimal (silhouette score 0.51), and the resulting clusters aligned closely with the rule-based segments for Loyal Customers, Hibernating, and New / Low Frequency, which supports the rule-based segmentation as a reasonable operational simplification.

## Contents

```
RFM Customer Segmentation/
├── README.md
├── requirements.txt
├── data/
│   ├── generate_data.py            # synthetic data generator (documented, reproducible)
│   └── online_retail_transactions.csv
└── notebooks/
    └── rfm_segmentation.ipynb
```

## Data

The dataset is synthetically generated transaction-level retail data spanning six customer archetypes (Champions, Loyal, New, At Risk, Hibernating, One-Time), so that Recency, Frequency, and Monetary behaviour vary in a realistic, clusterable way. The archetype label is retained only for validating the resulting segments and is not used as a model feature. The generation script is included in `data/generate_data.py`.

## Method

1. Recency, Frequency, and Monetary values are computed per customer relative to a snapshot date.
2. Each dimension is scored into quartiles and combined into a rule-based segment label.
3. RFM values are standardized and clustered with k-means, with the cluster count chosen by silhouette score, as an independent validation of the rule-based segments.

## How to Run

```
pip install -r requirements.txt
python data/generate_data.py
jupyter notebook notebooks/rfm_segmentation.ipynb
```
