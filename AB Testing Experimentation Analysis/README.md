# A/B Testing / Experimentation Analysis

## Executive Summary

A checkout page redesign was tested against 30,000 randomly assigned users, split 50.18% control and 49.82% treatment. No sample ratio mismatch was detected (chi-square p-value 0.5254), confirming the randomization was sound. The treatment group converted at 12.57% versus 11.38% for control, an absolute lift of 1.18 percentage points and a relative lift of 10.37%. The difference was statistically significant (z = 3.15, p = 0.0016), with a 95% confidence interval for the absolute lift of (0.0045, 0.0192), and the direction of the effect held across all three device segments examined.

Average order value among converting users was also higher in treatment, 44.30 versus 41.90 (t = 4.64, p < 0.001). A post-hoc power check confirmed the achieved sample size of approximately 15,000 users per group exceeded the 11,854 per group required for 80% power at the observed effect size.

## Contents

```
AB Testing Experimentation Analysis/
├── README.md
├── requirements.txt
├── data/
│   ├── generate_data.py       # synthetic data generator (documented, reproducible)
│   └── checkout_ab_test.csv
└── notebooks/
    └── ab_testing_analysis.ipynb
```

## Data

The dataset is a synthetically generated randomized experiment simulating a checkout page redesign test, with a modest, realistic treatment effect built into the generator across three device segments. The generation script is included in `data/generate_data.py` for full transparency of the assumed effect size and randomization.

## Method

1. A sample ratio mismatch check confirms the randomization split matches the intended 50/50 allocation.
2. Conversion rate is compared between groups using a two-proportion z-test, with a 95% confidence interval for the lift.
3. Average order value among converting users is compared as a secondary metric using a two-sample t-test.
4. A post-hoc power check confirms the achieved sample size was sufficient to detect the observed effect at 80% power.

## How to Run

```
pip install -r requirements.txt
python data/generate_data.py
jupyter notebook notebooks/ab_testing_analysis.ipynb
```
