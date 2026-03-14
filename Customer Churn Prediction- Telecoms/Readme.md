# 📡 Telecom Customer Churn Prediction

> A machine learning pipeline to identify customers at risk of churning, enabling proactive retention strategies in the telecom industry.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Recommendations](#recommendations)

---

## Overview

Customer churn is one of the most costly problems in the telecom industry. This project builds and evaluates classification models to predict which customers are likely to churn, so retention teams can intervene before it's too late.

**Primary metric: Recall** — due to class imbalance (26.49% churn rate), correctly identifying actual churners matters more than overall accuracy. A missed churner is a lost customer; a false alarm is just an unnecessary retention offer.

---

## Dataset

**Source:** [IBM Watson Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/palashfendarkar/wa-fnusec-telcocustomerchurn)

| Property | Value |
|---|---|
| Raw records | 7,043 |
| Clean records (after dedup + nulls) | 7,010 |
| Features | 21 |
| Churn rate | 26.49% |

### Features

| Variable | Type | Description |
|---|---|---|
| `CustomerID` | ID | Unique customer identifier (dropped) |
| `gender` | Categorical | Male / Female |
| `SeniorCitizen` | Binary | 1 = senior citizen |
| `Partner` | Binary | Has a partner |
| `Dependents` | Binary | Has dependents |
| `tenure` | Numeric | Months with the company |
| `PhoneService` | Binary | Has phone service |
| `MultipleLines` | Categorical | Single / Multiple / No service |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | Binary | Has online security add-on |
| `OnlineBackup` | Binary | Has online backup add-on |
| `DeviceProtection` | Binary | Has device protection add-on |
| `TechSupport` | Binary | Has tech support add-on |
| `StreamingTV` | Binary | Streams TV |
| `StreamingMovies` | Binary | Streams movies |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | Binary | Uses paperless billing |
| `PaymentMethod` | Categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| `MonthlyCharges` | Numeric | Monthly charge ($) |
| `TotalCharges` | Numeric | Cumulative charges ($) |
| `Churn` | Binary | **Target** — churned Yes / No |

---

## Project Structure

```
telecom-churn/
├── Telecom_Customer_Churn_Prediction.ipynb   # Main analysis notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv      # Raw dataset
├── images/                                    # Exported EDA & model charts
│   ├── customer_demographics_distribution.png
│   ├── service_features_distribution.png
│   ├── tenure_contract_distribution.png
│   ├── billing_and_charges_distribution.png
│   ├── Customer Churn Distribution.png
│   ├── Relationship between Customer Demographics and Churn.png
│   ├── Relationship between Services and Churn.png
│   ├── Relationship between Contract_Tenure and Churn.png
│   ├── Relationship between Billing and Charges with Churn.png
│   └── Confusion Matrix Heatmap.png
└── README.md
```

---

## Installation

**Requirements:** Python 3.8+

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Then open the notebook:

```bash
jupyter notebook Telecom_Customer_Churn_Prediction.ipynb
```

---

## Methodology

### 1. Data Cleaning
- Dropped `CustomerID` (no analytical value)
- Removed 22 duplicate rows
- Cast `TotalCharges` from object to numeric; dropped 11 resulting nulls

### 2. EDA
Explored distributions and churn relationships across four themes:
- Customer demographics (gender, senior citizen, partner, dependents)
- Services (phone, internet, security, streaming)
- Tenure & contract type
- Billing & charges

### 3. Preprocessing
- **Outlier removal** via IQR method on `tenure`, `MonthlyCharges`, `TotalCharges`
- **Binary encoding** — Yes/No columns and collapsed "No internet/phone service" → "No"
- **One-hot encoding** — `PaymentMethod`, `Contract`, `InternetService` (drop-first)
- **StandardScaler** normalization on numeric columns
- **80/20 train-test split** (`random_state=42`)

### 4. Modelling
Three classifiers were tuned via `GridSearchCV` (3-fold CV, scoring=`recall`):

| Model | Key Hyperparameters Searched |
|---|---|
| Decision Tree | `max_depth`, `min_samples_leaf`, `min_samples_split`, `criterion` |
| Random Forest | Same as above + `n_estimators` |
| KNN | `n_neighbors`, `weights`, `algorithm` |

---

## Results

| Model | Train Recall | Test Recall | Notes |
|---|---|---|---|
| **Decision Tree** | **0.65** | **0.64** | ✅ Best generalization |
| Random Forest | 0.63 | 0.53 | Moderate overfitting |
| KNN | 0.99 | 0.54 | Severe overfitting |

### Best Hyperparameters

**Decision Tree** (recommended model)
```python
DecisionTreeClassifier(
    criterion='entropy',
    max_depth=6,
    min_samples_leaf=6,
    min_samples_split=2,
    random_state=0
)
```

**Random Forest**
```python
RandomForestClassifier(
    criterion='entropy',
    max_depth=10,
    min_samples_leaf=2,
    min_samples_split=6,
    n_estimators=300,
    random_state=0
)
```

**KNN**
```python
KNeighborsClassifier(
    algorithm='auto',
    n_neighbors=10,
    weights='distance'
)
```

### Confusion Matrix — Decision Tree (test set)

|  | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | True Negatives | False Positives |
| **Actual: Churn** | 115 (missed) | **206 (caught)** |

The Decision Tree catches the most actual churners (206 TP) while missing the fewest (115 FN) — directly maximizing the business objective.

---

## Key Findings

**Demographics:** Gender has no meaningful impact on churn. Single customers without partners or dependents churn at higher rates. Senior citizens are actually more loyal.

**Services:** Customers without Online Security, Device Protection, or Tech Support churn more. Fiber optic internet customers show higher churn than DSL users.

**Tenure & Contract:** Churn is heavily concentrated in the first 5 months. Month-to-month contracts show dramatically higher churn than one- or two-year contracts. There is a clear inverse relationship between tenure and churn probability.

**Billing:** Higher monthly charges correlate with higher churn. Higher total charges (a proxy for long tenure) correlate with lower churn.

**Top churn predictors (by correlation):**
- 🔴 `InternetService_Fiber optic` (+0.31) — risk factor
- 🔴 `PaymentMethod_Electronic check` (+0.30) — risk factor
- 🟢 `tenure` (−0.35) — protective factor
- 🟢 `Contract_Two year` (−0.30) — protective factor

---

## Recommendations

1. **Early-lifecycle intervention** — the first 90 days are the highest-risk window. Onboarding campaigns and bundled service incentives can anchor new customers before they churn.

2. **Incentivize contract upgrades** — offer discounts or perks to move month-to-month customers onto one- or two-year plans.

3. **Cross-sell value-added services** — promote Online Security, Tech Support, and Device Protection to customers without them; these raise switching costs and deepen engagement.

4. **Monitor fiber optic + electronic check customers** — this combination carries the highest churn risk and warrants targeted loyalty outreach.

5. **Flexible pricing for at-risk segments** — high monthly charges are a churn driver; loyalty discounts for longer-tenure customers can reduce price-driven attrition.

---

*Dataset source: [Kaggle — WA_Fn-UseC_-Telco-Customer-Churn](https://www.kaggle.com/datasets/palashfendarkar/wa-fnusec-telcocustomerchurn)*
