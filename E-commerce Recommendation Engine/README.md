# E-commerce Recommendation Engine

## Executive Summary

A product recommendation engine was built and evaluated on the UCI Online Retail dataset, 541,909 real invoice line items from a UK-based online gift retailer between December 2010 and December 2011. After removing cancellations, guest checkouts without a customer ID, and non-product administrative line items, 396,470 transaction rows remained across 4,334 customers and 3,660 products, worth £8,767,752.65 in total revenue. Three approaches were compared: a non-personalized popularity baseline, item-based collaborative filtering using product co-purchase similarity, and ALS matrix factorization on implicit purchase-quantity feedback. All models were evaluated on a leave-last-basket-out split, where each customer's most recent order was held out and every earlier order was used for training, avoiding the unrealistic optimism of a random split.

ALS matrix factorization was the strongest model, reaching a Precision@10 of 5.26%, a Recall@10 of 3.22%, and a MAP@10 of 2.64% on 2,827 held-out customer baskets, more than three times the popularity baseline's Precision@10 of 1.68%. Item-based collaborative filtering also clearly outperformed the baseline, reaching a Precision@10 of 4.67%. The results confirm that purchase-history-based personalization adds real, measurable value on top of simply recommending bestsellers to every customer.

## Contents

```
E-commerce Recommendation Engine/
├── README.md
├── requirements.txt
├── data/
│   └── online_retail_2010_2011.csv.gz   # UCI Online Retail dataset, gzip-compressed
└── notebooks/
    ├── 01_data_exploration_and_cleaning.ipynb   # EDA, data quality checks, cleaning
    └── 02_recommendation_models.ipynb           # baseline, item CF, ALS, evaluation
```

## Data

The dataset is the UCI Machine Learning Repository's "Online Retail" dataset (https://archive.ics.uci.edu/dataset/352/online+retail), covering all transactions for a UK-registered, non-store online retailer selling all-occasion gift-ware between 1 December 2010 and 9 December 2011. Each row is a single product line within a customer invoice, giving invoice number, product code, description, quantity, unit price, invoice date, customer ID, and country.

Cleaning removed 9,288 cancelled-order rows (invoice numbers prefixed with 'C'), 135,080 rows with no customer ID attached, 10,624 rows with non-positive quantities, 2,517 rows with non-positive prices, and administrative line items such as postage and manual adjustments. The cleaned dataset retains 73.16% of the raw rows. Of the 4,334 remaining customers, 65.27% placed two or more orders, which is what makes purchase-history-based personalization possible; only customers with repeat orders can be evaluated under a held-out-last-basket protocol, since a one-time customer has no prior history to learn from.

## Method

1. The cleaned transaction log was aggregated into a customer-by-product interaction matrix (4,334 customers by 3,660 products, 1.68% dense), using the training portion only.
2. Each customer's single most recent invoice was held out as the test set; all earlier invoices formed the training set. Held-out products that never appeared in the training catalogue were excluded from evaluation, since no model could plausibly recommend an unseen item.
3. A popularity baseline recommended the same top-10 most-purchased products, by distinct buyer count, to every customer, excluding items they had already bought.
4. Item-based collaborative filtering computed cosine similarity between products from a binarized co-purchase matrix, then scored each candidate product by its summed similarity to everything the customer had already bought.
5. ALS matrix factorization (50 latent factors, 20 iterations, regularization 0.05) was trained on implicit-feedback confidence weights, following the Hu, Koren and Volinsky (2008) formulation, where confidence equals 1 plus twice the log of one plus quantity purchased.
6. All three models were scored with Precision@10, Recall@10, and Mean Average Precision@10 on the held-out baskets of the 2,827 evaluable test customers.

## Results

| Model | Precision@10 | Recall@10 | MAP@10 |
|---|---|---|---|
| Popularity baseline | 1.68% | 0.89% | 0.56% |
| Item-based collaborative filtering | 4.67% | 2.76% | 2.40% |
| ALS matrix factorization | 5.26% | 3.22% | 2.64% |

ALS matrix factorization improved Precision@10 by 213.10% relative to the popularity baseline and MAP@10 by 371.86%. Item-based collaborative filtering came close to ALS on Precision@10 while being simpler to explain and maintain, since its recommendations can always be traced back to a specific similar product rather than a latent factor. Precision and recall in the low single digits are expected and normal for this task: with 3,660 possible products and a 10-item recommendation slate, correctly guessing even one or two items from a customer's next basket represents a large improvement over chance.

## Limitations

Feedback is implicit (purchase quantity) rather than an explicit rating, so the models cannot distinguish a product a customer loved from one bought once and never repurchased. Evaluation is only possible for the 65.27% of customers with two or more orders; the recommender has no purchase history to personalize against for the remaining one-time customers, who would need a content-based or popularity-based cold-start strategy instead. The dataset covers a single UK retailer over one year, so the learned co-purchase and latent-factor patterns are specific to that catalogue and shopping season, and would need retraining on a new retailer's data before reuse elsewhere.

## How to Run

```
pip install -r requirements.txt
jupyter notebook notebooks/01_data_exploration_and_cleaning.ipynb
jupyter notebook notebooks/02_recommendation_models.ipynb
```
