# Entertainment Industry Performance Analytics

## Executive Summary

Two real, independently sourced datasets were combined to analyze the film industry from both a studio-economics and an individual-viewer perspective. The TMDB 5000 Movie Dataset provided real budget and revenue figures for 3,213 films with complete financial data, released between 1916 and 2016, of which 75.57% recovered their production budget. The MovieLens ml-latest-small dataset provided 100,836 real, timestamped ratings from 610 real users collected between 1996 and 2018. The two datasets were joined through each movie's TMDB identifier, giving 65,355 ratings across 2,803 financially tracked films and covering all 610 users.

A publisher-optimization framework, originally designed for comparing search-engine-marketing publishers on cost-per-click and probability of booking, was translated onto genre and studio performance, replacing publisher metrics with average budget, probability of profitability, and return on ad spend equivalent. Blumhouse Productions stood out with a 19.81 return on investment across 20 films, driven by a 95.00% probability of profitability against an average budget of only $4.31 million, versus major studios spending 15 to 40 times more per film for comparable or lower returns.

A second analysis modeled individual viewer engagement using the BG/NBD probabilistic framework, the same model used for monetary customer lifetime value, applied here to engagement frequency and recency since MovieLens contains no per-user monetary transactions. The model predicted holdout-period rating volume with a Pearson correlation of 0.83 against actual behavior. Viewers in the top predicted-engagement quartile engaged with content carrying a 45.89% higher average real budget than viewers in the bottom quartile, a modest but genuine link between individual engagement value and the content economics identified in the studio analysis.

## Contents

```
Entertainment Industry Performance Analytics/
├── README.md
├── requirements.txt
├── data/
│   ├── tmdb_5000_movies.csv   # TMDB 5000 Movie Dataset, real budget/revenue/genre data
│   ├── ratings.csv            # MovieLens ml-latest-small, real timestamped user ratings
│   ├── movies.csv             # MovieLens movie titles and genres
│   └── links.csv              # MovieLens movieId to TMDB id mapping, used to join the two datasets
└── notebooks/
    ├── 01_data_exploration_and_cleaning.ipynb      # load, clean, and join both datasets
    ├── 02_genre_studio_optimization.ipynb          # publisher-optimization framework applied to genre/studio
    └── 03_engagement_lifetime_value.ipynb          # BG/NBD engagement model and loyalty profile
```

## Data

Two real datasets were used, chosen because they can be joined at the individual film level through a shared TMDB identifier.

The TMDB 5000 Movie Dataset (originally sourced from The Movie Database's public API) contains 4,803 films with budget, revenue, genre, production company, release date, popularity, and audience vote data. Rows with a budget below $10,000 or a revenue of zero, TMDB's placeholder for unknown financial data, were excluded, leaving 3,213 films with genuine, complete financial records.

The MovieLens ml-latest-small dataset, published by GroupLens Research, contains 100,836 real ratings from 610 real users across 9,742 movies, collected between March 1996 and September 2018. Each rating carries a genuine Unix timestamp, making it suitable for frequency and recency modeling.

The two datasets were joined using `links.csv`, which maps each MovieLens `movieId` to the corresponding TMDB `id`. Of the 9,733 MovieLens movies with a TMDB identifier, 3,536 also appear in the cleaned financial dataset, yielding 65,355 individual ratings that can be traced to genuine box office outcomes.

## Method

1. TMDB budget and revenue fields were cleaned of placeholder zeros, genres and production companies were parsed out of their JSON-encoded fields, and profit and return on investment were computed per film.
2. MovieLens ratings were joined to TMDB identifiers through `links.csv`, producing both a full ratings table (for the engagement model) and a financially linked subset (for tying engagement back to content economics).
3. Genre-level and studio-level metrics were computed, adapting a search-engine-marketing publisher-optimization framework: average budget stood in for cost per click, probability of profitability (share of films that earned back their budget) stood in for probability of booking, and return on ad spend became net revenue divided by total budget. Each genre and studio (minimum 20 and 10 films respectively, for statistical reliability) was placed into one of four funding-strategy quadrants against the cross-group average probability of profitability and average budget.
4. An individual-level engagement model was fit using BG/NBD (Beta-Geometric/Negative Binomial Distribution), the standard non-contractual customer lifetime value model, using each user's real rating timestamps as the frequency and recency signal. The model was calibrated on the first 80% of the 22-year observation window and validated by predicting each user's rating volume in the remaining holdout period against what they actually did.
5. The validated model was refit on each user's complete history to predict expected engagement volume over a forward 180-day window and the probability that the user remains an active rater.
6. Each user's predicted engagement was joined back to the real budget and revenue of the financially tracked movies they rated, to test whether engagement value relates to a preference for higher-budget content.

## Results

| Metric | Value |
|---|---|
| Films with complete real financial data | 3,213 |
| Share of tracked films that were profitable | 75.57% |
| Top studio by return on investment (Blumhouse Productions, 20 films) | 19.81x, 95.00% profitable, $4.31M avg. budget |
| Top genre by return on investment (Documentary, 37 films) | 3.76x |
| BG/NBD holdout Pearson correlation (predicted vs. actual engagement) | 0.83 |
| BG/NBD holdout mean absolute error | 0.98 ratings |
| Avg. budget engaged with, top engagement quartile vs. bottom quartile | $65.33M vs. $44.78M (+45.89%) |

The genre and studio quadrant analysis confirmed the same pattern the source framework identifies in advertising: a small, low-cost, high-conversion-probability performer can substantially outperform high-spend competitors on efficiency. Blumhouse Productions' horror-focused, low-budget model produced the highest return on investment in the dataset among studios with a meaningful film count, while several major studios with high probability of profitability were classified into the "adjust tactics" quadrant on account of their high average budgets, mirroring the framework's treatment of expensive, high-converting advertising publishers.

The BG/NBD engagement model, fit purely on real timestamped activity with no monetary input, produced holdout predictions correlated at 0.83 with actual future engagement, comparable to validation results reported for the same model on monetary purchase data. The relationship between predicted engagement and the budget of content engaged with was modest across the middle of the distribution (Pearson correlation of 0.09) but clear at the extremes, with the top engagement quartile favoring meaningfully higher-budget content than the bottom quartile.

## Limitations

MovieLens contains no per-user monetary transactions, so the individual-level model predicts engagement volume rather than a dollar-denominated lifetime value; extending it to true financial customer lifetime value would require subscriber-level revenue data that is not publicly available for any real streaming platform. The genre and studio financial analysis is limited to theatrical releases with data in TMDB, so it does not reflect streaming-native or made-for-television content. The MovieLens panel is small relative to a commercial platform (610 users), so the engagement model's holdout results, while directionally validated, would benefit from confirmation on a larger user base before being used for real budget decisions. Genre and studio groupings are drawn from each film's full genre and production company list, so a single film can contribute to multiple groups.

## How to Run

```
pip install -r requirements.txt
jupyter notebook notebooks/01_data_exploration_and_cleaning.ipynb
jupyter notebook notebooks/02_genre_studio_optimization.ipynb
jupyter notebook notebooks/03_engagement_lifetime_value.ipynb
```
