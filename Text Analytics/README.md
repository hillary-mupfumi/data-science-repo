# Text Analytics

## Overview

This folder contains two independent text mining and NLP projects, both built as an end-to-end pipeline in R (tidytext, quanteda, topicmodels, tidymodels) against unstructured text pulled from MongoDB: Airbnb listing descriptions across nine countries, and Nike consumer discourse from Reddit and YouTube. Both projects apply the same core toolkit (tokenization, lemmatization, TF/IDF, bigram and word-correlation analysis, Bing/NRC sentiment lexicons, LDA topic modeling, and a Naive Bayes classifier) to different business questions, so they are documented together here but are otherwise standalone.

## Contents

```
Text Analytics/
├── README.md
├── Airbnb Listings Intelligence/
│   ├── Airbnb Text Mining Intelligence.Rmd
│   ├── Airbnb Project Report.pdf         # written report + Power BI dashboard writeup
│   └── Airbnb PowerBI Dashboard.png      # dashboard screenshot
└── Nike Brand Intelligence/
    ├── Nike brand Intelligence.Rmd
    └── Nike Brand Intelligence-1.pptx    # executive-facing slide deck
```

## Security Note

Both `.Rmd` files, as originally written, had a live MongoDB connection string (username and password) hardcoded in plain text, and the Nike file additionally had a real YouTube Data API key hardcoded in its appendix pipeline script. Both have been redacted in the versions here: the code now reads `MONGO_URI`, `DB_NAME`, and `YOUTUBE_API_KEY` from environment variables (`Sys.getenv(...)`), meant to be set locally in a `.Renviron` file that is excluded from version control. Since those credentials were exposed in files that were about to be pushed to a public GitHub repository, it is worth rotating the MongoDB password and regenerating the YouTube API key before pushing, since the original values may already be compromised regardless of this cleanup.

---

## Project 1: Airbnb Listings Descriptions Intelligence

### Executive Summary

Airbnb listing descriptions were pulled from a MongoDB sample dataset (`sample_airbnb.listingsAndReviews`) spanning nine countries: Australia, Brazil, Canada, China, Hong Kong, Portugal, Spain, Turkey, and the United States. The companion Power BI dashboard and written report cover 5,555 listings, an average host response rate of 94.50%, a median nightly price of $155.00, and roughly 101,000 total reviews. Review scores vary by market: the United States leads at 94%, followed by Australia and Canada at 93%, Spain at 91%, and China at 89%. Pricing scales with property type, from $51.00 for shared rooms and $55.00 for private rooms up to $140.00 for entire homes and above $300.00 for resorts and chalets.

After tokenization, stopword removal, and lemmatization, the text corpus contained 365,179 tokens and 26,648 unique lemmas, with 202,742 bigrams (103,566 unique). Bing lexicon sentiment analysis shows positive language outnumbering negative by roughly 2 to 1 across every market (around 20,000 positive versus 10,000 negative token instances per country), confirming that hosts write descriptions as marketing copy. A four-topic LDA model separated listings into Apartment Features & Amenities, Guest Experience & Host Hospitality, Location & Transportation, and Neighborhood & Local Attractions.

A Naive Bayes classifier was trained to predict whether a listing's review score is 90 or above from its description text alone. The published notebook does not print a formal accuracy or confusion-matrix score for this model; the result reported is qualitative, word-level evidence (terms like "fantastic" and "duplex" skew toward high-rated listings, while generic terms like "apartment" appear across the board), so the classifier should be read as exploratory rather than as a validated predictive model.

### Data

The dataset is MongoDB's public `sample_airbnb` sample database (`listingsAndReviews` collection), accessed live via `mongolite`. It is not included in this repository; a `MONGO_URI` pointing at a MongoDB Atlas cluster with that sample dataset loaded is required to re-run the notebook.

### Method

1. Listing description, country, property type, price, and review score are pulled from MongoDB and flattened into a data frame.
2. Text is lowercased, stripped of URLs/mentions/punctuation, and contractions are expanded before tokenization.
3. Tokens are lemmatized and stripped of stopwords (standard `tidytext::stop_words` plus a small custom list).
4. Word frequency, bigram, IDF-by-country, and word-correlation (US vs. Spain) analyses are run on the cleaned tokens.
5. Bing lexicon sentiment and a 4-topic Gibbs-sampled LDA model are fit on the token set.
6. A Naive Bayes classifier is trained on a document-feature matrix of the cleaned descriptions to predict a binary high/low rating label.

### How to Run

```r
# In a local .Renviron (not committed to git):
# MONGO_URI="mongodb+srv://<user>:<password>@<cluster>/sample_airbnb?retryWrites=true&w=majority"
# DB_NAME="sample_airbnb"

install.packages(c("mongolite","tidyverse","tidytext","tm","textstem",
                    "scales","e1071","quanteda","widyr","igraph","ggraph","topicmodels"))
rmarkdown::render("Airbnb Listings Intelligence/Airbnb Text Mining Intelligence.Rmd")
```

Published, rendered version: https://rpubs.com/hmupfumi/1414013

---

## Project 2: Nike Brand Intelligence

### Executive Summary

Consumer discourse about Nike was collected from Reddit's r/Sneakers community and YouTube product-review comments and stored in MongoDB (`nike_reviews_db`). After deduplication, the corpus contained 2,722 documents: 818 from Reddit (average score 2.6) and 1,904 from YouTube (average score 3.5). Tokenization and lemmatization produced 17,199 tokens and 4,102 unique lemmas, with 5,670 bigrams (4,787 unique).

A four-topic LDA model shows Sneaker Style & Fit as the dominant conversation theme at 41% of total topic weight, followed by Product Experience & Comfort at 29%, Retail & Sneaker Culture at 20%, and Digital Content & Athlete Endorsement at just 10%, suggesting athlete-endorsement spend generates comparatively little organic conversation. NRC emotion analysis shows Trust as Nike's dominant consumer emotion on both platforms, with Reddit skewing more toward Anger than YouTube. TF-IDF brand differentiation against Reddit-mentioned competitors shows Nike's language is experiential and sensory ("wear," "lace," "comfortable"), Adidas's is product-line-driven ("yeezy," "ultraboost"), and Under Armour's distinctive terms are dominated by quality and injury complaints ("suffer," "metatarsal," "deficiency"). A pricing/sustainability keyword scan finds "premium," "expensive," "cheap," and "worth" as the leading pricing terms, versus only "green" and "carbon" registering with any real frequency on sustainability, pointing to a shallow, surface-level sustainability conversation.

A three-class Naive Bayes classifier (Low/Medium/High star category, TF-IDF features, 80/20 split) reports 86.20% accuracy, but this number is misleading on its own: the confusion matrix shows every one of the 545 test reviews was predicted as "Low," including the 20 actual Medium and 55 actual High reviews. Kappa is 0.00, confirming the model has no real skill beyond exploiting the fact that "Low" is the majority class in this sample; it cannot currently distinguish Medium- or High-rated reviews and would need a rebalanced training sample before being used for anything like real-time churn flagging.

### Data

Text was collected from Reddit (r/Sneakers, via `RedditExtractoR`) and YouTube (Data API v3 comment threads), stored in MongoDB collections `nike_reviews_reddit` and `nike_reviews_youtube`, and accessed live via `mongolite`. Neither the MongoDB database nor the raw scraped text is included in this repository; a `MONGO_URI` and, to re-run the appendix collection script, a `YOUTUBE_API_KEY` are required.

### Method

1. Reddit and YouTube comments are pulled from MongoDB, combined, deduplicated, and filtered to comments longer than 15 characters.
2. The same cleaning/tokenization/lemmatization pipeline used in the Airbnb project is applied.
3. Word frequency, bigram, document-term-matrix, word-correlation (Reddit vs. YouTube), pairwise co-occurrence, and IDF analyses are run.
4. Bing and NRC lexicon sentiment, a 4-topic Gibbs-sampled LDA model, and TF-IDF brand differentiation (Nike vs. Adidas vs. Under Armour) are computed.
5. A keyword co-occurrence network and a pricing/sustainability keyword scan are built directly from the token set.
6. A 3-class Naive Bayes classifier (`tidymodels` + `klaR`) is trained on TF-IDF features to predict a Low/Medium/High star category from review text.

### How to Run

```r
# In a local .Renviron (not committed to git):
# MONGO_URI="mongodb+srv://<user>:<password>@<cluster>/nike_reviews_db?retryWrites=true&w=majority"
# DB_NAME="nike_reviews_db"
# YOUTUBE_API_KEY="<your-youtube-data-api-v3-key>"   # only needed for the appendix collection script

install.packages(c("mongolite","tidyverse","tidytext","tm","textstem","scales","e1071",
                    "quanteda","tidymodels","textrecipes","discrim","parsnip","rsample",
                    "klaR","widyr","igraph","ggraph","topicmodels"))
rmarkdown::render("Nike Brand Intelligence/Nike brand Intelligence.Rmd")
```

Published, rendered version: https://rpubs.com/hmupfumi/1413884

Executive slide deck: `Nike Brand Intelligence/Nike Brand Intelligence-1.pptx`
