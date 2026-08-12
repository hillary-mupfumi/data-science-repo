# Urban Heat Index Prediction

## Executive Summary

Urban Heat Island (UHI) intensity was classified into three levels (Low, Medium, High) using satellite spectral indices and building footprint features, as part of the EY Urban Heat Island Data Challenge. Ground-truth vehicle and bicycle temperature traverses from Santiago, Chile (21,661 points) and Rio de Janeiro, Brazil (24,519 points, after removing 3,969 duplicate feature rows) were used to train and benchmark classification models, with the goal of generalising predictions to Freetown, Sierra Leone (14,105 unlabeled validation points), a city with no ground-truth temperature data.

Thirteen features were extracted per location from Sentinel-2 satellite imagery (NDVI, NDBI, NDWI, NDMI, BSI, and their within-buffer variability) and from 3D-GloBFP building footprint data (density, coverage, average size, and height). On the labeled benchmark cities, a K-Nearest Neighbours model reached a macro F1 score of 0.72 on Santiago and 0.90 on Rio de Janeiro, while a Residual MLP neural network reached 0.90 on Rio de Janeiro, the strongest single-city result. For Freetown, where no full ground truth exists, a tree ensemble trained on the combined Santiago and Rio data plus a small set of confirmed and inferred Freetown labels reached a cross-validated F1 of 0.86 after three rounds of pseudo-labeling, with spatial smoothing applied to correct geographically implausible predictions. Built-up intensity (NDBI) and building height were consistently the strongest predictors of UHI class across all three cities, ahead of vegetation and moisture indices.

## Contents

```
Urban Heat Index Prediction/
├── README.md
├── requirements.txt
├── data/
│   ├── chile_combined.csv              # Santiago features + UHI class labels
│   ├── brazil_combined.csv             # Rio de Janeiro features + UHI class labels
│   ├── sierra_combined.csv             # Freetown features, validation set (unlabeled)
│   └── freetown_labeled_enhanced.csv   # Freetown points with confirmed/recovered labels
├── notebooks/
│   ├── 01_feature_extraction.ipynb     # spectral index + building feature extraction
│   └── 02_modeling_and_prediction.ipynb # model benchmarking, Freetown pipeline, results
└── docs/
    └── UHI_Challenge_Overview.pdf      # original challenge brief from EY
```

## Data

The underlying ground-truth temperature data was collected by CAPA Strategies via vehicle and bicycle traverses on a single afternoon in each city: Santiago, Chile, on 20 January 2024, and Rio de Janeiro, Brazil, on 27 January 2023. Each point's UHI index was computed as its temperature relative to the mean temperature across all points collected that day, and classed as Low (index at or below 0.98), Medium (between 0.98 and 1.02), or High (index at or above 1.02). Freetown, Sierra Leone data was collected on 24 January 2023 but withheld as the unlabeled validation target for the original challenge; `freetown_labeled_enhanced.csv` contains a partial set of Freetown labels recovered for this project, of which only 3,071 of 11,179 points are directly confirmed, with the remainder inferred through pseudo-labeling and nearest-neighbour spatial inference.

Two feature sources were combined for every point: Sentinel-2 satellite imagery (10m resolution, cloud-filtered median mosaics) for spectral indices, and the 3D-GloBFP global building footprint dataset for building density, coverage, size, and height within a 100m buffer.

## Method

1. Spectral indices (NDVI, NDBI, NDWI, NDMI, BSI, and within-buffer standard deviations) were extracted from Sentinel-2 GeoTIFFs using a 100m buffer mean around each ground-truth coordinate, and building features (density, coverage, average footprint size, mean and area-weighted height) were extracted via a spatial join to the building footprint shapefiles.
2. Random Forest, XGBoost, KNN, MLP, and a custom Residual MLP neural network were benchmarked independently on Santiago and Rio de Janeiro using a 70/30 train/test split, to confirm the models worked correctly and to establish single-city baselines before attempting cross-city generalisation.
3. For Freetown, features were scaled with a single StandardScaler fitted across all three cities combined, which reduces the risk of absolute feature-value differences between regions being mistaken for genuine signal by the model.
4. A tree ensemble (LightGBM, XGBoost, ExtraTrees, Random Forest) and the Residual MLP were trained on the combined Santiago and Rio data plus the confirmed Freetown labels, then improved through three rounds of pseudo-labeling, where high-confidence Freetown predictions were iteratively added back into the training set.
5. Spatial smoothing was applied to the final Freetown predictions, replacing low-confidence predictions with the majority vote of their seven nearest geographic neighbours, to correct physically implausible Low-to-High adjacent predictions.

## Results

| City | Rows (deduplicated) | Best model | Macro F1 |
|---|---|---|---|
| Santiago, Chile | 21,661 | Random Forest / KNN | 0.72 |
| Rio de Janeiro, Brazil | 24,519 | Residual MLP | 0.90 |
| Freetown, Sierra Leone (pseudo-labeled, cross-validated) | 11,179 labeled + 14,105 predicted | Tree ensemble (LightGBM/XGBoost/ExtraTrees/RF) | 0.86 |

Across all three cities, the most predictive features were NDMI, median NDBI, and area-weighted building height, confirming that built-up intensity and building density are the dominant drivers of urban heat, ahead of vegetation indices such as NDVI.

## Limitations

Freetown has no independently verified ground truth for the majority of its points; 3,071 of the 11,179 labels used to train the Freetown-adapted model were directly confirmed, and the remainder were themselves inferred through pseudo-labeling and spatial nearest-neighbour methods, so the reported Freetown F1 score should be read as an internal cross-validation estimate rather than a true held-out test score. Santiago and Rio de Janeiro also differ substantially in class balance (Santiago is majority Medium at 49.93%, Rio is majority High at 44.92%), which is a source of domain shift that within-region feature scaling only partially corrects. Each city's ground data was collected on a single afternoon, so the models reflect UHI patterns at that specific time of day and season rather than year-round urban heat behaviour.

## How to Run

```
pip install -r requirements.txt
jupyter notebook notebooks/01_feature_extraction.ipynb
jupyter notebook notebooks/02_modeling_and_prediction.ipynb
```
