# Flight Arrival Delay Prediction — Final Results

**Project:** AviationPrediction-1
**Data:** BTS On-Time Performance, Jan–Jun 2025
**Date:** March 2026

---

## Problem Statement

US domestic flights are delayed by 15 or more minutes in roughly one in five departures, causing cascading disruptions for passengers, airlines, and airports. This project builds a binary classifier to predict, at the time of scheduling, whether a given flight will arrive at least 15 minutes late (`ArrDel15 = 1`). The model uses only pre-departure schedule information — airline, route, departure time, and calendar features — making it actionable at the point of booking rather than requiring real-time data. The core challenge is an 80/20 class imbalance and a meaningful distribution shift between the winter training period (19.7% delay rate) and the summer test period (28.3%), which together constrain achievable precision.

---

## Approach

Seven months of BTS On-Time Reporting data (3,997,256 cleaned flights, Jan–Jul 2025) were processed through a four-notebook pipeline. Data was split chronologically before any feature engineering: January–March 2025 as training, April as validation, June as test, with May excluded as a temporal buffer. Features were engineered in three groups: schedule numerics with cyclical encodings for time-of-day, month, and day-of-week; route and airport signals via smoothed target encoding (k=100, training data only) for airline, origin, destination, route, and departure hour; and binary flags for hub airports, morning flights, and evening flights. Five model families were evaluated — majority-class baseline, Logistic Regression, Random Forest, LightGBM (with hyperparameter tuning across six configurations), and MLP — using ROC-AUC on the validation set as the selection criterion. The test set was touched exactly once for final reporting.

---

## Key Findings from EDA

- **Departure hour is the single strongest pre-departure signal.** Flights departing between 5–8 AM have delay rates below 12%; flights departing after 6 PM exceed 26%. This pattern is consistent across carriers and months, and motivated the priority given to hour-derived features (cyclical encodings + IsMorningFlight/IsEveningFlight + HourDelayRate).

- **Delay rates vary substantially by route corridor, not just airport.** Route-level delay rates span 7.5%–41.8% in the training set — more than four times the spread of origin or destination rates alone. Smoothed target encoding captures this signal while regularising sparse routes (5,908 unique training routes; 650 unseen in the June test set).

- **June 2025 represents a genuine distribution shift.** The test month has a 28.3% delay rate versus 19.7% in January–March — an 8.6 percentage-point shift attributable to summer convective weather. This shift was identified before model evaluation and was treated as a known limitation rather than a correctable artefact; it means the validation-tuned threshold (0.445) is not optimal for the test period.

---

## Model Comparison Results

| Model | Val ROC-AUC | Test ROC-AUC | Test PR-AUC | Test Precision | Test Recall | Test F1 |
|-------|-------------|--------------|-------------|----------------|-------------|---------|
| Majority-class baseline | 0.500 | 0.500 | 0.283 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.623 | 0.654 | 0.398 | 0.470 | 0.095 | 0.158 |
| **Random Forest** | **0.626** | **0.670** | **0.426** | **0.375** | **0.698** | **0.488** |
| LightGBM | 0.587 | 0.628 | 0.368 | 0.351 | 0.688 | 0.465 |
| MLP | 0.565 | 0.536 | 0.300 | 0.282 | 0.124 | 0.172 |
| LightGBM-Tuned | 0.602 | 0.652 | 0.393 | 0.377 | 0.686 | 0.487 |

RandomForest was selected as the best model based on the highest validation ROC-AUC (0.6256). Logistic Regression was a close second (0.6230) and shows stronger precision in the test results, albeit with very low recall — its threshold tuning did not transfer well across the distribution shift. LightGBM underperformed expectations given the dataset size (1.6M training rows): despite six hyperparameter configurations, the best tuned result reached only 0.602 val AUC versus RF's 0.626. This is likely because the smoothed target-encoded route and airport features create a high-signal numeric representation that favours RF's bagging robustness over gradient boosting's sequential correction. The MLP converged poorly relative to the tabular-optimised tree methods, achieving the lowest test ROC-AUC (0.536) despite 103 training epochs.

---

## Best Model Details

**Model:** Random Forest Classifier (200 trees, max_depth=15, min_samples_leaf=20, class_weight='balanced')
**Selection criterion:** Validation ROC-AUC on April 2025 (held-out, not used in training)
**Operating threshold:** 0.445 (Youden's J = argmax(TPR − FPR) on validation set)

At this threshold on the June 2025 test set:

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.670 |
| PR-AUC | 0.426 |
| Precision | 0.375 |
| Recall | 0.698 |
| F1 | 0.488 |
| Brier Score | 0.217 |

The model catches **69.8% of actual delays** (118,149 of 169,375) while generating false alarms on 62.5% of its positive predictions. For a passenger-alert use case this precision/recall trade-off is broadly acceptable — missing a delay is more costly than a false alarm — but the false alarm rate would need reduction before operational deployment. The dominant features (from corrected RF importances in `reports/28_rf_feature_importance_corrected.png`) are the target-encoded delay rates for destination, origin, and route, followed by cyclical hour features and distance, confirming that historical route-corridor patterns and time-of-day are the primary discriminating signals available at schedule time.

---

## Error Analysis Key Findings

- **False negatives concentrate in specific airline/hour combinations.** Subgroup analysis (notebook 05) shows that FN rates vary across carriers and are elevated at departure hours 0–4 (overnight) and 13–16 (early afternoon), where delay rates are either very low (overnight, few events) or moderate with high variance. The overall FN rate on the test set is 30.2% (51,226 missed delays).

- **Threshold transfer is the largest actionable gap.** The val-selected threshold (0.445, optimal for April's 19.7% prevalence) is measurably suboptimal for June's 28.3% prevalence. The threshold sweep shows that re-tuning the threshold on contemporaneous validation data would improve test-set F1 without any retraining. This is the single highest-leverage operational improvement identified.

- **A confirmed diagnostic bug was found during the pipeline audit.** In `notebooks/03_data_preparation.ipynb`, the unseen-route count reported `-569,831` for validation and `-579,453` for test — impossible negative values caused by using row-level `.isin().sum()` instead of a unique-value set-difference. The correct unseen counts are 137 (validation) and 650 (test). This bug had no impact on model training or predictions because the encoding fallback (`fillna(global_mean)`) operates independently, but the negative numbers appeared in printed output and went unnoticed during development review.

---

## Limitations and Next Steps

The model's most fundamental constraint is the absence of weather data: convective activity, fog, and winter storms are the primary causes of the delays the model most frequently misses, yet schedule-time features can only capture their *historical correlation* with routes and times — not their occurrence on any given day. A second constraint is temporal scope: with three months of training data from a single winter/spring, the model has not been exposed to major disruption events, multi-year baseline variation, or carrier network restructuring. In the near term, the highest-leverage improvements would be: (1) adding the NWS Aviation Weather Center convective outlook as a binary feature, which is available pre-departure and costs no additional data infrastructure; (2) incorporating inbound aircraft delay as a feature (a highly predictive propagated-delay signal available from public BTS data); and (3) implementing threshold monitoring in any production deployment so the operating point can be recalibrated as seasonal prevalence shifts. Longer-term, retraining on two to three years of data and adding a calibration layer (Platt scaling or isotonic regression) would address generalisation and probability reliability respectively.
