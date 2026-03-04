# Model Card — Flight Arrival Delay Classifier

**Version:** 1.0
**Date trained:** March 2025 (data); March 2026 (model fit)
**Maintained by:** AviationPrediction-1 project

---

## Model Details


| Field               | Value                                                                             |
| ------------------- | --------------------------------------------------------------------------------- |
| Model type          | Random Forest Classifier (sklearn 1.6.1)                                          |
| Algorithm           | Ensemble of 200 decision trees, max_depth=15, min_samples_leaf=20                 |
| Class weighting     | `class_weight='balanced'` — upweights minority (delayed) class by n/(2·n_delayed) |
| Operating threshold | 0.445 (Youden's J maximised on April 2025 validation set)                         |
| Input features      | 59 pre-processed features (details below)                                         |
| Output              | Probability of arrival delay ≥15 minutes; binary label at threshold               |
| Framework           | scikit-learn RandomForestClassifier, preprocessor via ColumnTransformer           |
| Saved artifact      | `data/processed/best_model.joblib`                                                |


### Why RandomForest?

RandomForest was selected over Logistic Regression, LightGBM, and MLP based on the highest validation ROC-AUC (0.6256) among five candidates trained on the January–March 2025 period. LightGBM achieved 0.587–0.602 despite hyperparameter tuning across six configurations, likely because bagging is more robust to the target-encoded route features that dominate the feature space. A full comparison is available in `reports/model_comparison.csv`.

---

## Intended Use

### Primary use case

Pre-departure prediction of whether a scheduled US domestic flight will arrive 15 or more minutes late, using only information available at the time of booking or check-in (scheduled times, airline, route, calendar date).

### Appropriate applications

- Passenger-facing alerts: informing travellers of elevated delay risk before departure
- Airline operational planning: flagging high-risk departures for proactive crew or gate management
- Research and benchmarking: baseline for comparing more sophisticated delay prediction approaches

### Out-of-scope uses

- **Real-time inference during flight:** the model has no access to live weather, ATC status, or departure actuals
- **International or cargo flights:** trained exclusively on US domestic passenger operations (BTS On-Time Reporting carriers)
- **Specific delay duration prediction:** this is a binary classifier (≥15 min / not); it does not predict delay length
- **Individual flight guarantee:** predicted probabilities are population-level estimates, not certainties for any single flight
- **High-stakes automated decision-making without human review:** model errors (false negatives and false positives) are material

---

## Training Data


| Split                 | Months       | Flights   | Delay Rate |
| --------------------- | ------------ | --------- | ---------- |
| Training              | Jan–Mar 2025 | 1,610,648 | 19.7%      |
| Validation            | Apr 2025     | 577,496   | 19.7%      |
| **Buffer (excluded)** | **May 2025** | **—**     | **—**      |
| Test (held-out)       | Jun 2025     | 599,209   | 28.3%      |


**Source:** BTS On-Time Performance dataset (Form 41 Traffic — T-100 Domestic), accessed via the BTS TranStats portal. Seven months present in raw data (Jan–Jul 2025); July is an unused additional holdout.

**Cleaning:** Cancelled flights (67K removed), diverted flights (12K removed), flights with missing ArrDel15 (4K removed), duplicates on (FlightDate, Airline, Origin, Dest, CRSDepTime) (1.4K removed). Final cleaned dataset: 3,997,256 rows.

**Target variable:** `ArrDel15` — 1 if the flight arrived ≥15 minutes late, 0 otherwise. Derived directly from the BTS `ARR_DEL15` field.

### Feature Engineering Summary (59 features)

All features use only pre-departure schedule information. No actuals (departure delay, cause codes, etc.) are included.


| Group           | Count | Features                                                                                                                                                                                             |
| --------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scaled numerics | 17    | CRSDepTime, CRSArrTime, CRSElapsedTime, Distance, DistanceGroup, CRSDepMinute, cyclical sin/cos for hour/month/day-of-week, smoothed target-encoded delay rates (airline, origin, dest, route, hour) |
| One-hot encoded | 33    | Reporting_Airline (14 categories), DepTimeBlk (19 time blocks)                                                                                                                                       |
| Passthrough     | 9     | IsWeekend, IsEveningFlight, IsMorningFlight, IsHubOrigin, IsHubDest, Quarter, Month, DayofMonth, DayOfWeek                                                                                           |


**Leakage prevention:** The chronological split was applied before any aggregate statistic was computed. Smoothed target-encoding maps (k=100) were fitted on training data only, with the global training mean (19.7%) as fallback for unseen categories. The preprocessing ColumnTransformer was fitted on training data only.

---

## Evaluation Data

The model was evaluated on **June 2025** US domestic flights (599,209 flights), a full calendar month held out from all training and validation activity. June 2025 exhibits a materially higher delay rate (28.3%) than the training period (19.7%), attributable to summer convective weather. This constitutes a genuine distribution shift and is the primary reason test-set metrics differ in character from validation metrics.

---

## Performance Metrics

All metrics computed on the June 2025 test set at the operating threshold of 0.445.


| Metric      | Value     | Notes                                                              |
| ----------- | --------- | ------------------------------------------------------------------ |
| ROC-AUC     | **0.670** | Primary discrimination metric; threshold-independent               |
| PR-AUC      | **0.426** | Area under precision-recall curve; relevant for imbalanced data    |
| Precision   | 0.375     | 37.5% of predicted delays are actual delays                        |
| Recall      | 0.698     | 69.8% of actual delays are flagged                                 |
| F1 Score    | 0.488     | Harmonic mean of precision and recall                              |
| Accuracy    | 0.586     | Misleading at 28.3% prevalence; included for completeness          |
| Brier Score | 0.217     | Probability calibration quality (lower = better; baseline = 0.283) |


**Baseline comparison:** A majority-class classifier (always predict on-time) achieves 71.7% accuracy, 0.000 F1, and 0.500 ROC-AUC on the test set. All reported metrics are strictly above this floor.

**Threshold sensitivity:** The operating threshold (0.445) was tuned on April 2025 (19.7% delay rate). Due to the June distribution shift, the threshold that maximises F1 on the test set is lower. For deployments in high-prevalence seasons, the threshold should be re-calibrated on contemporaneous validation data.

### Subgroup Performance (ROC-AUC by airline, test set)

Model discrimination varies across airlines. Subgroup ROC-AUC is available in `reports/27_subgroup_auc.png`. No carrier falls below 0.55 ROC-AUC; performance differences likely reflect structural variation in how each carrier's delays relate to the available schedule-time features.

---

## Ethical Considerations

### Fairness concerns

- **Airline-level disparity:** Subgroup analysis shows variation in ROC-AUC across carriers. Airlines with structurally lower prediction quality may face disproportionate false alarms or missed warnings if the model is used operationally. The source of these disparities (route network, hub complexity, aircraft type) is not investigated in this version.
- **Route-level sparse coverage:** Routes with fewer than 100 training-set flights receive target-encoded delay rates shrunk toward the global mean (by design). Predictions for low-frequency routes are less personalised and more uncertain.
- **No protected-class analysis:** BTS data does not include passenger demographics. Disparate impact on passengers by protected characteristics (disability status, etc.) cannot be assessed from this model alone.

### Data provenance

All data is sourced from the US Bureau of Transportation Statistics, a public federal dataset. No personally identifiable information is present.

---

## Limitations

1. **US domestic operations only.** The model was trained on BTS-reporting US domestic carriers. It is not validated for international, regional, or cargo operations with different delay dynamics.
2. **Schedule-time features only.** The model has no access to weather conditions, NOTAM data, ATC ground stops, or real-time aircraft positioning. These are the primary causes of large delays; the model can only capture their *historical correlation* with schedule characteristics.
3. **Single year of data.** Training on January–March 2025 means the model has not seen winter storm seasons, major disruption events (e.g., holiday surges, air traffic system failures), or multi-year baseline variation. Generalisability to future years is uncertain.
4. **Seasonal distribution shift.** The test set (June 2025, 28.3% delay rate) demonstrates that model performance and optimal operating threshold shift materially with season. Deployment in peak-delay periods (summer, holiday) requires threshold recalibration and ongoing monitoring.
5. **Confirmed diagnostic bug (no model impact).** An error in the route-unseen-count diagnostic in notebook 03 printed impossible negative values (-569,831 for validation). This did not affect model training or predictions (the encoding fallback operates independently of the diagnostic), but went unnoticed during development. See `reports/error_analysis_summary.md` Issue A.
6. **Feature importance from suboptimal model in notebook 04.** The feature importance visualisation in notebook 04 was hardcoded to LightGBM-Tuned regardless of model selection outcome. The corrected RandomForest importance chart is available at `reports/28_rf_feature_importance_corrected.png`.

---

## Recommendations for Improvement


| Priority | Improvement                                                            | Expected Impact                                                    |
| -------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------ |
| High     | Add weather features (NWS gridded forecasts, convective outlook)       | Delay cause codes show weather is the largest missed signal        |
| High     | Re-train annually or seasonally with rolling window                    | Reduces distribution shift between training and deployment periods |
| Medium   | Live aircraft positioning / inbound delay feature                      | Propagated delay is highly predictive; available pre-departure     |
| Medium   | Calibration layer (Platt scaling or isotonic regression on validation) | Improves probability reliability for passenger-facing risk scores  |
| Medium   | Threshold monitoring in production                                     | Optimal decision threshold shifts with seasonal delay prevalence   |
| Low      | Expand training to 2–3 years of BTS data                               | Captures year-to-year variation and rare disruption events         |
| Low      | SHAP explanations per flight                                           | Required for any operational deployment requiring explainability   |


