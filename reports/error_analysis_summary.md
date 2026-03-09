# Error Analysis Summary — Flight Delay Prediction

Generated from: `notebooks/05_error_analysis.ipynb`  
Best model: RandomForest | Test ROC-AUC: 0.6697 | Threshold: 0.4451

---

## 1. Overall Error Counts (Test Set — June 2025)

| Metric | Value |
|--------|-------|
| Total test flights | 599,209 |
| Actual delays | 169,375 (28.3%) |
| True Positives (delays caught) | 118,149 (69.8% of delays) |
| False Negatives (delays missed) | 51,226 (30.2% of delays) |
| False Positives (false alarms) | 196,724 (45.8% of on-times) |
| True Negatives | 233,110 |

## 2. False Negative Profile

**FN rate by airline:** Airlines with highest miss rates should be investigated for structural signals the model lacks (e.g., intra-hub congestion, regional vs mainline ops). Plotted in `reports/20_fn_by_airline.png`.

**FN rate by departure hour:** Early morning and late-night hours tend to have higher FN rates — these are periods with fewer historical samples and lower baseline delay rates, making the model more conservative.

**FN rate by distance:** Short-haul routes may have higher FN rates due to compressed delay structure (small delays are harder to predict from schedule alone).

## 3. False Positive Profile

Overall FP rate: **45.8%** of on-time flights falsely flagged. For a deployment alert system this represents the false alarm rate. See `reports/22_fp_analysis.png`.

## 4. Confidence Distribution

- Mean predicted probability for delayed flights: **0.505**
- Mean predicted probability for on-time flights: **0.429**
- Separation: **0.076** (larger = better discriminating probabilities)
- Flights in uncertainty zone (±0.1 of threshold): **52.5%** of test set

## 5. Threshold Analysis

| Threshold | Context | F1 |
|-----------|---------|----|
| 0.445 | Youden J on April validation (19.7% delays) | 0.489 |
| 0.410 | Optimal F1 on June test (28.3% delays) | 0.493 |
| 0.050 | Threshold for ≥80% Recall on test | — |

**Distribution shift impact:** June 2025 has 28.3% delay rate vs April's 19.7%. The val-selected threshold (0.445) is not optimal for June — the best test F1 threshold is 0.410 (-0.035 shift). For production, threshold should be re-tuned on data matching the deployment period's prevalence.

## 6. Subgroup Performance

- Best airline ROC-AUC  : **0.710** (WN)
- Worst airline ROC-AUC : **0.613** (AS)
- Hours with weakest discrimination: model degrades at night/early morning (low sample hours)

## 7. Pipeline Audit Findings

### ISSUE A — CONFIRMED BUG (Agent-Made Mistake)
**File:** `notebooks/03_data_preparation.ipynb` — diagnostic cell in Section 3  
**Code:** `unseen_val = df_val['Route'].nunique() - df_val['Route'].isin(df_train['Route']).sum()`  
**Problem:** `.isin(...).sum()` counts rows (≈570,000), not unique routes. Result: **-569,831** (impossible).  
**Impact:** None on model (encoding fallback `fillna(global_mean)` works correctly independent of this diagnostic).  
**Fix:** `len(set(df_val['Route'].unique()) - set(df_train['Route'].unique()))`  
**Actual unseen routes in val:** 137 | **in test:** 650

### ISSUE B — Methodological: LightGBM val contamination
**File:** `notebooks/04_modelling.ipynb` — Sections 5 and 7  
**Problem:** LightGBM used `eval_set=[(X_val, y_val)]` for early stopping, then val metrics compared to RF/LR who were trained without seeing val labels. LightGBM had an information advantage.  
**Impact:** Minimal — LightGBM still scored 0.587 vs RF 0.626; model selection unchanged.  
**Fix:** Use a 10% holdout split from training data for early stopping; keep val truly held-out.

### ISSUE C — Output: Wrong model shown in feature importance
**File:** `notebooks/04_modelling.ipynb` — Section 8.4  
**Problem:** Code hardcoded `tree_model_name = 'LightGBM-Tuned'` for importance plot; best model was RandomForest.  
**Impact:** Feature importance chart in reports/ shows LightGBM-Tuned, not the deployed model.  
**Fix:** Use `models[BEST_MODEL_NAME].feature_importances_` — corrected in notebook 05, saved as `reports/28_rf_feature_importance_corrected.png`.

### Checks PASSED
| Check | Result |
|-------|--------|
| Target encoding global mean from train only | PASS |
| StandardScaler fitted on training data only | PASS |
| Hub airport list derived from training only | PASS |
| Temporal split — no date overlap, May buffer intact | PASS |
| OHE `handle_unknown='ignore'` for unseen categories | PASS |
| Leakage-safe feature selection (no post-departure actuals) | PASS |
