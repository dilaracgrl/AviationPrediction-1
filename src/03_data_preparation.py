#!/usr/bin/env python
# coding: utf-8

# # 03 — Feature Engineering & Data Preparation
# 
# **Project:** Flight Arrival Delay Prediction  
# **Input:** `data/processed/model_dataset.parquet`  
# **Outputs:** `data/processed/{train,val,test}_{X,y}.parquet`, `data/processed/preprocessor.joblib`  
# 
# ---
# 
# ## Leakage Safety Principle
# 
# Every feature engineered here uses **only information that would be known before the flight departs**. The temporal split is applied **first**, before any aggregate is computed, so that training-set statistics never contaminate validation or test sets.
# 
# | Section | Content |
# |---------|--------|
# | 1 | Chronological split |
# | 2 | Time features (sin/cos cyclical, binary flags) |
# | 3 | Route features (route string, hub flags) |
# | 4 | Historical delay rates — smoothed target encoding |
# | 5 | Feature selection and column grouping |
# | 6 | Preprocessing pipeline — fit on train, transform all |
# | 7 | Validation checks |
# | 8 | Save artifacts |

# ## 0. Setup

# In[35]:


import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60)
pd.set_option('display.float_format', '{:.4f}'.format)

# ── Path resolution (CWD-agnostic) ───────────────────────────────────────────
_cwd = Path(os.path.abspath('')).resolve()
PROJECT_ROOT = _cwd.parent if _cwd.name == 'notebooks' else _cwd

INPUT_PATH    = PROJECT_ROOT / 'data' / 'processed' / 'model_dataset.parquet'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print(f'Project root : {PROJECT_ROOT}')
print(f'Input        : {INPUT_PATH}  (exists={INPUT_PATH.exists()})')
print(f'Output dir   : {PROCESSED_DIR}')
print()
import sklearn; print(f'scikit-learn : {sklearn.__version__}')
print(f'pandas       : {pd.__version__}')


# ## 1. Load Data

# In[36]:


df = pd.read_parquet(INPUT_PATH)

# Normalise nullable integer columns to standard int64 for sklearn compatibility
for col in ['CRSDepHour', 'CRSDepMinute']:
    if col in df.columns:
        df[col] = df[col].astype('int64')

print(f'Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns')
print(f'Date range: {df["FlightDate"].min().date()} → {df["FlightDate"].max().date()}')
print(f'Months present: {sorted(df["Month"].unique())}')
print(f'ArrDel15 dtype: {df["ArrDel15"].dtype}')


# ---
# ## Section 1 — Chronological Split
# 
# **Why split before feature engineering?** Any aggregate statistic (mean delay rate per airline, per airport, etc.) computed on the full dataset would incorporate future information for the validation and test flights. Splitting first, then computing statistics on the training set only, is the only correct approach.
# 
# | Split | Months | Rationale |
# |-------|--------|-----------|
# | Train | 1, 2, 3 | Q1 winter / spring shoulder |
# | Validation | 4 | April — immediate post-training month |
# | **Month 5** | **excluded** | **Buffer — prevents adjacent-month leakage** |
# | Test | 6 | June — held out for final evaluation |
# | Month 7 | excluded | Extra holdout, not used in this study |

# In[20]:


TRAIN_MONTHS = [1, 2, 3]
VAL_MONTHS   = [4]
TEST_MONTHS  = [6]

train_mask = df['Month'].isin(TRAIN_MONTHS)
val_mask   = df['Month'].isin(VAL_MONTHS)
test_mask  = df['Month'].isin(TEST_MONTHS)

df_train = df[train_mask].copy().reset_index(drop=True)
df_val   = df[val_mask].copy().reset_index(drop=True)
df_test  = df[test_mask].copy().reset_index(drop=True)

# ── Verify no date overlap ────────────────────────────────────────────────────
train_dates = set(df_train['FlightDate'])
val_dates   = set(df_val['FlightDate'])
test_dates  = set(df_test['FlightDate'])
assert len(train_dates & val_dates)  == 0, 'OVERLAP: train ∩ val'
assert len(train_dates & test_dates) == 0, 'OVERLAP: train ∩ test'
assert len(val_dates   & test_dates) == 0, 'OVERLAP: val ∩ test'
print('Date overlap check: PASS — no overlap between splits.')

# ── Summary ───────────────────────────────────────────────────────────────────
GLOBAL_TRAIN_MEAN = df_train['ArrDel15'].mean()
print(f'\nGlobal training delay rate: {GLOBAL_TRAIN_MEAN:.4f}  ({GLOBAL_TRAIN_MEAN*100:.1f}%)')
print()
print(f'{"Split":<12} {"Rows":>10} {"Delay Rate":>12} {"Date Range"}')
print('-' * 60)
for name, split_df in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
    dr = split_df['ArrDel15'].mean() * 100
    d0 = split_df['FlightDate'].min().date()
    d1 = split_df['FlightDate'].max().date()
    print(f'{name:<12} {len(split_df):>10,} {dr:>11.1f}%  {d0} → {d1}')


# ---
# ## Section 2 — Time Feature Engineering
# 
# These features are derived purely from scheduled calendar and clock values — no aggregate statistics, no training-set dependency. They are therefore identical for train, val, and test and can be computed on the full dataset in one pass.
# 
# ### 2.1 Binary Flags
# 
# | Feature | Definition | Rationale |
# |---------|------------|-----------|
# | `IsWeekend` | DayOfWeek ∈ {6, 7} | Lower traffic, different delay dynamics |
# | `IsEveningFlight` | CRSDepHour ≥ 17 | Peak delay accumulation period |
# | `IsMorningFlight` | CRSDepHour ≤ 8 | Lowest delay rates, clean aircraft |
# 
# ### 2.2 Cyclical Encodings
# 
# Linear models and distance-based models incorrectly treat hour 23 and hour 0 as maximally distant. Cyclical (sin/cos) encoding wraps the feature into a circle so that the topology is preserved.
# 
# | Feature | Period | Produces |
# |---------|--------|----------|
# | CRSDepHour | 24 | `DepHour_sin`, `DepHour_cos` |
# | Month | 12 | `Month_sin`, `Month_cos` |
# | DayOfWeek | 7 | `DayOfWeek_sin`, `DayOfWeek_cos` |

# In[21]:


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Pure time-derived features — no training-data dependency."""
    out = frame.copy()

    # Binary flags
    out['IsWeekend']       = (out['DayOfWeek'] >= 6).astype(np.int8)
    out['IsEveningFlight'] = (out['CRSDepHour'] >= 17).astype(np.int8)
    out['IsMorningFlight'] = (out['CRSDepHour'] <= 8).astype(np.int8)

    # Cyclical: hour (period=24)
    out['DepHour_sin'] = np.sin(2 * np.pi * out['CRSDepHour'] / 24)
    out['DepHour_cos'] = np.cos(2 * np.pi * out['CRSDepHour'] / 24)

    # Cyclical: month (period=12)
    out['Month_sin'] = np.sin(2 * np.pi * out['Month'] / 12)
    out['Month_cos'] = np.cos(2 * np.pi * out['Month'] / 12)

    # Cyclical: day of week (period=7)
    out['DayOfWeek_sin'] = np.sin(2 * np.pi * out['DayOfWeek'] / 7)
    out['DayOfWeek_cos'] = np.cos(2 * np.pi * out['DayOfWeek'] / 7)

    return out

df_train = add_time_features(df_train)
df_val   = add_time_features(df_val)
df_test  = add_time_features(df_test)

new_time_cols = ['IsWeekend', 'IsEveningFlight', 'IsMorningFlight',
                 'DepHour_sin', 'DepHour_cos', 'Month_sin', 'Month_cos',
                 'DayOfWeek_sin', 'DayOfWeek_cos']
print('Time features added:')
print(df_train[new_time_cols].describe().T[['mean', 'min', 'max']].to_string())

print(f'\nIsWeekend       : {df_train["IsWeekend"].mean()*100:.1f}% weekend flights in train')
print(f'IsEveningFlight : {df_train["IsEveningFlight"].mean()*100:.1f}% evening flights in train')
print(f'IsMorningFlight : {df_train["IsMorningFlight"].mean()*100:.1f}% morning flights in train')


# ---
# ## Section 3 — Route Features
# 
# ### 3.1 Route String
# 
# `Route = Origin + '_' + Dest` creates a composite key (e.g., `LAX_JFK`) capturing the specific city-pair, which is more granular than origin or destination individually. High cardinality (~4,000+ unique routes) means it will be target-encoded rather than one-hot encoded.
# 
# ### 3.2 Hub Airport Flags
# 
# Major hub airports have structurally different delay dynamics (high traffic density, complex ground operations). We define "hub" as the top 15 airports by **training-set flight volume** — this is computed from training data only and applied to val/test as a fixed lookup.

# In[22]:


# ── Route string (leakage-free: pure string concatenation) ───────────────────
for split in [df_train, df_val, df_test]:
    split['Route'] = split['Origin'] + '_' + split['Dest']

print(f'Unique routes in train : {df_train["Route"].nunique():,}')
print(f'Unique routes in val   : {df_val["Route"].nunique():,}')
print(f'Unique routes in test  : {df_test["Route"].nunique():,}')

# Routes in val/test not seen in training (will fall back to global mean in target encoding)
# Correct: count unique route VALUES not in train (not row count — that produced impossible negatives)
train_route_set = set(df_train['Route'].unique())
unseen_val  = len(set(df_val['Route'].unique()) - train_route_set)
unseen_test = len(set(df_test['Route'].unique()) - train_route_set)
print(f'Unseen routes in val   : {unseen_val:,}  (→ will use global train mean)')
print(f'Unseen routes in test  : {unseen_test:,}  (→ will use global train mean)')


# In[23]:


# ── Hub airport list — derived from TRAINING DATA ONLY ───────────────────────
N_HUBS = 15
hub_airports = (
    df_train['Origin'].value_counts()
    .head(N_HUBS)
    .index
    .tolist()
)
print(f'Top {N_HUBS} hub airports (by origin volume in training set):')
print(hub_airports)

HUB_SET = set(hub_airports)

for split in [df_train, df_val, df_test]:
    split['IsHubOrigin'] = split['Origin'].isin(HUB_SET).astype(np.int8)
    split['IsHubDest']   = split['Dest'].isin(HUB_SET).astype(np.int8)

print(f'\nIsHubOrigin (train): {df_train["IsHubOrigin"].mean()*100:.1f}% of flights')
print(f'IsHubDest   (train): {df_train["IsHubDest"].mean()*100:.1f}% of flights')


# ---
# ## Section 4 — Historical Delay Rates (Smoothed Target Encoding)
# 
# Target encoding replaces a categorical value with the historical mean of the target within that category. This compresses high-cardinality features (Origin: ~350 airports, Route: ~4,000+ pairs) into a single informative numeric column.
# 
# **Why smoothed (regularised) encoding?**  
# Rare categories have noisy mean estimates. A route flown only 5 times in training might have 0% or 100% delay — neither is reliable. Smoothing blends the category mean with the global mean, weighted by sample count:
# 
# $$\hat{\mu}_c = \frac{n_c \cdot \bar{y}_c + k \cdot \bar{y}_{global}}{n_c + k}$$
# 
# where $k = 100$ (minimum effective sample threshold). Categories with $n_c \gg k$ are trusted; rare categories are shrunk toward the global mean.
# 
# **Leakage guarantee:** Encoding maps are computed **only on df_train**. Val and test rows receive their train-derived estimate. Unseen categories (routes/airlines not in training) receive the global training mean.

# In[24]:


SMOOTHING_K = 100  # equivalent to requiring 100 samples before trusting category mean

def smoothed_target_encode(
    train_series: pd.Series,
    train_target: pd.Series,
    global_mean: float,
    k: int = SMOOTHING_K,
) -> dict:
    """
    Compute smoothed target-encoding map from training data.
    Returns a dict {category: smoothed_rate} for application to any split.
    """
    stats = (
        pd.DataFrame({'cat': train_series, 'target': train_target})
        .groupby('cat')['target']
        .agg(count='count', mean='mean')
    )
    stats['smoothed'] = (
        (stats['count'] * stats['mean'] + k * global_mean) /
        (stats['count'] + k)
    )
    return stats['smoothed'].to_dict()


def apply_encoding(series: pd.Series, mapping: dict, fallback: float) -> pd.Series:
    """Apply a target-encoding map; unseen categories receive fallback (global mean)."""
    return series.map(mapping).fillna(fallback)


print(f'Global training delay rate (k denominator): {GLOBAL_TRAIN_MEAN:.4f}')
print(f'Smoothing k = {SMOOTHING_K} (categories with < {SMOOTHING_K} samples are shrunk toward global mean)')


# In[25]:


# ── Compute all encoding maps from training data ONLY ────────────────────────
ENCODE_SPECS = [
    ('Reporting_Airline', 'AirlineDelayRate'),
    ('Origin',            'OriginDelayRate'),
    ('Dest',              'DestDelayRate'),
    ('Route',             'RouteDelayRate'),
    ('CRSDepHour',        'HourDelayRate'),
]

encoding_maps = {}
print(f'{"Feature":<22} {"Unique cats":>12} {"Rate range"}')
print('-' * 55)
for src_col, rate_col in ENCODE_SPECS:
    enc_map = smoothed_target_encode(
        df_train[src_col].astype(str),
        df_train['ArrDel15'],
        GLOBAL_TRAIN_MEAN,
    )
    encoding_maps[rate_col] = (src_col, enc_map)
    rates = list(enc_map.values())
    print(f'{src_col:<22} {len(enc_map):>12,} {min(rates):.3f} – {max(rates):.3f}')

print(f'\nAll encoding maps computed from training data only.')


# In[26]:


# ── Apply encoding maps to all splits ────────────────────────────────────────
for rate_col, (src_col, enc_map) in encoding_maps.items():
    for split in [df_train, df_val, df_test]:
        split[rate_col] = apply_encoding(
            split[src_col].astype(str), enc_map, GLOBAL_TRAIN_MEAN
        )

rate_cols = [v[0] for v in ENCODE_SPECS]  # just for display
rate_new  = [r for _, r in ENCODE_SPECS]

print('Encoded delay rate columns — train statistics:')
print(df_train[rate_new].describe().T[['mean', 'std', 'min', 'max']].to_string())

# Leakage check: val/test rates must be computed FROM TRAIN — verify no NaN
for name, split in [('val', df_val), ('test', df_test)]:
    nan_counts = split[rate_new].isna().sum()
    total_nan  = nan_counts.sum()
    print(f'\nNaN in {name} encoded rates: {total_nan}  (should be 0 — fallback applied)')


# ---
# ## Section 5 — Feature Selection and Column Grouping
# 
# We now define exactly which columns feed into each preprocessing transformer:
# 
# | Group | Transformer | Columns |
# |-------|-------------|--------|
# | `scale_cols` | StandardScaler | Continuous numerics + encoded rates + cyclical features |
# | `ohe_cols` | OneHotEncoder | Low-cardinality categoricals |
# | `passthrough_cols` | Passthrough | Binary flags + ordinal integers |
# 
# **Dropped columns (not fed to the model):**
# 
# | Column | Reason |
# |--------|--------|
# | `FlightDate` | Used only for splitting; datetime, not a model input |
# | `Tail_Number` | Very high cardinality (~5,000+), excluded per spec |
# | `Origin`, `Dest` | Replaced by `OriginDelayRate`, `DestDelayRate`, hub flags |
# | `OriginAirportID`, `DestAirportID` | Surrogate integer keys — no ordinal meaning |
# | `OriginState`, `DestState` | Hub flags + route encoding capture this signal |
# | `Route` | Replaced by `RouteDelayRate` |
# | `CRSDepHour` | Replaced by cyclical encoding + binary flags |
# | `Year` | Single value (2025) — zero variance |

# In[27]:


# ── Column groups ─────────────────────────────────────────────────────────────
SCALE_COLS = [
    # Raw schedule numerics
    'CRSDepTime', 'CRSArrTime', 'CRSElapsedTime', 'Distance', 'DistanceGroup', 'CRSDepMinute',
    # Cyclical time encodings
    'DepHour_sin', 'DepHour_cos', 'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
    # Smoothed target-encoded delay rates
    'AirlineDelayRate', 'OriginDelayRate', 'DestDelayRate', 'RouteDelayRate', 'HourDelayRate',
]

OHE_COLS = [
    'Reporting_Airline',  # ~15 unique carriers
    'DepTimeBlk',         # 19 departure time blocks
]

PASSTHROUGH_COLS = [
    # Binary flags
    'IsWeekend', 'IsEveningFlight', 'IsMorningFlight', 'IsHubOrigin', 'IsHubDest',
    # Ordinal/calendar integers (kept raw per spec)
    'Quarter', 'Month', 'DayofMonth', 'DayOfWeek',
]

FEATURE_COLS = SCALE_COLS + OHE_COLS + PASSTHROUGH_COLS
TARGET_COL   = 'ArrDel15'
DROP_COLS    = [
    'FlightDate', 'Year', 'Tail_Number', 'Origin', 'Dest',
    'OriginAirportID', 'DestAirportID', 'OriginState', 'DestState',
    'Route', 'CRSDepHour',
]

# ── Validate all feature columns exist in all splits ─────────────────────────
for name, split in [('train', df_train), ('val', df_val), ('test', df_test)]:
    missing = [c for c in FEATURE_COLS if c not in split.columns]
    if missing:
        raise ValueError(f'Missing columns in {name}: {missing}')

print(f'Feature column groups:')
print(f'  Scale       : {len(SCALE_COLS)} columns')
print(f'  OneHot      : {len(OHE_COLS)} columns')
print(f'  Passthrough : {len(PASSTHROUGH_COLS)} columns')
print(f'  Total input : {len(FEATURE_COLS)} columns')
print(f'  Dropped     : {DROP_COLS}')
print()
print(f'Scale cols      : {SCALE_COLS}')
print(f'OHE cols        : {OHE_COLS}')
print(f'Passthrough cols: {PASSTHROUGH_COLS}')


# In[28]:


# ── Slice X, y for each split ─────────────────────────────────────────────────
X_train = df_train[FEATURE_COLS].copy()
y_train = df_train[TARGET_COL].copy()

X_val   = df_val[FEATURE_COLS].copy()
y_val   = df_val[TARGET_COL].copy()

X_test  = df_test[FEATURE_COLS].copy()
y_test  = df_test[TARGET_COL].copy()

print(f'X_train shape: {X_train.shape}    y_train positives: {y_train.mean()*100:.1f}%')
print(f'X_val   shape: {X_val.shape}    y_val   positives: {y_val.mean()*100:.1f}%')
print(f'X_test  shape: {X_test.shape}    y_test  positives: {y_test.mean()*100:.1f}%')

print(f'\nNaN counts before preprocessing:')
for name, X in [('train', X_train), ('val', X_val), ('test', X_test)]:
    total_nan = X.isna().sum().sum()
    if total_nan > 0:
        print(f'  {name}: {total_nan} NaN  →', X.isna().sum()[X.isna().sum() > 0].to_dict())
    else:
        print(f'  {name}: 0 NaN ✓')


# ---
# ## Section 6 — Preprocessing Pipeline
# 
# A `sklearn` `ColumnTransformer` assembles the three transformation groups:
# 
# - **StandardScaler** — zero-mean, unit-variance. Necessary for linear models (logistic regression) and regularisation terms. Tree models don't require it but it doesn't hurt.
# - **OneHotEncoder** — `handle_unknown='ignore'` safely passes through categories unseen in training (rare but possible for new airlines). `max_categories=20` caps each column at 20 dummies to prevent explosion from very long tails.
# - **Passthrough** — binary flags and low-ordinal integers are already in [0,1] or small integer ranges; scaling would not harm them but is unnecessary.
# 
# **Critical:** the pipeline is `.fit()` on training data **only** and `.transform()` is applied to val and test.

# In[29]:


preprocessor = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), SCALE_COLS),
        ('ohe',   OneHotEncoder(handle_unknown='ignore', max_categories=20, sparse_output=False), OHE_COLS),
        ('pass',  'passthrough', PASSTHROUGH_COLS),
    ],
    remainder='drop',   # silently drop any columns not listed above
    verbose_feature_names_out=True,
)

print('Fitting preprocessor on TRAINING data only...')
preprocessor.fit(X_train)
print('Fit complete.')

print()
# Show OHE categories learned
ohe = preprocessor.named_transformers_['ohe']
for col, cats in zip(OHE_COLS, ohe.categories_):
    print(f'  OHE {col}: {len(cats)} categories → {list(cats)}')


# In[30]:


# ── Transform all splits ──────────────────────────────────────────────────────
print('Transforming splits...')
X_train_pp = preprocessor.transform(X_train)
X_val_pp   = preprocessor.transform(X_val)
X_test_pp  = preprocessor.transform(X_test)

# Recover feature names for DataFrame construction
feature_names = preprocessor.get_feature_names_out()

X_train_df = pd.DataFrame(X_train_pp, columns=feature_names, dtype=np.float32)
X_val_df   = pd.DataFrame(X_val_pp,   columns=feature_names, dtype=np.float32)
X_test_df  = pd.DataFrame(X_test_pp,  columns=feature_names, dtype=np.float32)

print(f'X_train_pp : {X_train_df.shape}')
print(f'X_val_pp   : {X_val_df.shape}')
print(f'X_test_pp  : {X_test_df.shape}')
print(f'Feature names ({len(feature_names)} total):')
for i, name in enumerate(feature_names):
    print(f'  [{i:>3}]  {name}')


# ---
# ## Section 7 — Validation Checks
# 
# Before saving, we verify four invariants that would silently corrupt downstream modelling if violated.

# In[31]:


print('=== Validation Checks ===')
all_pass = True

# 1. No NaN in any preprocessed split
for name, X in [('X_train', X_train_df), ('X_val', X_val_df), ('X_test', X_test_df)]:
    nan_count = X.isna().sum().sum()
    status = 'PASS' if nan_count == 0 else f'FAIL ({nan_count} NaN)'
    if nan_count > 0:
        all_pass = False
        print(f'  {name} NaN check           : {status}')
        print('    NaN columns:', X.isna().sum()[X.isna().sum() > 0].to_dict())
    else:
        print(f'  {name} NaN check           : {status}')

# 2. No date overlap between splits
overlap_tv = len(train_dates & val_dates)
overlap_tt = len(train_dates & test_dates)
overlap_vt = len(val_dates   & test_dates)
print(f'  Train ∩ Val date overlap    : {"PASS" if overlap_tv == 0 else f"FAIL ({overlap_tv})"}')  
print(f'  Train ∩ Test date overlap   : {"PASS" if overlap_tt == 0 else f"FAIL ({overlap_tt})"}')  
print(f'  Val ∩ Test date overlap     : {"PASS" if overlap_vt == 0 else f"FAIL ({overlap_vt})"}')  

# 3. Shape consistency: X and y row counts match
for name, X, y in [('train', X_train_df, y_train), ('val', X_val_df, y_val), ('test', X_test_df, y_test)]:
    ok = len(X) == len(y)
    print(f'  {name} X/y row alignment    : {"PASS" if ok else "FAIL"} ({len(X):,} vs {len(y):,})')
    if not ok:
        all_pass = False

# 4. Target encoding applied from train — check val/test rate columns are in train range
train_min = X_train_df[[c for c in feature_names if 'DelayRate' in c]].min().min()
train_max = X_train_df[[c for c in feature_names if 'DelayRate' in c]].max().max()
for name, X in [('val', X_val_df), ('test', X_test_df)]:
    rate_cols_pp = [c for c in feature_names if 'DelayRate' in c]
    in_range = (X[rate_cols_pp].min().min() >= train_min - 1e-6 and
                X[rate_cols_pp].max().max() <= train_max + 1e-6)
    print(f'  {name} delay rates in train range: {"PASS" if in_range else "WARN (check)"}') 

print()
print(f'Overall: {"ALL CHECKS PASSED ✓" if all_pass else "SOME CHECKS FAILED — review above"}')


# In[32]:


# Final shape and target summary
print('=== Final Dataset Summary ===')
print(f'{'Split':<10} {'Rows':>10} {'Features':>10} {'Delay Rate':>12}')
print('-' * 46)
for name, X, y in [('train', X_train_df, y_train), ('val', X_val_df, y_val), ('test', X_test_df, y_test)]:
    print(f'{name:<10} {len(X):>10,} {X.shape[1]:>10} {y.mean()*100:>11.1f}%')
print()
print(f'Feature breakdown:')
print(f'  StandardScaler features : {len(SCALE_COLS)}')
print(f'  OHE features            : {len(feature_names) - len(SCALE_COLS) - len(PASSTHROUGH_COLS)}')
print(f'  Passthrough features    : {len(PASSTHROUGH_COLS)}')
print(f'  Total features          : {len(feature_names)}')


# ---
# ## Section 8 — Save Artifacts

# In[33]:


artifacts = {
    'train_X': X_train_df,
    'train_y': y_train.reset_index(drop=True).rename('ArrDel15').to_frame(),
    'val_X'  : X_val_df,
    'val_y'  : y_val.reset_index(drop=True).rename('ArrDel15').to_frame(),
    'test_X' : X_test_df,
    'test_y' : y_test.reset_index(drop=True).rename('ArrDel15').to_frame(),
}

print('Saving parquet artifacts...')
for name, frame in artifacts.items():
    out_path = PROCESSED_DIR / f'{name}.parquet'
    frame.to_parquet(out_path, index=False, compression='snappy')
    size_mb = out_path.stat().st_size / 1e6
    print(f'  {name}.parquet  →  {out_path.name}  ({size_mb:.1f} MB,  {len(frame):,} rows × {frame.shape[1]} cols)')

print()
print('Saving preprocessor...')
preprocessor_path = PROCESSED_DIR / 'preprocessor.joblib'
joblib.dump(preprocessor, preprocessor_path)
print(f'  preprocessor.joblib  →  {preprocessor_path.stat().st_size / 1e3:.1f} KB')


# In[34]:


# ── Round-trip verification ───────────────────────────────────────────────────
print('Round-trip verification...')
for name in ['train_X', 'val_X', 'test_X']:
    reloaded = pd.read_parquet(PROCESSED_DIR / f'{name}.parquet')
    expected = artifacts[name]
    assert reloaded.shape == expected.shape, f'{name}: shape mismatch'
    assert list(reloaded.columns) == list(expected.columns), f'{name}: column mismatch'
    print(f'  {name}: {reloaded.shape} ✓')

# Reload and verify preprocessor
pp_reloaded = joblib.load(preprocessor_path)
test_transform = pp_reloaded.transform(X_test.head(5))
assert test_transform.shape[1] == len(feature_names), 'Preprocessor round-trip: feature count mismatch'
print(f'  preprocessor.joblib: reloaded and transform tested ✓')

print()
print('=== All artifacts saved and verified. ===')
print(f'Output directory: {PROCESSED_DIR}')
for f in sorted(PROCESSED_DIR.glob('*.parquet')) :
    print(f'  {f.name}  ({f.stat().st_size/1e6:.1f} MB)')
print(f'  preprocessor.joblib  ({preprocessor_path.stat().st_size/1e3:.1f} KB)')


# ---
# ## Summary
# 
# This notebook produced a fully preprocessed, leakage-safe dataset ready for modelling:
# 
# 1. **Chronological split** applied first — no future data contaminated training statistics
# 2. **Time features** — 3 binary flags + 6 cyclical (sin/cos) encoding features
# 3. **Route features** — route string + hub flags (hub list from training volume only)
# 4. **Smoothed target encoding** — 5 delay rate features computed from training, applied to all splits with global-mean fallback for unseen categories
# 5. **ColumnTransformer** — StandardScaler + OHE + passthrough, fit on training only
# 6. **Validation** — NaN checks, date overlap checks, shape alignment confirmed
# 
# **Next step:** `04_modelling.ipynb` — train baseline (logistic regression) and primary (gradient-boosted trees) models, evaluate on validation set using ROC-AUC and PR-AUC, touch test set once for final reporting.
