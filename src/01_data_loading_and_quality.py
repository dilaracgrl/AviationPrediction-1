#!/usr/bin/env python
# coding: utf-8

# # 01 — Data Loading & Quality Assessment
# 
# **Project:** Flight Arrival Delay Prediction  
# **Data Source:** BTS On-Time Performance (January–June 2025)  
# **Target Variable:** `ArrDel15` — binary indicator, 1 if arrival delay ≥ 15 minutes  
# 
# ---
# 
# This notebook covers:
# 1. Loading and concatenating six monthly BTS CSV files
# 2. Standardising column names
# 3. Quality checks (shapes, dtypes, value ranges, missingness, duplicates, monthly counts)
# 4. A documented cleaning pipeline with row-count checkpoints
# 5. Building a leakage-safe model dataset and saving it as Parquet
# 
# All decisions about which columns to keep or drop are logged in `agent_logs/decision_register.md`.

# ## 0. Imports & Configuration

# In[1]:


import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.2f}'.format)

RAW_DIR       = '../data/raw/'
PROCESSED_DIR = '../data/processed/'
OUTPUT_PATH   = os.path.join(PROCESSED_DIR, 'model_dataset.parquet')

os.makedirs(PROCESSED_DIR, exist_ok=True)

print('Libraries loaded.')
print(f'NumPy  : {np.__version__}')
print(f'Pandas : {pd.__version__}')


# ## 1. Load Raw CSV Files
# 
# BTS publishes one CSV per month. We discover all CSVs in `data/raw/` automatically using `glob`, sort them so they are read in chronological order, and concatenate them into a single DataFrame.
# 
# > **Why glob?** Hard-coding six filenames is brittle; glob picks up the files regardless of exact naming conventions and makes it trivial to add more months later.

# In[2]:


csv_files = sorted(glob.glob(os.path.join(RAW_DIR, '*.csv')))

print(f'Found {len(csv_files)} CSV file(s):')
for f in csv_files:
    print(f'  {os.path.basename(f)}')


# In[3]:


# Read each file, store in a list, then concatenate once — cheaper than repeated pd.concat inside a loop
frames = []
for path in csv_files:
    month_df = pd.read_csv(
        path,
        low_memory=False,   # avoids mixed-type warnings on large files
    )
    frames.append(month_df)
    print(f'  {os.path.basename(path):45s}  rows={len(month_df):>8,}  cols={month_df.shape[1]}')

raw = pd.concat(frames, ignore_index=True)
print(f'\nConcatenated shape: {raw.shape[0]:,} rows × {raw.shape[1]} columns')


# ## 2. Column Renaming
# 
# BTS uses `UPPER_WITH_UNDERSCORES` naming. We rename to a mixed-case convention that is more readable in code and consistent with common BTS documentation references (e.g., `ArrDel15`, `DepDelay`).
# 
# Only the columns listed in the project specification are renamed; any extra BTS columns are left as-is and will naturally be excluded when we select the final feature set.

# In[4]:


RENAME_MAP = {
    'YEAR'                : 'Year',
    'QUARTER'             : 'Quarter',
    'MONTH'               : 'Month',
    'DAY_OF_MONTH'        : 'DayofMonth',
    'DAY_OF_WEEK'         : 'DayOfWeek',
    'FL_DATE'             : 'FlightDate',
    'OP_UNIQUE_CARRIER'   : 'Reporting_Airline',
    'TAIL_NUM'            : 'Tail_Number',
    'ORIGIN_AIRPORT_ID'   : 'OriginAirportID',
    'ORIGIN'              : 'Origin',
    'ORIGIN_STATE_ABR'    : 'OriginState',
    'DEST_AIRPORT_ID'     : 'DestAirportID',
    'DEST'                : 'Dest',
    'DEST_STATE_ABR'      : 'DestState',
    'CRS_DEP_TIME'        : 'CRSDepTime',
    'DEP_DELAY'           : 'DepDelay',
    'DEP_DEL15'           : 'DepDel15',
    'DEP_TIME_BLK'        : 'DepTimeBlk',
    'CRS_ARR_TIME'        : 'CRSArrTime',
    'ARR_DELAY'           : 'ArrDelay',
    'ARR_DEL15'           : 'ArrDel15',
    'CRS_ELAPSED_TIME'    : 'CRSElapsedTime',
    'DISTANCE'            : 'Distance',
    'DISTANCE_GROUP'      : 'DistanceGroup',
    'CANCELLED'           : 'Cancelled',
    'DIVERTED'            : 'Diverted',
    'CARRIER_DELAY'       : 'CarrierDelay',
    'WEATHER_DELAY'       : 'WeatherDelay',
    'NAS_DELAY'           : 'NASDelay',
    'SECURITY_DELAY'      : 'SecurityDelay',
    'LATE_AIRCRAFT_DELAY' : 'LateAircraftDelay',
}

# Only rename columns that actually exist to avoid KeyErrors on optional BTS fields
rename_existing = {k: v for k, v in RENAME_MAP.items() if k in raw.columns}
raw = raw.rename(columns=rename_existing)

print(f'Renamed {len(rename_existing)} columns.')
missing_from_map = set(RENAME_MAP.keys()) - set(rename_existing.keys())
if missing_from_map:
    print(f'Columns in rename map but NOT in data: {missing_from_map}')


# ## 3. Quality Checks
# 
# Before any cleaning we freeze a snapshot of the raw data and run four diagnostic checks:
# 
# - **3.1** Shape and dtypes
# - **3.2** Value-range validation for key categorical/flag columns
# - **3.3** Missingness per column
# - **3.4** Duplicate detection on the flight uniqueness key
# - **3.5** Month-by-month row counts

# ### 3.1 Shape & Data Types

# In[5]:


print(f'Shape : {raw.shape[0]:,} rows × {raw.shape[1]} columns')
print()
print('Column dtypes:')
print(raw.dtypes.to_string())


# In[24]:


print('First 5 rows:')
raw.head(5)


# ### 3.2 Value-Range Validation
# 
# We check the logical bounds for the calendar and flag columns. Violations indicate either upstream data issues or unexpected file contents.

# In[7]:


range_checks = [
    ('Month',      1,    12),
    ('DayofMonth', 1,    31),
    ('DayOfWeek',  1,     7),
    ('Cancelled',  0,     1),
    ('Diverted',   0,     1),
]

print(f'{'Column':<20} {'Min':>10} {'Max':>10} {'Out-of-range':>15}')
print('-' * 60)
for col, lo, hi in range_checks:
    if col not in raw.columns:
        print(f'{col:<20} {'MISSING COLUMN':>35}')
        continue
    col_clean = pd.to_numeric(raw[col], errors='coerce')
    col_min   = col_clean.min()
    col_max   = col_clean.max()
    out       = ((col_clean < lo) | (col_clean > hi)).sum()
    flag      = '  <<< CHECK' if out > 0 else ''
    print(f'{col:<20} {col_min:>10.0f} {col_max:>10.0f} {out:>15,}{flag}')


# ### 3.3 Missingness Per Column
# 
# A sorted table of null counts helps prioritise which columns need attention. Columns with 100% missingness after filtering (e.g., delay-cause columns for non-delayed flights) are expected.

# In[8]:


null_counts = raw.isnull().sum()
null_pct    = (null_counts / len(raw) * 100).round(2)

missingness = (
    pd.DataFrame({'null_count': null_counts, 'null_pct': null_pct})
    .sort_values('null_count', ascending=False)
)

# Show only columns with at least one null
print('Columns with missing values (sorted by count):')
display(missingness[missingness['null_count'] > 0])


# ### 3.4 Duplicate Detection
# 
# A flight is conceptually unique if (FlightDate, Reporting_Airline, Origin, Dest, CRSDepTime) is unique — two rows sharing all five values represent the same scheduled flight.

# In[9]:


DUP_KEY = ['FlightDate', 'Reporting_Airline', 'Origin', 'Dest', 'CRSDepTime']

# Check which key columns are present
missing_key_cols = [c for c in DUP_KEY if c not in raw.columns]
if missing_key_cols:
    print(f'WARNING: key columns not found in data: {missing_key_cols}')
else:
    n_dups = raw.duplicated(subset=DUP_KEY).sum()
    print(f'Duplicate rows on key {DUP_KEY}:')
    print(f'  {n_dups:,} duplicate rows ({n_dups / len(raw) * 100:.3f}% of total)')
    if n_dups > 0:
        print('\nSample of duplicated records:')
        dup_mask = raw.duplicated(subset=DUP_KEY, keep=False)
        display(raw[dup_mask][DUP_KEY + ['ArrDel15']].sort_values(DUP_KEY).head(10))


# ### 3.5 Month-by-Month Row Counts
# 
# This sanity check confirms that each month contributed a reasonable number of rows and that no month was accidentally loaded twice or missed.

# In[25]:


monthly = (
    raw.groupby('Month', sort=True)
    .size()
    .rename('row_count')
    .reset_index()
)
monthly['month_name'] = monthly['Month'].map({
    1: 'January', 2: 'February', 3: 'March',
    4: 'April',   5: 'May',      6: 'June',
    7: 'July',    8: 'August',   9: 'September',
    10: 'October', 11: 'November', 12: 'December'
})
monthly['cumulative'] = monthly['row_count'].cumsum()

print('Month-by-month row counts:')
display(monthly[['Month', 'month_name', 'row_count', 'cumulative']])
print(f'\nTotal: {monthly["row_count"].sum():,}')


# ## 4. Cleaning Pipeline
# 
# We clean the data in six documented steps. After each step we print the row count removed so you can trace exactly where data was lost.
# 
# | Step | Action | Rationale |
# |------|--------|-----------|
# | 1 | Remove cancelled flights | No arrival outcome |
# | 2 | Remove diverted flights | Arrival airport differs from plan |
# | 3 | Drop rows where `ArrDel15` is NaN | Target is unobserved |
# | 4 | Parse `FlightDate` to datetime | Required for feature engineering later |
# | 5 | Filter invalid Distance / time values | Likely data-entry errors |
# | 6 | Deduplicate on flight key | Keep first occurrence |

# In[26]:


df = raw.copy()
cleaning_log = []

def checkpoint(df, step_name, rows_before):
    rows_after   = len(df)
    rows_removed = rows_before - rows_after
    cleaning_log.append({
        'step'         : step_name,
        'rows_before'  : rows_before,
        'rows_removed' : rows_removed,
        'rows_after'   : rows_after,
        'pct_removed'  : rows_removed / rows_before * 100 if rows_before > 0 else 0,
    })
    return rows_after

rows = len(df)
print(f'Starting rows: {rows:,}')


# In[27]:


# Step 1: Remove cancelled flights (Cancelled == 1)
rows_before = rows
df = df[df['Cancelled'] != 1].copy()
rows = checkpoint(df, 'Step 1: Remove cancelled flights', rows_before)
print(f'Step 1 complete — rows remaining: {rows:,}')


# In[28]:


# Step 2: Remove diverted flights (Diverted == 1)
rows_before = rows
df = df[df['Diverted'] != 1].copy()
rows = checkpoint(df, 'Step 2: Remove diverted flights', rows_before)
print(f'Step 2 complete — rows remaining: {rows:,}')


# In[29]:


# Step 3: Drop rows where ArrDel15 is NaN (target unobserved)
rows_before = rows
df = df.dropna(subset=['ArrDel15']).copy()
rows = checkpoint(df, 'Step 3: Drop missing ArrDel15', rows_before)
print(f'Step 3 complete — rows remaining: {rows:,}')


# In[15]:


row_count = len(df)
print(row_count)


# In[30]:


# Step 4: Parse FlightDate to datetime (auto-detect BTS format)
rows_before = rows
# BTS files encode FlightDate as 'M/D/YYYY 12:00:00 AM' (e.g., '1/9/2025 12:00:00 AM')
# Do not specify format, so pandas infers correctly. Normalize to strip time.
df['FlightDate'] = pd.to_datetime(df['FlightDate'], errors='coerce').dt.normalize()
n_bad_dates = df['FlightDate'].isna().sum()
if n_bad_dates > 0:
    print(f'  {n_bad_dates:,} rows had unparseable FlightDate — dropping them.')
    df = df.dropna(subset=['FlightDate'])
rows = checkpoint(df, 'Step 4: Parse FlightDate', rows_before)
print(f'Step 4 complete — rows remaining: {rows:,}')
print(f'  FlightDate range: {df["FlightDate"].min().date()} to {df["FlightDate"].max().date()}')


# Here we had an issue where the agent incorrectly read the data - our data was in date time format but when trying to change a date format it assumed our normal date format was Year-month-day without time which allowed it to be unparseable and that consequently led to dropping the rows and delting the data.

# In[31]:


# Step 5: Remove rows with invalid Distance, CRSDepTime, or CRSArrTime
rows_before = rows

distance_ok  = pd.to_numeric(df['Distance'],    errors='coerce').fillna(0) > 0
dep_time_ok  = pd.to_numeric(df['CRSDepTime'],  errors='coerce').between(0, 2359, inclusive='both')
arr_time_ok  = pd.to_numeric(df['CRSArrTime'],  errors='coerce').between(0, 2359, inclusive='both')

valid_mask = distance_ok & dep_time_ok & arr_time_ok
df = df[valid_mask].copy()
rows = checkpoint(df, 'Step 5: Remove invalid Distance/CRSDepTime/CRSArrTime', rows_before)
print(f'Step 5 complete — rows remaining: {rows:,}')


# In[33]:


# Step 6: Deduplicate on flight key — keep first occurrence
rows_before = rows
dup_key_present = [c for c in DUP_KEY if c in df.columns]
df = df.drop_duplicates(subset=dup_key_present, keep='first').copy()
rows = checkpoint(df, 'Step 6: Deduplicate on flight key', rows_before)
print(f'Step 6 complete — rows remaining: {rows:,}')


# ### 4.1 Cleaning Summary

# In[19]:


cleaning_summary = pd.DataFrame(cleaning_log)
cleaning_summary['pct_removed'] = cleaning_summary['pct_removed'].map('{:.3f}%'.format)

print('=== Cleaning Pipeline Summary ===')
display(cleaning_summary)

total_removed = len(raw) - len(df)
print(f'\nTotal rows removed : {total_removed:,}  ({total_removed / len(raw) * 100:.2f}% of raw)')
print(f'Final clean rows   : {len(df):,}')


# ## 5. Build Leakage-Safe Model Dataset
# 
# A core discipline in predictive modelling is ensuring that no information available only **after** the event leaks into the training features. For flight delay prediction, the "event" is departure — anything that occurs once the plane leaves the gate is off-limits.
# 
# **Excluded columns (post-departure actuals):**
# 
# | Column | Reason |
# |--------|--------|
# | `DepDelay` | Actual departure delay — unknown at prediction time |
# | `DepDel15` | Binary version of `DepDelay` — same issue |
# | `ArrDelay` | Actual arrival delay (in minutes) — this *is* the target |
# | `CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay` | BTS delay cause codes — assigned post-flight |
# | `Cancelled`, `Diverted` | Already filtered out; including them would be redundant |
# 
# **Derived features added:**
# 
# - `CRSDepHour` — hour component of `CRSDepTime` (0–23)
# - `CRSDepMinute` — minute component of `CRSDepTime` (0–59)

# In[20]:


# Derive hour and minute from scheduled departure time
df['CRSDepTime']   = pd.to_numeric(df['CRSDepTime'], errors='coerce')
df['CRSDepHour']   = (df['CRSDepTime'] // 100).astype('Int16')
df['CRSDepMinute'] = (df['CRSDepTime'] %  100).astype('Int16')

print('Derived features added: CRSDepHour, CRSDepMinute')
print(df[['CRSDepTime', 'CRSDepHour', 'CRSDepMinute']].head(5))


# In[21]:


# Define the final leakage-safe feature set
MODEL_COLUMNS = [
    # Calendar
    'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate',
    # Carrier
    'Reporting_Airline', 'Tail_Number',
    # Route
    'Origin', 'Dest', 'OriginAirportID', 'DestAirportID', 'OriginState', 'DestState',
    # Schedule
    'CRSDepTime', 'CRSArrTime', 'CRSElapsedTime', 'Distance', 'DistanceGroup', 'DepTimeBlk',
    # Derived
    'CRSDepHour', 'CRSDepMinute',
    # Target
    'ArrDel15',
]

# Keep only columns that exist in the cleaned DataFrame
final_cols = [c for c in MODEL_COLUMNS if c in df.columns]
missing_cols = [c for c in MODEL_COLUMNS if c not in df.columns]

if missing_cols:
    print(f'WARNING — expected columns not found, will be omitted: {missing_cols}')

# Defence in depth: assert no post-departure leakage columns in model dataset
LEAKAGE_COLS = ['DepDelay', 'ArrDelay', 'DepDel15']
leaked = [c for c in LEAKAGE_COLS if c in final_cols]
assert len(leaked) == 0, f'LEAKAGE: post-departure columns in model dataset: {leaked}'

model_df = df[final_cols].copy()

# Cast ArrDel15 to int8 (binary 0/1 target)
model_df['ArrDel15'] = model_df['ArrDel15'].astype('int8')

print(f'Model dataset shape: {model_df.shape[0]:,} rows × {model_df.shape[1]} columns')
print(f'\nTarget distribution (ArrDel15):')
vc = model_df['ArrDel15'].value_counts().sort_index()
for val, count in vc.items():
    label = 'On-time (0)' if val == 0 else 'Delayed (1)'
    print(f'  {label}: {count:>9,}  ({count / len(model_df) * 100:.1f}%)')


# In[22]:


print('Final model dataset columns and dtypes:')
print(model_df.dtypes.to_string())


# ## 6. Save to Parquet
# 
# Parquet is chosen over CSV for the processed dataset because:
# - It preserves column dtypes (especially `datetime64` for `FlightDate` and `int8` for `ArrDel15`)
# - Snappy compression reduces file size ~5× compared to raw CSV
# - Column-oriented storage makes downstream column-subset reads fast

# In[23]:


model_df.to_parquet(OUTPUT_PATH, index=False, compression='snappy')

file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 ** 2)
print(f'Saved: {OUTPUT_PATH}')
print(f'File size: {file_size_mb:.1f} MB')

# Verification round-trip
verify = pd.read_parquet(OUTPUT_PATH)
assert verify.shape == model_df.shape, 'Shape mismatch on verification read!'
assert list(verify.columns) == list(model_df.columns), 'Column mismatch on verification read!'
print(f'Verification passed — {verify.shape[0]:,} rows × {verify.shape[1]} columns read back correctly.')


# ## 7. Summary
# 
# This notebook has:
# 
# 1. **Loaded** 6 monthly BTS CSV files from `data/raw/` and concatenated them into a single DataFrame
# 2. **Renamed** 31 columns from BTS `UPPER_UNDERSCORE` convention to a readable mixed-case convention
# 3. **Checked** data quality across shape, dtypes, value ranges, missingness, duplicates, and monthly counts
# 4. **Cleaned** the data through 6 documented steps, removing cancelled/diverted flights, missing targets, invalid values, and duplicates
# 5. **Selected** a leakage-safe feature set (pre-departure only) with two derived time features
# 6. **Saved** the processed dataset to `data/processed/model_dataset.parquet`
# 
# **Next step:** `02_exploratory_data_analysis.ipynb` — univariate and bivariate analysis of the feature set.
