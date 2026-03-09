# Data Access Instructions

Raw data files and processed parquet files are excluded from this repository due to file size. Follow these instructions to reproduce the dataset.

For a full description of every raw variable, renamed column, and engineered feature used in this project, see **[DATA_DICTIONARY.md](DATA_DICTIONARY.md)**.

---

## Source

**Dataset:** BTS Reporting Carrier On-Time Performance
**Provider:** U.S. Bureau of Transportation Statistics
**Download URL:** https://transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr

---

## Months to Download

Download **each month separately** (one download per month):

| Month | Year |
|-------|------|
| January | 2025 |
| February | 2025 |
| March | 2025 |
| April | 2025 |
| May | 2025 |
| June | 2025 |
| July | 2025 |

---

## Columns to Select

On the BTS download page, tick **every** field listed below. Untick all others.

**Time Period**
- Year
- Quarter
- Month
- DayofMonth
- DayOfWeek
- FlightDate

**Airline**
- Reporting_Airline
- Tail_Number

**Origin**
- OriginAirportID
- Origin
- OriginState

**Destination**
- DestAirportID
- Dest
- DestState

**Departure Performance**
- CRSDepTime
- DepDelay
- DepDel15
- DepTimeBlk

**Arrival Performance**
- CRSArrTime
- ArrDelay
- ArrDel15

**Cancellations and Diversions**
- Cancelled
- CancellationCode
- Diverted

**Flight Summaries**
- CRSElapsedTime
- Distance
- DistanceGroup

**Cause of Delay**
- CarrierDelay
- WeatherDelay
- NASDelay
- SecurityDelay
- LateAircraftDelay

---

## Setup After Downloading

1. Place all downloaded CSV files into `data/raw/`:

```
data/raw/T_ONTIME_REPORTING_JAN.csv
data/raw/T_ONTIME_REPORTING_FEV.csv
data/raw/T_ONTIME_REPORTING_march.csv
data/raw/T_ONTIME_REPORTING_APRIL.csv
data/raw/T_ONTIME_REPORTING_MAY.csv
data/raw/T_ONTIME_REPORTING_JUNE.csv
data/raw/T_ONTIME_REPORTING_JULY.csv
```

> **Note:** BTS export filenames vary by month. The column rename map in notebook 01 handles any naming inconsistencies automatically via an `if k in raw.columns` guard.

2. Run the pipeline to regenerate all processed files. Either run the notebooks in order (01 → 05) or, from project root, run `./run_pipeline.sh` (executes `src/` scripts 01–05).

```
notebooks/01_data_loading_and_quality.ipynb   → data/processed/model_dataset.parquet
notebooks/02_eda.ipynb                         → reports/01–14_*.png, reports/eda_summary.md
notebooks/03_data_preparation.ipynb            → data/processed/{train,val,test}_{X,y}.parquet
                                                 data/processed/preprocessor.joblib
notebooks/04_modelling.ipynb                   → data/processed/best_model.joblib
                                                 reports/15–19_*.png, reports/model_comparison.csv
notebooks/05_error_analysis.ipynb              → reports/20–29_*.png
                                                 reports/error_analysis_summary.md
```

---

## Notes on BTS Date Format

BTS exports `FL_DATE` as `'M/D/YYYY 12:00:00 AM'` (not ISO format). Notebook 01 handles this automatically using `pd.to_datetime()` without an explicit format argument.

## Expected Output

After running notebook 01, `data/processed/model_dataset.parquet` should be approximately **62 MB** containing **3,997,256 rows × 23 columns** with a delay rate of approximately 23%.
