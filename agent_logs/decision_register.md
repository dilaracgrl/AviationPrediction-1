# Decision Register
| # | Stage | Decision | Agent Suggested | I Decided | Reason |
|---|-------|----------|-----------------|-----------|--------|
| 1 | Data Loading | Column rename strategy | Apply a single dict-based rename map after concatenating all CSVs (one rename pass, not per-file) to keep code DRY and reduce overhead | | |
| 2 | Data Loading | File discovery pattern | Use `glob.glob('data/raw/*.csv')` sorted alphabetically; assumes all CSVs in raw/ are monthly BTS files | | |
| 3 | Cleaning | Order of cleaning steps | Apply steps in sequence: (1) remove cancelled, (2) remove diverted, (3) drop ArrDel15 NaN, (4) parse FlightDate, (5) range filters on Distance/CRSDepTime/CRSArrTime, (6) deduplicate — so each step's row count is a clean checkpoint | | |
| 4 | Leakage prevention | Columns excluded from model dataset | Excluded DepDelay, DepDel15, ArrDelay, CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay, Cancelled, Diverted — all are post-departure actuals or consequence variables | | |
| 5 | Feature engineering | Derived time features | Extracted CRSDepHour (CRSDepTime // 100) and CRSDepMinute (CRSDepTime % 100) from scheduled departure time as continuous numeric features representing time-of-day | | |
| 6 | Output format | Parquet for processed dataset | Used pandas to_parquet with default snappy compression; preserves dtypes (especially datetime and int8 for ArrDel15) efficiently | | |
| 7 | Quality checks | Duplicate key definition | Key = (FlightDate, Reporting_Airline, Origin, Dest, CRSDepTime) — matches BTS conceptual uniqueness; a flight is uniquely identified by date, carrier, route, and scheduled departure | | |
