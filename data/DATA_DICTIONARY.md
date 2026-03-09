# Data Dictionary — Flight Arrival Delay Prediction

**Project:** AviationPrediction-1
**Target variable:** `ArrDel15` — binary indicator, 1 = flight arrived ≥15 minutes late
**Prediction point:** Scheduled departure time (all model features must be known before the aircraft pushes back)

---

## Section 1 — Raw BTS Variables

Variables as downloaded from the BTS Reporting Carrier On-Time Performance dataset. Column names follow the BTS export convention (uppercase with underscores). Not all of these enter the model — see Section 4 for the leakage boundary.

| Variable Name | Description | Type | Example Value | Source |
|---------------|-------------|------|---------------|--------|
| `YEAR` | Year of the flight | int | 2025 | BTS |
| `QUARTER` | Calendar quarter (1–4) | int | 1 | BTS |
| `MONTH` | Calendar month (1–12) | int | 1 | BTS |
| `DAY_OF_MONTH` | Day of the month (1–31) | int | 15 | BTS |
| `DAY_OF_WEEK` | Day of week: 1 = Monday, 7 = Sunday | int | 3 | BTS |
| `FL_DATE` | Full flight date. BTS exports this as `'M/D/YYYY 12:00:00 AM'` — auto-parsed with `pd.to_datetime()` | date | 2025-01-15 | BTS |
| `OP_UNIQUE_CARRIER` | Reporting carrier IATA code (unique carrier identifier) | string | `"AA"` | BTS |
| `TAIL_NUM` | Aircraft tail number (FAA registry identifier) | string | `"N12345"` | BTS |
| `ORIGIN_AIRPORT_ID` | Origin airport numeric ID assigned by the US DOT. Not used as a model feature (surrogate key with no ordinal meaning) | int | 10140 | BTS |
| `ORIGIN` | Origin airport IATA three-letter code | string | `"DFW"` | BTS |
| `ORIGIN_STATE_ABR` | Two-letter abbreviation of origin state | string | `"TX"` | BTS |
| `DEST_AIRPORT_ID` | Destination airport numeric ID assigned by the US DOT. Not used as a model feature (surrogate key) | int | 11298 | BTS |
| `DEST` | Destination airport IATA three-letter code | string | `"LAX"` | BTS |
| `DEST_STATE_ABR` | Two-letter abbreviation of destination state | string | `"CA"` | BTS |
| `CRS_DEP_TIME` | Scheduled departure time in local time, HHMM integer format. Note: not a continuous variable — jumps at the hour boundary (e.g. 1259 → 1300) | int | 1430 | BTS |
| `DEP_DELAY` | Actual departure delay in minutes. Negative values indicate early departure. **Post-departure actual — excluded from model.** | float | 15.0 | BTS |
| `DEP_DEL15` | Binary indicator: 1 if the flight departed 15 or more minutes late. **Post-departure actual — excluded from model.** | int | 0 | BTS |
| `DEP_TIME_BLK` | Scheduled departure time block in hourly intervals (e.g. `'1400-1459'`). Derived from scheduled time, so safe to use as a feature | string | `"1400-1459"` | BTS |
| `CRS_ARR_TIME` | Scheduled arrival time in local time, HHMM integer format | int | 1730 | BTS |
| `ARR_DELAY` | Actual arrival delay in minutes. Negative values indicate early arrival. **Post-departure actual — excluded from model.** | float | −5.0 | BTS |
| `ARR_DEL15` | **TARGET VARIABLE.** Binary indicator: 1 if the flight arrived 15 or more minutes late, 0 otherwise. Flights with missing values are dropped during cleaning | int | 1 | BTS |
| `CRS_ELAPSED_TIME` | Scheduled flight duration in minutes (block time) | float | 180.0 | BTS |
| `DISTANCE` | Great-circle distance between origin and destination airports in statute miles | float | 569.0 | BTS |
| `DISTANCE_GROUP` | Categorical distance grouping: interval = 250 miles (1 = 0–250 mi, 2 = 251–500 mi, etc.) | int | 3 | BTS |
| `CANCELLED` | Flight cancellation indicator: 1 = cancelled. Cancelled flights are removed during cleaning | int | 0 | BTS |
| `CANCELLATION_CODE` | Reason for cancellation: A = Carrier, B = Weather, C = National Air System, D = Security. Null for non-cancelled flights | string | `"B"` | BTS |
| `DIVERTED` | Flight diversion indicator: 1 = diverted. Diverted flights are removed during cleaning | int | 0 | BTS |
| `CARRIER_DELAY` | Minutes of delay attributable to carrier (maintenance, crew, aircraft cleaning, etc.). Null for on-time flights. **Post-departure actual — excluded from model.** | float | 30.0 | BTS |
| `WEATHER_DELAY` | Minutes of delay attributable to weather. Null for on-time flights. **Post-departure actual — excluded from model.** | float | 0.0 | BTS |
| `NAS_DELAY` | Minutes of delay attributable to the National Air System (ATC, congestion, non-extreme weather). Null for on-time flights. **Post-departure actual — excluded from model.** | float | 10.0 | BTS |
| `SECURITY_DELAY` | Minutes of delay attributable to security issues. Null for on-time flights. **Post-departure actual — excluded from model.** | float | 0.0 | BTS |
| `LATE_AIRCRAFT_DELAY` | Minutes of delay attributable to a late-arriving aircraft on a previous leg. Null for on-time flights. **Post-departure actual — excluded from model.** | float | 20.0 | BTS |

---

## Section 2 — Column Rename Mapping

Raw BTS column names are renamed to consistent mixed-case names in this project. The rename map is applied in `notebooks/01_data_loading_and_quality.ipynb` using a single dict-based pass after concatenating all monthly CSVs. Columns absent from a given month's export are silently skipped (`if k in raw.columns`).

| Raw BTS Name | Project Name | Description |
|--------------|--------------|-------------|
| `YEAR` | `Year` | Year of flight |
| `QUARTER` | `Quarter` | Calendar quarter |
| `MONTH` | `Month` | Calendar month |
| `DAY_OF_MONTH` | `DayofMonth` | Day of month |
| `DAY_OF_WEEK` | `DayOfWeek` | Day of week (1=Mon, 7=Sun) |
| `FL_DATE` | `FlightDate` | Parsed flight date (datetime, time component stripped) |
| `OP_UNIQUE_CARRIER` | `Reporting_Airline` | Carrier IATA code |
| `TAIL_NUM` | `Tail_Number` | Aircraft tail number |
| `ORIGIN_AIRPORT_ID` | `OriginAirportID` | DOT origin airport ID |
| `ORIGIN` | `Origin` | Origin IATA code |
| `ORIGIN_STATE_ABR` | `OriginState` | Origin state abbreviation |
| `DEST_AIRPORT_ID` | `DestAirportID` | DOT destination airport ID |
| `DEST` | `Dest` | Destination IATA code |
| `DEST_STATE_ABR` | `DestState` | Destination state abbreviation |
| `CRS_DEP_TIME` | `CRSDepTime` | Scheduled departure time (HHMM) |
| `DEP_DELAY` | `DepDelay` | Actual departure delay (minutes) |
| `DEP_DEL15` | `DepDel15` | Departure delay ≥15 min indicator |
| `DEP_TIME_BLK` | `DepTimeBlk` | Scheduled departure time block |
| `CRS_ARR_TIME` | `CRSArrTime` | Scheduled arrival time (HHMM) |
| `ARR_DELAY` | `ArrDelay` | Actual arrival delay (minutes) |
| `ARR_DEL15` | `ArrDel15` | **Target variable** — arrival delay ≥15 min |
| `CRS_ELAPSED_TIME` | `CRSElapsedTime` | Scheduled elapsed time (minutes) |
| `DISTANCE` | `Distance` | Route distance (statute miles) |
| `DISTANCE_GROUP` | `DistanceGroup` | Distance interval category |
| `CANCELLED` | `Cancelled` | Cancellation indicator |
| `CANCELLATION_CODE` | `CancellationCode` | Cancellation reason code |
| `DIVERTED` | `Diverted` | Diversion indicator |
| `CARRIER_DELAY` | `CarrierDelay` | Carrier-caused delay (minutes) |
| `WEATHER_DELAY` | `WeatherDelay` | Weather-caused delay (minutes) |
| `NAS_DELAY` | `NASDelay` | NAS-caused delay (minutes) |
| `SECURITY_DELAY` | `SecurityDelay` | Security-caused delay (minutes) |
| `LATE_AIRCRAFT_DELAY` | `LateAircraftDelay` | Late-aircraft delay (minutes) |

---

## Section 3 — Engineered Features

All features below were created in `notebooks/03_data_preparation.ipynb`. The chronological split (train: Jan–Mar 2025, val: Apr 2025, test: Jun 2025) was applied **before** any feature engineering, ensuring that aggregate statistics computed on training data never contaminate the validation or test sets.

The final preprocessed dataset contains **59 features** grouped into three transformer groups: StandardScaler (17), OneHotEncoder (33), and passthrough (9).

### 3.1 Time Decomposition Features

These are deterministic transformations of scheduled times — no training-data dependency.

| Feature Name | Derivation | Type | Leakage Safe? |
|--------------|------------|------|---------------|
| `CRSDepHour` | `CRSDepTime // 100` — extracts the hour component (0–23) from the HHMM integer | int | ✅ Yes — derived from scheduled time known at booking |
| `CRSDepMinute` | `CRSDepTime % 100` — extracts the minute component (0–59) from the HHMM integer | int | ✅ Yes — derived from scheduled time known at booking |

### 3.2 Binary Flag Features

| Feature Name | Derivation | Type | Leakage Safe? |
|--------------|------------|------|---------------|
| `IsWeekend` | `1 if DayOfWeek >= 6 else 0` — flags Saturday (6) and Sunday (7) departures | int8 (0/1) | ✅ Yes — day of week is known at scheduling |
| `IsEveningFlight` | `1 if CRSDepHour >= 17 else 0` — flags departures from 5 PM onward, when delay accumulation is highest | int8 (0/1) | ✅ Yes — derived from scheduled departure hour |
| `IsMorningFlight` | `1 if CRSDepHour <= 8 else 0` — flags departures up to and including 8 AM, when delay rates are lowest (clean aircraft, no propagation) | int8 (0/1) | ✅ Yes — derived from scheduled departure hour |
| `IsHubOrigin` | `1 if Origin is in the top-15 airports by training-set flight volume else 0`. Hub list: DEN, DFW, ATL, ORD, PHX, CLT, LAS, LAX, MCO, SEA, DCA, BOS, SFO, LGA, MIA | int8 (0/1) | ✅ Yes — hub list derived from training data only (volume, not delay rate), frozen before transformation |
| `IsHubDest` | `1 if Dest is in the same top-15 hub list else 0` | int8 (0/1) | ✅ Yes — same hub list, derived from training data only |

### 3.3 Cyclical Encoding Features

Raw hour and day integers have artificial discontinuities at their period boundaries (e.g. hour 23 and hour 0 are adjacent but numerically distant). Cyclical sin/cos encoding wraps these features onto a circle, preserving the natural topology.

Formula: `sin(2π × value / period)` and `cos(2π × value / period)`. Both sin and cos are always included together — neither alone encodes the direction within the cycle.

| Feature Name | Derivation | Period | Leakage Safe? |
|--------------|------------|--------|---------------|
| `DepHour_sin` | `sin(2π × CRSDepHour / 24)` | 24 | ✅ Yes — derived from scheduled departure hour |
| `DepHour_cos` | `cos(2π × CRSDepHour / 24)` | 24 | ✅ Yes — derived from scheduled departure hour |
| `Month_sin` | `sin(2π × Month / 12)` | 12 | ✅ Yes — calendar month is known at scheduling |
| `Month_cos` | `cos(2π × Month / 12)` | 12 | ✅ Yes — calendar month is known at scheduling |
| `DayOfWeek_sin` | `sin(2π × DayOfWeek / 7)` | 7 | ✅ Yes — day of week is known at scheduling |
| `DayOfWeek_cos` | `cos(2π × DayOfWeek / 7)` | 7 | ✅ Yes — day of week is known at scheduling |

> **Note:** `CRSDepHour` is dropped from the model after these features are created. It is replaced by `DepHour_sin`/`DepHour_cos` (cyclical), `IsMorningFlight`/`IsEveningFlight` (binary), and `HourDelayRate` (target encoded). The raw integer is not fed to the model to avoid conflating ordinal and cyclical structure.

### 3.4 Route Feature

| Feature Name | Derivation | Type | Leakage Safe? |
|--------------|------------|------|---------------|
| `Route` | `Origin + "_" + Dest` — string concatenation of IATA codes (e.g. `"LAX_JFK"`). Creates a composite city-pair key with higher granularity than origin or destination individually. ~5,908 unique routes in training data | string | ✅ Yes — purely deterministic from scheduled routing, no target information used |

> **Note:** `Route` itself is not passed to the model. It serves as the grouping key for `RouteDelayRate` target encoding, after which it is dropped.

### 3.5 Smoothed Target-Encoded Delay Rate Features

Target encoding replaces a categorical value with the historical mean of the target within that category, computed **on training data only**. A smoothing term (k = 100) regularises sparse categories by blending the category mean toward the global training mean:

$$\hat{\mu}_c = \frac{n_c \cdot \bar{y}_c + k \cdot \bar{y}_{\text{global}}}{n_c + k}$$

where $n_c$ is the number of training observations in category $c$, $\bar{y}_c$ is the category's mean delay rate, $\bar{y}_{\text{global}}$ = 0.197 (training mean), and $k$ = 100. Categories with fewer than 100 training observations are shrunk heavily toward the global mean. Unseen categories in validation and test receive the global training mean as fallback.

| Feature Name | Grouping Key | Training Range | Leakage Safe? |
|--------------|-------------|----------------|---------------|
| `AirlineDelayRate` | `Reporting_Airline` (14 carriers) | 0.160 – 0.287 | ✅ Yes — encoding map computed from training labels only; applied to val/test without refitting |
| `OriginDelayRate` | `Origin` (333 airports) | 0.099 – 0.347 | ✅ Yes — same guarantee |
| `DestDelayRate` | `Dest` (333 airports) | 0.106 – 0.346 | ✅ Yes — same guarantee |
| `RouteDelayRate` | `Route` (5,908 city pairs) | 0.075 – 0.418 | ✅ Yes — same guarantee. Unseen routes in val (137) and test (650) receive 0.197 fallback |
| `HourDelayRate` | `CRSDepHour` (24 values) | 0.108 – 0.261 | ✅ Yes — same guarantee |

> **Dual encoding of Reporting_Airline:** The airline column is both one-hot encoded (capturing which carrier) and target encoded as `AirlineDelayRate` (capturing expected delay magnitude). The redundancy is intentional — tree models benefit from having both the identity signal and the aggregate rate signal available.

---

## Section 4 — Leakage Boundary

### Prediction Point

This model predicts at **scheduled departure time**: the moment a flight is about to push back from the gate. All features must represent information that is genuinely known at that point — either from the published schedule or from historical statistics computed before the flight departs.

### Pre-Departure Variables — Safe to Use as Features

| Category | Variables | Rationale |
|----------|-----------|-----------|
| Schedule | `CRSDepTime`, `CRSArrTime`, `CRSElapsedTime`, `DepTimeBlk` | Published in the schedule; known at booking |
| Route | `Origin`, `Dest`, `Distance`, `DistanceGroup` | Fixed routing information |
| Carrier | `Reporting_Airline` | The operating carrier is known at ticket purchase |
| Aircraft | `Tail_Number` | Known at gate assignment (not used in final model due to high cardinality) |
| Calendar | `FlightDate`, `Year`, `Quarter`, `Month`, `DayofMonth`, `DayOfWeek` | Known at scheduling |
| Historical aggregates | `AirlineDelayRate`, `OriginDelayRate`, `DestDelayRate`, `RouteDelayRate`, `HourDelayRate` | Computed from *past* training flights, not from the flight being predicted |

### Post-Departure Variables — Excluded to Prevent Leakage

These variables are **only available after the flight departs or lands**. Using them as features would mean the model sees the answer before making a prediction — inflating apparent performance while producing a model that is useless in practice.

| Variable | Why Excluded |
|----------|-------------|
| `DepDelay` | Actual departure delay — only known after pushback |
| `DepDel15` | Actual departure delay indicator — only known after pushback |
| `ArrDelay` | Actual arrival delay — only known after landing |
| `CarrierDelay` | Cause-coded delay breakdown — only assigned post-arrival |
| `WeatherDelay` | Cause-coded delay breakdown — only assigned post-arrival |
| `NASDelay` | Cause-coded delay breakdown — only assigned post-arrival |
| `SecurityDelay` | Cause-coded delay breakdown — only assigned post-arrival |
| `LateAircraftDelay` | Cause-coded delay breakdown — only assigned post-arrival |
| `Cancelled` | Cancellation only known when the flight does not operate |
| `Diverted` | Diversion only known after the flight lands at the wrong airport |

### Why This Matters

A model trained with post-departure actuals as features would learn, for example, that `DepDelay > 0` strongly predicts `ArrDel15 = 1` — which is trivially true but operationally useless (if you know the flight has already departed late, you do not need a model to tell you it will probably arrive late). Such a model would report inflated metrics in evaluation but produce random or meaningless predictions when deployed, because the post-departure features it relied on would not yet exist at prediction time.

The strict leakage boundary applied in this project — split first, then engineer features from training statistics only — ensures that every reported metric reflects genuine out-of-sample predictive ability from schedule-time information alone.

---

## Variable Usage Summary

| Variable Group | Count | In Cleaned Dataset | In Model (59 features) |
|----------------|-------|--------------------|------------------------|
| Raw BTS columns (pre-cleaning) | 32 | — | — |
| Columns in `model_dataset.parquet` | 23 | ✅ | — |
| Leakage-excluded post-departure actuals | 10 | ❌ dropped at cleaning | ❌ |
| Surrogate ID columns dropped | 3 | ❌ dropped | ❌ |
| High-cardinality dropped (`Tail_Number`, states) | 3 | ✅ kept in parquet | ❌ |
| Schedule features (scaled) | 6 | ✅ | ✅ |
| Cyclical encodings | 6 | — (derived) | ✅ |
| Binary flags | 5 | — (derived) | ✅ |
| Target-encoded delay rates | 5 | — (derived) | ✅ |
| One-hot encoded (airline + time block) | 33 | — (derived) | ✅ |
| Calendar integers (passthrough) | 4 | ✅ | ✅ |
| **Total model features** | | | **59** |
