# Flight Arrival Delay Prediction (ArrDel15)

Predictive Analytics Coursework Project — Dilara Cigerli

---

## Project Overview

This project builds a binary classifier to predict whether a US domestic flight will arrive **15 or more minutes late** (`ArrDel15 = 1`) using only pre-departure schedule information. The model uses the BTS Reporting Carrier On-Time Performance dataset (January–July 2025, ~4M flights) and is designed with strict leakage prevention — no post-departure actuals are used as features.

The best model is a **Random Forest** (ROC-AUC 0.670 on June 2025 test set, F1 0.488), selected from five candidates including Logistic Regression, LightGBM, and MLP. The project follows responsible AI-assisted development principles: all agent decisions are logged, every pipeline step is audited for leakage, and a confirmed diagnostic bug found during post-hoc review is documented.

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd AviationPrediction-1
```

### 2. Install dependencies

Tested with **Python 3.10–3.12**. From the project root:

```bash
pip install -r requirements.txt
```

Optional: use the provided conda environment (Python 3.10):

```bash
conda env create -f environment.yml
conda activate aviation
```

### 3. Download data

Raw CSV files are not included in this repository (too large for GitHub). Follow the instructions in [data/README.md](data/README.md) to download the BTS dataset and place the files in `data/raw/`.

### 4. Run the pipeline

**Option A — Notebooks (recommended for inspection):** Run in order 01 → 05 (see table below).

**Option B — Scripts from command line:** From project root, run `./run_pipeline.sh` (runs `src/01_...` through `src/05_...`). Requires data in `data/raw/` first.

| # | Notebook | Output |
|---|----------|--------|
| 01 | `notebooks/01_data_loading_and_quality.ipynb` | `data/processed/model_dataset.parquet` |
| 02 | `notebooks/02_eda.ipynb` | `reports/01–14_*.png`, `reports/eda_summary.md` |
| 03 | `notebooks/03_data_preparation.ipynb` | `data/processed/{train,val,test}_{X,y}.parquet`, `preprocessor.joblib` |
| 04 | `notebooks/04_modelling.ipynb` | `data/processed/best_model.joblib`, `reports/model_comparison.csv` |
| 05 | `notebooks/05_error_analysis.ipynb` | `reports/error_analysis_summary.md`, `reports/20–29_*.png` |

---

## Project Structure

```
AviationPrediction-1/
├── data/
│   ├── README.md              ← Data download instructions
│   ├── raw/                   ← BTS CSVs (not tracked by git)
│   └── processed/             ← Generated parquet files and model artifacts (not tracked)
├── notebooks/                 ← Jupyter notebooks (01–05, run in order)
├── src/                       ← Python scripts (mirror of notebooks; used by run_pipeline.sh)
├── reports/                   ← Figures, summaries, model card, final results
│   ├── model_card.md
│   ├── final_results.md
│   ├── error_analysis_summary.md
│   └── model_comparison.csv
├── agent_logs/
│   ├── interaction_log.md     ← Full record of every AI-assisted task
│   └── decision_register.md   ← All design decisions with rationale
├── tests/                      ← pytest (run: pytest tests/)
├── run_pipeline.sh             ← One-command pipeline (scripts 01→05)
├── environment.yml             ← Optional conda env (Python 3.10)
├── requirements.txt
└── README.md
```

---

## Key Results

| Model | Val ROC-AUC | Test ROC-AUC | Test F1 |
|-------|-------------|--------------|---------|
| Majority-class baseline | 0.500 | 0.500 | 0.000 |
| Logistic Regression | 0.623 | 0.654 | 0.158 |
| **Random Forest** ✓ | **0.626** | **0.670** | **0.488** |
| LightGBM | 0.587 | 0.628 | 0.465 |
| MLP | 0.565 | 0.536 | 0.172 |
| LightGBM-Tuned | 0.602 | 0.652 | 0.487 |

Test set: June 2025 (599,209 flights, 28.3% delay rate). Full metrics and interpretation in [reports/final_results.md](reports/final_results.md). Model card in [reports/model_card.md](reports/model_card.md).

---

## Assumptions and limitations

- **Prediction point:** The model uses only information available at **scheduled departure time** (airline, route, scheduled times, calendar). No post-departure actuals (e.g. departure delay, weather, ATC) are used.
- **Data scope:** US domestic BTS-reporting carriers only; one year (2025). Not validated for international, cargo, or other years.
- **Single temporal split:** Train = Jan–Mar, Val = Apr, Test = Jun (May excluded). No cross-validation; model selection and threshold are based on the single validation month.
- **Threshold:** The operating threshold (0.445 for RF) was chosen on the validation set (Youden’s J). Due to distribution shift (test delay rate 28.3% vs val 19.7%), this threshold is not necessarily optimal for June or for production; re-tuning on deployment-period data is recommended.
- **No weather or real-time data:** Delay causes (weather, NOTAMs, etc.) are not in the feature set; the model captures only historical correlations with schedule and route.
- **Reproducibility:** Requires running notebooks 01 → 05 in order (or `./run_pipeline.sh` from project root). Raw and processed data are not in the repo; see [data/README.md](data/README.md).

---

## AI Usage Policy

This project was developed with AI assistance under a documented protocol:

- **Interaction log** (`agent_logs/interaction_log.md`): every AI-assisted task is recorded with a description of what was produced.
- **Decision register** (`agent_logs/decision_register.md`): all design decisions (feature choices, model hyperparameters, encoding strategies, evaluation metrics) are logged with rationale.
- **Human verification** columns in both logs are left for the student to fill in, recording what was independently checked and whether each output was accepted, modified, or rejected.
- A **confirmed agent-made bug** was found during post-hoc audit (notebook 03, unseen route count diagnostic) and is documented in `reports/error_analysis_summary.md`.

---

## Project Rules

### 1. Logging is mandatory
After every meaningful AI-assisted action: a new entry is appended to `agent_logs/interaction_log.md`, files changed are listed, and human verification is completed separately.

### 2. Decision register
AI suggestions requiring judgement (features, models, metrics, validation strategy) are proposed but not automatically accepted. All decisions are recorded in `agent_logs/decision_register.md`.

### 3. No silent changes
All AI-generated changes include an explanation of what changed and why.

### 4. Reproducibility
All results are reproducible through documented data access, version-controlled code, logged AI interactions, and fixed random seeds (`SEED = 42`). Run `pytest tests/` (after generating `data/processed/` artifacts) to verify pipeline outputs.
