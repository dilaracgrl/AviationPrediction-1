#!/usr/bin/env bash
# Run the full AviationPrediction pipeline (01 → 05) using src/ scripts.
# From project root: ./run_pipeline.sh
# Requires: data/raw/*.csv (see data/README.md), pip install -r requirements.txt
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "=== 01 Data loading & quality ==="
(cd src && python 01_data_loading_and_quality.py) || exit 1

echo "=== 02 EDA ==="
(cd src && python 02_eda.py) || exit 1

echo "=== 03 Data preparation ==="
python src/03_data_preparation.py || exit 1

echo "=== 04 Modelling ==="
python src/04_modelling.py || exit 1

echo "=== 05 Error analysis ==="
python src/05_error_analysis.py || exit 1

echo "=== Pipeline complete ==="
