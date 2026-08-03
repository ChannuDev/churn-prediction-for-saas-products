#!/usr/bin/env bash
# Runs the full pipeline end-to-end: generate data -> build features -> train -> score.
# Usage: ./run_pipeline.sh
set -euo pipefail

echo "==> [1/4] Generating synthetic usage data (replace with your own extract in production)"
python3 data/generate_synthetic_data.py

echo -e "\n==> [2/4] Building feature snapshots + labels"
python3 src/feature_engineering.py

echo -e "\n==> [3/4] Training the churn model (time-based validation split)"
python3 src/train_model.py

echo -e "\n==> [4/4] Scoring active accounts -> reports/churn_risk_report.csv"
python3 src/predict.py

echo -e "\nDone. Open reports/churn_risk_report.csv for the Customer Success team's action list."
