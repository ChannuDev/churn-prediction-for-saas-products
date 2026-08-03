"""
train_model.py
----------------
Trains the 30-day churn-risk classifier on the engineered snapshot table.

Key design choices:
- TIME-BASED split (not random) to simulate real deployment: train on the
  past, validate on the future. Random splits leak information because
  snapshots from the same account are correlated across time.
- GradientBoostingClassifier as the primary model (strong tabular baseline,
  no extra dependencies beyond scikit-learn). Swap in XGBoost/LightGBM by
  editing MODEL_FACTORY if you have them installed and want a speed/accuracy
  edge on larger datasets.
- Class weighting to address churn's natural imbalance rather than blind
  accuracy, since accuracy is a misleading metric when ~93% of rows are
  "not churning."
- Evaluation centers on PR-AUC, recall at fixed precision, and a lift/gains
  table, because that's what a CS team actually cares about: "if we can
  only work 100 accounts this week, how many true churners do we catch?"
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, brier_score_loss
)

from feature_engineering import get_feature_columns

MODEL_PATH = "models/churn_model.joblib"
METRICS_PATH = "reports/training_metrics.json"
VALIDATION_HOLDOUT_FRACTION = 0.2  # most recent 20% of snapshot dates held out


def time_based_split(df: pd.DataFrame):
    df = df.sort_values("date")
    cutoff = df["date"].quantile(1 - VALIDATION_HOLDOUT_FRACTION)
    train = df[df["date"] < cutoff]
    valid = df[df["date"] >= cutoff]
    return train, valid, cutoff


def build_pipeline(numeric_cols, categorical_cols, model_type="gbrt"):
    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])

    if model_type == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        clf = GradientBoostingClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    return Pipeline([("preprocess", preprocess), ("model", clf)])


def _sample_weights(y):
    # GradientBoostingClassifier has no native class_weight param, so we
    # emulate "balanced" weighting manually to counter the churn imbalance.
    pos_rate = y.mean()
    w_pos = 0.5 / max(pos_rate, 1e-6)
    w_neg = 0.5 / max(1 - pos_rate, 1e-6)
    return np.where(y == 1, w_pos, w_neg)


def gains_table(y_true, y_score, n_bins=10):
    """How many true churners are caught if CS works the top decile, top 2 deciles, etc."""
    df = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score", ascending=False).reset_index(drop=True)
    df["bin"] = pd.qcut(df.index, n_bins, labels=False, duplicates="drop")
    total_pos = df["y"].sum()
    rows = []
    cum_pos = 0
    cum_n = 0
    for b in sorted(df["bin"].unique()):
        chunk = df[df["bin"] == b]
        cum_pos += chunk["y"].sum()
        cum_n += len(chunk)
        rows.append({
            "top_pct_of_accounts": round((b + 1) / (df["bin"].max() + 1) * 100, 1),
            "accounts_flagged": int(cum_n),
            "true_churners_caught": int(cum_pos),
            "pct_of_all_churners_caught": round(cum_pos / total_pos * 100, 1) if total_pos else 0.0,
            "precision_in_flagged_set": round(cum_pos / cum_n, 3) if cum_n else 0.0,
        })
    return rows


def main():
    df = pd.read_csv("data/model_snapshots.csv", parse_dates=["date", "signup_date", "churn_date"])
    numeric_cols, categorical_cols = get_feature_columns()

    train, valid, cutoff = time_based_split(df)
    print(f"Train: {len(train):,} rows (through {train['date'].max().date()})")
    print(f"Valid: {len(valid):,} rows (from {valid['date'].min().date()} to {valid['date'].max().date()})")
    print(f"Time-based split cutoff: {cutoff.date()}")

    X_train, y_train = train[numeric_cols + categorical_cols], train["will_churn_30d"].values
    X_valid, y_valid = valid[numeric_cols + categorical_cols], valid["will_churn_30d"].values

    pipeline = build_pipeline(numeric_cols, categorical_cols, model_type="gbrt")
    sw = _sample_weights(y_train)
    pipeline.fit(X_train, y_train, model__sample_weight=sw)

    valid_scores = pipeline.predict_proba(X_valid)[:, 1]
    train_scores = pipeline.predict_proba(X_train)[:, 1]

    roc_auc = roc_auc_score(y_valid, valid_scores)
    pr_auc = average_precision_score(y_valid, valid_scores)
    brier = brier_score_loss(y_valid, valid_scores)

    # Pick an operating threshold that targets ~40% recall at the best precision achievable
    # (illustrative default; tune this against your CS team's actual outreach capacity).
    precisions, recalls, thresholds = precision_recall_curve(y_valid, valid_scores)
    target_recall = 0.5
    idx = np.argmin(np.abs(recalls[:-1] - target_recall))
    chosen_threshold = float(thresholds[idx]) if len(thresholds) else 0.5

    report = classification_report(
        y_valid, (valid_scores >= chosen_threshold).astype(int), output_dict=True, zero_division=0
    )

    gains = gains_table(y_valid, valid_scores)

    print("\n--- Validation performance (time-based holdout) ---")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC (average precision): {pr_auc:.3f}")
    print(f"Brier score (lower=better calibration): {brier:.4f}")
    print(f"Chosen decision threshold (~{target_recall:.0%} recall target): {chosen_threshold:.3f}")
    print(f"Precision at threshold: {report['1']['precision']:.3f}  Recall at threshold: {report['1']['recall']:.3f}")
    print("\n--- Gains table (top-N targeting for CS capacity planning) ---")
    for row in gains:
        print(row)

    joblib.dump({
        "pipeline": pipeline,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "threshold": chosen_threshold,
    }, MODEL_PATH)

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "chosen_threshold": chosen_threshold,
        "classification_report": report,
        "gains_table": gains,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "split_cutoff_date": str(cutoff.date()),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
