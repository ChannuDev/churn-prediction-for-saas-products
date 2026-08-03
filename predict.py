"""
predict.py
------------
Scores every active account as of the most recent snapshot date and produces
an actionable CSV report for the Customer Success team:

    reports/churn_risk_report.csv

Each row includes:
    - risk_score        (0-1 probability of churning in the next 30 days)
    - risk_tier         (High / Medium / Low, based on score thresholds)
    - top_risk_drivers  (plain-language explanation of what's driving the score)
    - recommended_action (a starting-point playbook suggestion per tier/driver)

This is deliberately NOT a black box: the whole point is that a CSM opening
this file on Monday morning can immediately understand *why* an account is
flagged and what to do about it, without needing to interpret raw feature
values or SHAP plots.
"""

import joblib
import numpy as np
import pandas as pd

from feature_engineering import get_feature_columns

MODEL_PATH = "models/churn_model.joblib"
SNAPSHOTS_PATH = "data/model_snapshots.csv"
OUTPUT_PATH = "reports/churn_risk_report.csv"

# Human-readable descriptions and playbooks keyed to specific feature signals.
# Each entry: (feature_column, "worsening" check, label, recommended_action)
DRIVER_RULES = [
    ("days_since_last_login", lambda v: v is not None and v >= 10,
     "No login in {v:.0f}+ days",
     "Reach out directly to the primary champion; confirm they still have a live use case."),
    ("login_momentum_7_30", lambda v: v is not None and v < 0.6,
     "Login frequency down {pct:.0f}% vs their own 30-day average",
     "Check in with an async usage summary or a light-touch 'how's it going' email."),
    ("api_momentum_7_30", lambda v: v is not None and v < 0.5,
     "API/integration usage down {pct:.0f}% vs baseline",
     "Loop in a solutions engineer to check for integration or technical blockers."),
    ("active_days_ratio_7_30", lambda v: v is not None and v < 0.5,
     "Active days per week down {pct:.0f}% vs baseline",
     "Schedule a proactive check-in call before the next renewal conversation."),
    ("support_ticket_trend_14_60", lambda v: v is not None and v > 1.8,
     "Support ticket volume trending up",
     "Escalate to support lead; review recent tickets for unresolved friction or bugs."),
    ("features_used_mean_30d", lambda v: v is not None and v < 2,
     "Very narrow feature adoption (avg {v:.1f} features/day)",
     "Offer a tailored onboarding refresh or feature-adoption session."),
    ("latest_nps", lambda v: v is not None and not np.isnan(v) and v <= 6,
     "Low NPS/survey response ({v:.0f}/10)",
     "Personally follow up on their survey feedback; treat as a save-play priority."),
]


def _tier(score, threshold):
    if score >= max(threshold, 0.66):
        return "High"
    elif score >= threshold * 0.4:
        return "Medium"
    else:
        return "Low"


def _explain(row):
    drivers = []
    actions = []
    for col, check, label_tmpl, action in DRIVER_RULES:
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if check(v):
            pct = (1 - v) * 100 if v < 1 else 0
            try:
                drivers.append(label_tmpl.format(v=v, pct=pct))
            except (KeyError, ValueError):
                drivers.append(label_tmpl)
            actions.append(action)
    if not drivers:
        drivers = ["No single dominant signal — risk driven by a broad combination of factors"]
        actions = ["Review full usage trend in the dashboard before deciding on outreach."]
    return " | ".join(drivers[:3]), actions[0] if actions else "Monitor; no immediate action required."


def main():
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]
    threshold = bundle["threshold"]

    df = pd.read_csv(SNAPSHOTS_PATH, parse_dates=["date", "signup_date", "churn_date"])

    # Score using each account's MOST RECENT snapshot only (this is what you'd run in production
    # on a daily/weekly cadence against freshly computed features).
    latest = df.sort_values("date").groupby("account_id", as_index=False).tail(1).copy()

    # Exclude accounts that have already churned in the historical data (nothing to act on)
    latest = latest[latest["churned"] == False].copy()  # noqa: E712

    X = latest[numeric_cols + categorical_cols]
    scores = pipeline.predict_proba(X)[:, 1]
    latest["risk_score"] = scores
    latest["risk_tier"] = latest["risk_score"].apply(lambda s: _tier(s, threshold))

    explanations = latest.apply(_explain, axis=1, result_type="expand")
    latest["top_risk_drivers"] = explanations[0]
    latest["recommended_action"] = explanations[1]

    report_cols = [
        "account_id", "plan_tier", "industry", "seats", "contract_value_monthly",
        "tenure_days", "risk_score", "risk_tier", "top_risk_drivers", "recommended_action",
        "days_since_last_login", "login_momentum_7_30", "support_ticket_trend_14_60", "latest_nps",
        "date",
    ]
    report = latest[report_cols].rename(columns={"date": "as_of_date"})
    report = report.sort_values("risk_score", ascending=False)
    report["risk_score"] = report["risk_score"].round(3)

    report.to_csv(OUTPUT_PATH, index=False)

    tier_counts = report["risk_tier"].value_counts()
    print(f"Scored {len(report):,} active accounts as of their latest snapshot.")
    print(f"Risk tier breakdown:\n{tier_counts}")
    print(f"\nTop 10 highest-risk accounts:")
    print(report.head(10)[["account_id", "risk_score", "risk_tier", "top_risk_drivers"]].to_string(index=False))
    print(f"\nFull report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
