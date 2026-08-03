# SaaS Churn Early-Warning System

Predict which customer accounts are at risk of churning **within the next 30 days**, early enough for a Customer Success (CS) team to actually intervene — not a post-mortem model that tells you why someone already left.

[![CI](https://github.com/YOUR_USERNAME/churn-prediction-saas/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/churn-prediction-saas/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## The problem this solves

Most SaaS churn happens quietly. An account doesn't announce it's leaving — engagement just fades: fewer logins, narrower feature use, a support ticket that goes unresolved, a champion who's gone quiet. By the time someone cancels or doesn't renew, the relationship was effectively over weeks earlier.

This project builds a pipeline that:
1. Turns raw product-usage telemetry into rolling engagement signals (trend, momentum, recency).
2. Trains a classifier to predict `P(account churns in next 30 days)` from those signals.
3. Outputs a **ranked, explained risk list** that a CS team can act on directly — not just a probability, but *why* the account is flagged and *what to do about it*.

## Why this isn't a toy classification exercise

A few decisions were made specifically because this is meant to be operationally useful, not just accurate on paper:

- **Sliding snapshots, not one row per account.** The model is trained on many (account, as-of-date) snapshots per account, each with a forward-looking label. This lets it learn general early-warning *patterns* rather than memorizing which specific accounts happened to churn, and mirrors how you'd actually run it in production — rescoring every account on a recurring cadence.
- **Time-based validation split**, not a random split. Random splits let information leak across time for the same account and overstate performance. Here, the model is trained on the past and validated on a strictly later time window, the same way it will be used in reality.
- **PR-AUC and a gains table, not accuracy.** With a healthy churn base rate (~5-10%/month is typical), a model that predicts "never churns" is >90% accurate and useless. What actually matters to a CS team: *"If we can only work our top 200 accounts this week, how many real churners do we catch?"* The gains table in `reports/training_metrics.json` answers exactly that.
- **Explanations, not just scores.** `predict.py` attaches plain-language "risk drivers" (e.g. *"No login in 14+ days"*, *"Support ticket volume trending up"*) and a starting-point recommended action to every flagged account, so a CSM doesn't need to interpret raw feature values or model internals to act on it.

## Architecture

```
Raw usage telemetry ──▶ feature_engineering.py ──▶ labeled snapshot table
   (daily events)         (rolling windows,           (one row per account
                            trend/momentum,             per as-of-date, with
                            recency features)           30-day forward label)
                                                              │
                                                              ▼
                                                       train_model.py
                                                  (time-split GBRT classifier,
                                                   class-imbalance handling,
                                                   PR-AUC / gains evaluation)
                                                              │
                                                              ▼
                                                        predict.py
                                             (scores latest snapshot per account,
                                              risk tier + explanations + actions)
                                                              │
                                                              ▼
                                          reports/churn_risk_report.csv
                                          ──▶ CS team's weekly action list
```

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction-saas.git
cd churn-prediction-saas
pip install -r requirements.txt

# Run the entire pipeline: generate data -> features -> train -> score
./run_pipeline.sh
```

That's it. Open `reports/churn_risk_report.csv` — that's the file you'd hand to a CS team.

Run steps individually if you want to inspect intermediate output:

```bash
python3 data/generate_synthetic_data.py   # or point at your own extract, see schema below
python3 src/feature_engineering.py
python3 src/train_model.py
python3 src/predict.py
```

## Using your own data instead of the synthetic generator

The synthetic generator exists purely so this repo runs out of the box. To plug in real data, produce two CSVs matching this schema and skip straight to `feature_engineering.py`:

**`data/accounts.csv`** — one row per account
| column | type | notes |
|---|---|---|
| `account_id` | string | unique account identifier |
| `plan_tier` | string | e.g. Starter / Growth / Scale / Enterprise |
| `industry` | string | optional, for segmentation |
| `seats` | int | licensed seats |
| `contract_value_monthly` | float | MRR attributable to the account |
| `signup_date` | date | |
| `churned` | bool | ground truth — did this account eventually cancel? |
| `churn_date` | date or null | required for churned accounts, null otherwise |

**`data/daily_usage.csv`** — one row per account per day
| column | type | notes |
|---|---|---|
| `account_id` | string | |
| `date` | date | |
| `logins` | int | |
| `active_users` | int | distinct seats active that day |
| `distinct_features_used` | int | breadth of product usage |
| `api_calls` | int | if applicable |
| `avg_session_minutes` | float | |
| `support_tickets` | int | tickets opened that day |
| `nps_score` | float or null | sparse is fine — most days will be null |

Swap in whatever telemetry your product actually emits (Amplitude, Mixpanel, Segment, a data warehouse table) as long as it maps to something like this shape. Add or remove columns and update `get_feature_columns()` in `src/feature_engineering.py` to match.

## What's actually in the risk report

| column | meaning |
|---|---|
| `risk_score` | Model's estimated probability (0-1) the account churns in the next 30 days |
| `risk_tier` | High / Medium / Low, derived from the score |
| `top_risk_drivers` | Plain-language reasons the account is flagged |
| `recommended_action` | Suggested first move for the CSM (a starting playbook, not a mandate) |
| `days_since_last_login`, `login_momentum_7_30`, etc. | Supporting signals, for anyone who wants the underlying numbers |

Sort by `risk_score` descending, work down the list according to your team's outreach capacity, and use the gains table (`reports/training_metrics.json`) to decide how far down that list is worth going.

## Model performance (on the bundled synthetic data)

Numbers from the included synthetic dataset — expect these to look different, and to need re-tuning, on real usage data with noisier signal:

- **ROC-AUC / PR-AUC**: printed at the end of `train_model.py` and saved to `reports/training_metrics.json`
- **Gains table**: shows, for each decile of accounts ranked by risk score, what fraction of true churners you'd catch and at what precision — this is the number to bring to a conversation about CS staffing/capacity.

Note: the synthetic generator produces cleaner separation between "healthy" and "decaying" usage patterns than real customers usually show, so treat the bundled metrics as a ceiling, not a benchmark to expect on day one with live data.

## Project structure

```
churn-prediction-saas/
├── data/
│   └── generate_synthetic_data.py   # swap for your own extract
├── src/
│   ├── feature_engineering.py       # rolling windows, trend/momentum, labels
│   ├── train_model.py               # time-split training + evaluation
│   └── predict.py                   # scoring + explanations + CS report
├── tests/
│   └── test_pipeline.py             # end-to-end smoke test
├── models/                          # trained model artifact (gitignored)
├── reports/                         # metrics + churn risk report (gitignored)
├── .github/workflows/ci.yml         # runs tests on every push/PR
├── run_pipeline.sh                  # one-command full run
├── requirements.txt
└── README.md
```

## Extending this

- **Swap the model**: `train_model.py` isolates the model in `build_pipeline()` — drop in XGBoost, LightGBM, or a survival-analysis model (e.g. Cox proportional hazards, if you want time-to-churn rather than a binary window) without touching the rest of the pipeline.
- **Automate scoring**: wire `run_pipeline.sh` (or just `feature_engineering.py` + `predict.py`, reusing the already-trained model) into a scheduled job — cron, Airflow, GitHub Actions on a schedule — to regenerate `churn_risk_report.csv` daily or weekly, and push it to wherever your CS team lives (Slack digest, Salesforce field update, a dashboard).
- **Feedback loop**: log which accounts CS actually contacted and the outcome, and feed that back in — over time this tells you whether the interventions are working and lets you retrain on outcomes, not just usage.
- **Calibration**: re-tune the decision threshold in `train_model.py` (currently targets ~50% recall) against your CS team's actual weekly outreach capacity rather than a fixed statistical target.

## License

MIT — see [LICENSE](LICENSE).
