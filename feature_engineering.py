"""
feature_engineering.py
------------------------
Turns raw daily usage telemetry into a modeling table of one row per
(account, as_of_date) snapshot, with:
  - rolling usage aggregates over multiple windows (7 / 14 / 30 / 60 days)
  - trend / momentum features (is engagement accelerating or decaying?)
  - recency features (days since last login, since last feature use, etc.)
  - a forward-looking label: will this account churn within the next
    LABEL_HORIZON_DAYS days of the as_of_date?

The "sliding snapshot" approach (rather than one row per account) is what
lets the model learn general early-warning patterns instead of memorizing
which specific accounts churned, and lets you re-score every account on
every day in production.
"""

import numpy as np
import pandas as pd

LABEL_HORIZON_DAYS = 30
ROLLING_WINDOWS = [7, 14, 30, 60]
MIN_HISTORY_DAYS = 30          # need at least this much history to form a snapshot
SNAPSHOT_STRIDE_DAYS = 7       # take one snapshot every N days per account (keeps dataset size sane)


def _load_raw(accounts_path="data/accounts.csv", usage_path="data/daily_usage.csv"):
    accounts = pd.read_csv(accounts_path, parse_dates=["signup_date", "churn_date"])
    usage = pd.read_csv(usage_path, parse_dates=["date"])
    return accounts, usage


def _rolling_block(g: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling sums/means for one window size, indexed by date, for a single account."""
    r = g.rolling(window=f"{window}D", on="date")
    out = pd.DataFrame({
        f"logins_sum_{window}d": r["logins"].sum(),
        f"active_users_mean_{window}d": r["active_users"].mean(),
        f"features_used_mean_{window}d": r["distinct_features_used"].mean(),
        f"api_calls_sum_{window}d": r["api_calls"].sum(),
        f"session_minutes_mean_{window}d": r["avg_session_minutes"].mean(),
        f"support_tickets_sum_{window}d": r["support_tickets"].sum(),
        f"active_days_{window}d": r["logins"].apply(lambda x: (x > 0).sum()),
    })
    return out


def build_features(accounts: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    usage = usage.sort_values(["account_id", "date"]).copy()
    usage["logins"] = usage["logins"].fillna(0)

    all_snapshots = []

    for account_id, g in usage.groupby("account_id", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        acct_row = accounts.loc[accounts.account_id == account_id].iloc[0]
        signup_date = acct_row["signup_date"]
        churn_date = acct_row["churn_date"] if pd.notna(acct_row["churn_date"]) else None

        # Compute rolling windows once per account (efficient), then sample snapshot dates
        rolled = [g[["date"]].copy()]
        for w in ROLLING_WINDOWS:
            rolled.append(_rolling_block(g, w))
        feat_df = pd.concat(rolled, axis=1)
        feat_df = feat_df.loc[:, ~feat_df.columns.duplicated()]

        # days since last login / feature use, computed causally (no lookahead)
        last_login_date = None
        last_feature_date = None
        days_since_login_list = []
        days_since_feature_list = []
        for _, row in g.iterrows():
            if row["logins"] > 0:
                last_login_date = row["date"]
            if row["distinct_features_used"] > 0:
                last_feature_date = row["date"]
            days_since_login_list.append((row["date"] - last_login_date).days if last_login_date is not None else np.nan)
            days_since_feature_list.append((row["date"] - last_feature_date).days if last_feature_date is not None else np.nan)
        feat_df["days_since_last_login"] = days_since_login_list
        feat_df["days_since_last_feature_use"] = days_since_feature_list

        # tenure at each date
        feat_df["tenure_days"] = (feat_df["date"] - signup_date).dt.days

        # rolling NPS (sparse; forward-fill within account, causal)
        nps_ffill = g[["date", "nps_score"]].copy()
        nps_ffill["nps_score"] = nps_ffill["nps_score"].ffill()
        feat_df["latest_nps"] = nps_ffill["nps_score"].values

        # sample snapshot dates: every SNAPSHOT_STRIDE_DAYS, once enough history exists,
        # and stop once the account is within LABEL_HORIZON_DAYS of its last observed day
        # (so we always have a fully-observed forward window for labeling) OR it churned.
        eligible = feat_df[feat_df["tenure_days"] >= MIN_HISTORY_DAYS].copy()
        if eligible.empty:
            continue
        eligible = eligible.iloc[::SNAPSHOT_STRIDE_DAYS]

        eligible["account_id"] = account_id
        all_snapshots.append(eligible)

    snap = pd.concat(all_snapshots, ignore_index=True)

    # --- Momentum / trend features: compare recent window to a longer baseline ---
    snap["login_momentum_7_30"] = _safe_ratio(snap["logins_sum_7d"] / 7, snap["logins_sum_30d"] / 30)
    snap["api_momentum_7_30"] = _safe_ratio(snap["api_calls_sum_7d"] / 7, snap["api_calls_sum_30d"] / 30)
    snap["active_days_ratio_7_30"] = _safe_ratio(snap["active_days_7d"] / 7, snap["active_days_30d"] / 30)
    snap["session_length_trend_14_60"] = _safe_ratio(snap["session_minutes_mean_14d"], snap["session_minutes_mean_60d"])
    snap["support_ticket_trend_14_60"] = _safe_ratio(snap["support_tickets_sum_14d"], snap["support_tickets_sum_60d"] + 1e-6)

    # --- Merge static account attributes ---
    static_cols = ["account_id", "plan_tier", "industry", "seats", "contract_value_monthly",
                   "signup_date", "churned", "churn_date"]
    snap = snap.merge(accounts[static_cols], on="account_id", how="left")

    # --- Forward-looking label: churns within LABEL_HORIZON_DAYS of this snapshot date ---
    def make_label(row):
        if pd.isna(row["churn_date"]):
            return 0
        days_to_churn = (row["churn_date"] - row["date"]).days
        return int(0 <= days_to_churn <= LABEL_HORIZON_DAYS)

    snap["will_churn_30d"] = snap.apply(make_label, axis=1)

    # Drop snapshots that occur AFTER an account already churned (no longer a live decision point)
    snap = snap[(snap["churn_date"].isna()) | (snap["date"] <= snap["churn_date"])]

    # Drop trailing snapshots for still-active accounts where we can't yet confirm the label
    # would be observable (i.e. within LABEL_HORIZON_DAYS of the end of the observation window).
    max_date_per_account = usage.groupby("account_id")["date"].max().rename("last_observed_date")
    snap = snap.merge(max_date_per_account, on="account_id", how="left")
    censored_mask = (snap["churn_date"].isna()) & \
                     ((snap["last_observed_date"] - snap["date"]).dt.days < LABEL_HORIZON_DAYS)
    snap = snap[~censored_mask]

    snap = snap.drop(columns=["last_observed_date"])
    return snap.reset_index(drop=True)


def _safe_ratio(numerator, denominator):
    denom = denominator.replace(0, np.nan)
    ratio = numerator / denom
    return ratio.fillna(1.0).replace([np.inf, -np.inf], 1.0)


def get_feature_columns():
    """Central place defining which columns feed the model (keeps train/predict in sync)."""
    numeric = []
    for w in ROLLING_WINDOWS:
        numeric += [
            f"logins_sum_{w}d", f"active_users_mean_{w}d", f"features_used_mean_{w}d",
            f"api_calls_sum_{w}d", f"session_minutes_mean_{w}d", f"support_tickets_sum_{w}d",
            f"active_days_{w}d",
        ]
    numeric += [
        "days_since_last_login", "days_since_last_feature_use", "tenure_days", "latest_nps",
        "login_momentum_7_30", "api_momentum_7_30", "active_days_ratio_7_30",
        "session_length_trend_14_60", "support_ticket_trend_14_60",
        "seats", "contract_value_monthly",
    ]
    categorical = ["plan_tier", "industry"]
    return numeric, categorical


if __name__ == "__main__":
    accounts, usage = _load_raw()
    features = build_features(accounts, usage)
    features.to_csv("data/model_snapshots.csv", index=False)
    print(f"Built {len(features):,} snapshot rows across {features['account_id'].nunique()} accounts.")
    print(f"Positive label rate (will_churn_30d): {features['will_churn_30d'].mean():.2%}")
