"""
generate_synthetic_data.py
---------------------------
Generates a realistic synthetic dataset simulating daily product-usage
telemetry for a B2B SaaS product, plus an account roster with contract
and firmographic details.

This exists so the pipeline is runnable end-to-end out of the box.
Swap this out for your own data warehouse extract (e.g. from Snowflake,
Amplitude, Mixpanel, Segment) by matching the output schema described
in the README.

Output:
    data/accounts.csv          - one row per account (static attributes)
    data/daily_usage.csv       - one row per account per day (usage telemetry)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
N_ACCOUNTS = 1200
N_DAYS = 270  # ~9 months of history
START_DATE = datetime(2025, 11, 1)

PLAN_TIERS = ["Starter", "Growth", "Scale", "Enterprise"]
PLAN_WEIGHTS = [0.35, 0.35, 0.20, 0.10]
INDUSTRIES = ["Retail", "FinTech", "Healthcare", "EdTech", "Logistics",
              "Media", "Manufacturing", "Professional Services"]


def _account_base_engagement(rng, plan_tier):
    """Higher tiers tend to have more seats and slightly higher baseline engagement."""
    tier_mult = {"Starter": 1.0, "Growth": 1.6, "Scale": 2.4, "Enterprise": 4.0}[plan_tier]
    seats = max(1, int(rng.lognormal(mean=1.3, sigma=0.7) * tier_mult))
    base_daily_logins_per_seat = rng.uniform(0.35, 0.85)
    base_feature_breadth = rng.uniform(3, 12)  # distinct features touched per week
    base_api_calls = rng.uniform(20, 400) * tier_mult
    contract_value = round(seats * rng.uniform(30, 120) * tier_mult, 2)
    return seats, base_daily_logins_per_seat, base_feature_breadth, base_api_calls, contract_value


def generate():
    rng = np.random.default_rng(RNG_SEED)
    accounts = []
    daily_rows = []

    for acc_idx in range(N_ACCOUNTS):
        account_id = f"ACC-{acc_idx:05d}"
        plan_tier = rng.choice(PLAN_TIERS, p=PLAN_WEIGHTS)
        industry = rng.choice(INDUSTRIES)
        signup_offset = int(rng.integers(0, N_DAYS - 60))  # ensure some tenure history
        signup_date = START_DATE + timedelta(days=signup_offset)

        seats, base_logins, base_breadth, base_api, contract_value = _account_base_engagement(rng, plan_tier)

        # --- Decide the account's "storyline" ---
        # healthy: stable/growing engagement, does not churn
        # slow_decay: engagement quietly erodes over 60-120 days, then cancels
        # sharp_drop: sudden engagement collapse (e.g. champion left company), cancels within 30-45 days
        # seasonal: naturally variable usage, does not churn
        # recovered: dips but re-engages after an intervention-like bump, does not churn
        storyline = rng.choice(
            ["healthy", "slow_decay", "sharp_drop", "seasonal", "recovered"],
            p=[0.45, 0.20, 0.10, 0.15, 0.10]
        )

        churned = storyline in ("slow_decay", "sharp_drop")
        # churn_date only defined if the account actually cancels within the observation window
        churn_date = None

        if churned:
            min_life = 90  # accounts need some history before churning
            max_possible = N_DAYS - signup_offset - 5
            if max_possible > min_life:
                life_days = int(rng.integers(min_life, max_possible))
                churn_date = signup_date + timedelta(days=life_days)
            else:
                churned = False  # not enough runway in the window, treat as active/censored

        accounts.append({
            "account_id": account_id,
            "plan_tier": plan_tier,
            "industry": industry,
            "seats": seats,
            "contract_value_monthly": contract_value,
            "signup_date": signup_date.date().isoformat(),
            "churned": bool(churned),
            "churn_date": churn_date.date().isoformat() if churn_date else None,
            "storyline": storyline,  # kept for validation/debugging; drop before modeling
        })

        # --- Generate the daily usage series for this account ---
        obs_start = signup_offset
        obs_end = (churn_date - START_DATE).days if churn_date else N_DAYS
        obs_end = min(obs_end, N_DAYS)

        # noise processes
        weekly_phase = rng.uniform(0, 2 * np.pi)

        for day_offset in range(obs_start, obs_end):
            date = START_DATE + timedelta(days=day_offset)
            t = day_offset - obs_start  # days since signup
            life_span = max(obs_end - obs_start, 1)
            progress = t / life_span  # 0 -> 1 across the account's life in-window

            # weekday seasonality: lower usage on weekends
            dow = date.weekday()
            weekend_damp = 0.35 if dow >= 5 else 1.0

            # storyline-specific engagement multiplier over time
            if storyline == "healthy":
                trend_mult = 1.0 + 0.15 * progress  # gentle growth
                noise_scale = 0.15
            elif storyline == "seasonal":
                trend_mult = 1.0 + 0.25 * np.sin(2 * np.pi * t / 30 + weekly_phase)
                noise_scale = 0.20
            elif storyline == "recovered":
                # dips around 40-60% of life then recovers
                dip_center = 0.5 * life_span
                dip = -0.5 * np.exp(-((t - dip_center) ** 2) / (2 * (life_span * 0.08) ** 2))
                trend_mult = 1.0 + dip + (0.2 if progress > 0.7 else 0)
                noise_scale = 0.18
            elif storyline == "slow_decay":
                # engagement quietly declines starting ~40% through life, accelerating near the end
                decay_start = 0.35
                if progress < decay_start:
                    trend_mult = 1.0
                else:
                    decay_progress = (progress - decay_start) / (1 - decay_start)
                    trend_mult = 1.0 - 0.9 * (decay_progress ** 1.4)
                noise_scale = 0.12
            elif storyline == "sharp_drop":
                drop_point = 0.7
                if progress < drop_point:
                    trend_mult = 1.0 + 0.1 * progress
                else:
                    trend_mult = max(0.05, 1.0 - 3.5 * (progress - drop_point))
                noise_scale = 0.10
            else:
                trend_mult = 1.0
                noise_scale = 0.15

            trend_mult = max(trend_mult, 0.02)
            noise = rng.normal(1.0, noise_scale)
            activity_mult = max(trend_mult * noise * weekend_damp, 0.0)

            logins = rng.poisson(max(base_logins * seats * activity_mult, 0.01))
            active_users = min(seats, rng.poisson(max(seats * 0.5 * activity_mult, 0.01)) + (1 if logins > 0 else 0))
            features_used = max(0, min(30, rng.poisson(max(base_breadth * activity_mult, 0.01))))
            api_calls = max(0, rng.poisson(max(base_api * activity_mult, 0.01)))
            session_minutes = max(0.0, rng.normal(18 * activity_mult, 4))

            # support tickets tend to spike a bit before sharp drops and during decay (frustration signal)
            ticket_base = 0.03
            if storyline == "sharp_drop" and 0.55 < progress < 0.75:
                ticket_base = 0.15
            if storyline == "slow_decay" and progress > 0.5:
                ticket_base = 0.08
            support_tickets = rng.poisson(ticket_base)

            # NPS / in-app survey response, sparse (only ~2% of days have one)
            nps_score = np.nan
            if rng.random() < 0.02:
                base_nps = 8.5 * activity_mult if activity_mult > 0 else 2
                nps_score = int(np.clip(rng.normal(base_nps, 1.5), 0, 10))

            daily_rows.append({
                "account_id": account_id,
                "date": date.date().isoformat(),
                "logins": int(logins),
                "active_users": int(active_users),
                "distinct_features_used": int(features_used),
                "api_calls": int(api_calls),
                "avg_session_minutes": round(session_minutes, 2),
                "support_tickets": int(support_tickets),
                "nps_score": nps_score,
            })

    accounts_df = pd.DataFrame(accounts)
    daily_df = pd.DataFrame(daily_rows)

    accounts_df.to_csv("data/accounts.csv", index=False)
    daily_df.to_csv("data/daily_usage.csv", index=False)

    print(f"Generated {len(accounts_df)} accounts and {len(daily_df):,} daily usage rows.")
    print(f"Churn rate in dataset: {accounts_df['churned'].mean():.1%}")
    print("Storyline breakdown:")
    print(accounts_df["storyline"].value_counts())


if __name__ == "__main__":
    generate()
