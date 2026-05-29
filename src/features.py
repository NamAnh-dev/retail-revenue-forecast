import pandas as pd
import numpy as np


BRAZIL_HOLIDAYS = pd.to_datetime([
    # 2016
    "2016-01-01", "2016-04-21", "2016-05-01", "2016-09-07",
    "2016-10-12", "2016-11-02", "2016-11-15", "2016-12-25",
    "2016-11-25",  # Black Friday
    # 2017
    "2017-01-01", "2017-04-21", "2017-05-01", "2017-09-07",
    "2017-10-12", "2017-11-02", "2017-11-15", "2017-12-25",
    "2017-11-24",  # Black Friday
    # 2018
    "2018-01-01", "2018-04-21", "2018-05-01", "2018-09-07",
    "2018-10-12", "2018-11-02", "2018-11-15", "2018-12-25",
    "2018-11-23",  # Black Friday
])


def add_lag_features(df: pd.DataFrame, lags: list[int] = [7, 14, 30]) -> pd.DataFrame:

    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["revenue"].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] = [7, 14, 30]
) -> pd.DataFrame:
    df = df.copy()
    shifted = df["revenue"].shift(1)
    for w in windows:
        df[f"rolling_mean_{w}"] = shifted.rolling(w).mean()
        df[f"rolling_std_{w}"]  = shifted.rolling(w).std()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"]             = df["date"].dt.month
    df["quarter"]           = df["date"].dt.quarter
    df["dayofweek"]         = df["date"].dt.dayofweek          
    df["day"]               = df["date"].dt.day
    df["is_weekend"]        = (df["dayofweek"] >= 5).astype(int)
    df["days_to_month_end"] = df["date"].dt.days_in_month - df["date"].dt.day
    df["week_of_year"]      = df["date"].dt.isocalendar().week.astype(int)

    # Cyclical encoding cho month & dayofweek
    df["month_sin"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]      = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofweek_sin"]  = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"]  = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["is_holiday"]      = df["date"].isin(BRAZIL_HOLIDAYS).astype(int)
    df["is_pre_holiday"]  = df["date"].isin(BRAZIL_HOLIDAYS - pd.Timedelta(days=1)).astype(int)
    df["is_post_holiday"] = df["date"].isin(BRAZIL_HOLIDAYS + pd.Timedelta(days=1)).astype(int)
    return df


def build_feature_matrix(daily: pd.DataFrame) -> pd.DataFrame:

    daily = daily.sort_values("date").reset_index(drop=True)
    daily = add_lag_features(daily)
    daily = add_rolling_features(daily)
    daily = add_calendar_features(daily)
    daily = add_holiday_features(daily)
    daily = daily.dropna().reset_index(drop=True)

    print(f"Feature matrix: {daily.shape[0]} rows × {daily.shape[1]} cols")
    print(f"Features: {[c for c in daily.columns if c not in ['date', 'revenue']]}")
    return daily


FEATURE_COLS = [
    "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_std_7",
    "rolling_mean_14",
    "rolling_mean_30", "rolling_std_30",
    "month_sin", "month_cos",
    "dayofweek_sin", "dayofweek_cos",
    "quarter",
    "is_weekend",
    "is_holiday", "is_pre_holiday", "is_post_holiday",
    "days_to_month_end",
    "week_of_year",
]
TARGET_COL = "revenue"


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:

    snapshot_date = df["date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        recency   = ("date", lambda x: (snapshot_date - x.max()).days),
        frequency = ("order_id", "nunique"),
        monetary  = ("payment_value", "sum"),
    ).reset_index()

    # Score: Recency ngược (thấp = tốt → score cao)
    rfm["R_score"] = pd.qcut(rfm["recency"], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(
        rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]
    ).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"], 4, labels=[1, 2, 3, 4]).astype(int)

    rfm["RFM_total"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    rfm["segment"] = rfm.apply(_assign_segment, axis=1)

    print(f"\n=== RFM Summary ===")
    print(rfm["segment"].value_counts().to_string())
    return rfm


def _assign_segment(row: pd.Series) -> str:
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    if r >= 3 and f >= 3 and m >= 3:
        return "Champions"
    elif r >= 3 and f >= 2:
        return "Loyal"
    elif r >= 3 and f == 1:
        return "Recent"
    elif r <= 2 and f >= 3:
        return "At Risk"
    elif r == 1 and f == 1:
        return "Lost"
    else:
        return "Needs Attention"


def compute_cohort(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["order_month"]  = df["date"].dt.to_period("M")
    df["cohort_month"] = df.groupby("customer_id")["date"].transform("min").dt.to_period("M")
    df["cohort_index"] = (df["order_month"] - df["cohort_month"]).apply(lambda x: x.n)

    cohort_data = (
        df.groupby(["cohort_month", "cohort_index"])["customer_id"]
        .nunique()
        .reset_index()
    )

    cohort_pivot = cohort_data.pivot(
        index="cohort_month", columns="cohort_index", values="customer_id"
    )


    cohort_size   = cohort_pivot.iloc[:, 0]
    retention     = cohort_pivot.divide(cohort_size, axis=0) * 100

    return retention
