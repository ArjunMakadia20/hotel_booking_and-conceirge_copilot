from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Return a summary of the DataFrame for EDA reporting."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "summary_statistics": df.describe(include="all", datetime_is_numeric=True).to_dict(),
    }


def clean_hotel_booking_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the hotel booking dataset with standard preprocessing steps."""
    df = df.copy()

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Impute sparse numeric columns
    if "children" in df.columns:
        df["children"] = df["children"].fillna(0).astype(int)

    if "country" in df.columns:
        df["country"] = df["country"].fillna("UNKNOWN")

    if "agent" in df.columns:
        df["agent"] = df["agent"].fillna("UNKNOWN")

    if "company" in df.columns:
        df["company"] = df["company"].fillna("UNKNOWN")

    # Convert booking date fields if present
    if "reservation_status_date" in df.columns:
        df["reservation_status_date"] = pd.to_datetime(
            df["reservation_status_date"], errors="coerce"
        )

    # Replace negative or impossible values if they exist
    if "adr" in df.columns:
        df.loc[df["adr"] < 0, "adr"] = np.nan

    # Drop rows with missing target values
    if "adr" in df.columns:
        df = df.dropna(subset=["adr"])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features suitable for regression modeling."""
    df = df.copy()

    if "reservation_status_date" in df.columns:
        df["reservation_month"] = df["reservation_status_date"].dt.month
        df["reservation_day_of_week"] = df["reservation_status_date"].dt.dayofweek

    if "arrival_date_year" in df.columns and "arrival_date_month" in df.columns:
        month_map = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12,
        }
        df["arrival_month_num"] = df["arrival_date_month"].map(month_map)
        df["arrival_quarter"] = ((df["arrival_month_num"] - 1) // 3) + 1

    # Compute total guests and total nights as business features
    if "adults" in df.columns and "children" in df.columns and "babies" in df.columns:
        df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    if "stays_in_weekend_nights" in df.columns and "stays_in_week_nights" in df.columns:
        df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

    return df


def prepare_features(df: pd.DataFrame, target_column: str = "adr") -> tuple[pd.DataFrame, pd.Series]:
    """Prepare X and y for regression by encoding categorical fields and selecting features."""
    df = df.copy()
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    y = df[target_column].astype(float)
    X = df.drop(columns=[target_column])

    # Remove columns that are identifiers or redundant for regression
    drop_columns = [
        "reservation_status_date",
        "arrival_date_month",
        "arrival_date_year",
        "arrival_date_week_number",
        "arrival_date_day_of_month",
    ]
    X = X.drop(columns=[col for col in drop_columns if col in X.columns])

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Ensure no infinite values remain
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split feature matrix and target vector into training and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
