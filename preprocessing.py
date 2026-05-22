"""
Shared preprocessing utilities for the Census income project.

Expected files in the same directory by default:
- census-bureau.columns
- census-bureau.data

The functions in this file are used by both classification_model.py and
segmentation_model.py so the cleaning logic stays consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


NUMERIC_ZERO_COLS = [
    "wage per hour",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "num persons worked for employer",
    "weeks worked in year",
]

SKEWED_MONEY_COLS = [
    "wage per hour",
    "capital gains",
    "capital losses",
    "dividends from stocks",
]

EDUCATION_ORDER = {
    "Children": 0,
    "Less than 1st grade": 1,
    "1st 2nd 3rd or 4th grade": 2,
    "5th or 6th grade": 3,
    "7th and 8th grade": 4,
    "9th grade": 5,
    "10th grade": 6,
    "11th grade": 7,
    "12th grade no diploma": 8,
    "High school graduate": 9,
    "Some college but no degree": 10,
    "Associates degree-occup /vocational": 11,
    "Associates degree-academic program": 12,
    "Bachelors degree(BA AB BS)": 13,
    "Masters degree(MA MS MEng MEd MSW MBA)": 14,
    "Doctorate degree(PhD EdD)": 15,
    "Prof school degree (MD DDS DVM LLB JD)": 16,
}

LABEL_MAPPING = {
    "- 50000.": 0,
    "50000+.": 1,
}


def load_raw_data(
    data_path: str | Path = "census-bureau.data",
    columns_path: str | Path = "census-bureau.columns",
) -> pd.DataFrame:
    """Load the raw census data and assign column names."""
    data_path = Path(data_path)
    columns_path = Path(columns_path)

    columns = pd.read_csv(columns_path, header=None)
    data = pd.read_csv(data_path, header=None)
    data.columns = columns[0].tolist()
    return data


def clean_missing_values(df: pd.DataFrame, missing_threshold: float = 0.40) -> pd.DataFrame:
    """Strip string values, convert '?' to missing, drop high-missing columns, and fill remaining categorical missing values."""
    df_clean = df.copy()
    df_clean = df_clean.map(lambda x: x.strip() if isinstance(x, str) else x)
    df_clean = df_clean.replace("?", np.nan)

    missing_rate = df_clean.isna().mean()
    cols_to_drop = missing_rate[missing_rate > missing_threshold].index.tolist()
    df_clean = df_clean.drop(columns=cols_to_drop)

    categorical_cols = df_clean.select_dtypes(include=["object", "category"]).columns
    df_clean[categorical_cols] = df_clean[categorical_cols].fillna("missing")
    return df_clean


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add zero indicators, log transforms, age group, marital group, education ordinal, and mapped label."""
    df_model = df.copy()

    for col in NUMERIC_ZERO_COLS:
        if col in df_model.columns:
            df_model[f"{col}_is_nonzero"] = (df_model[col] != 0).astype(int)

    for col in SKEWED_MONEY_COLS:
        if col in df_model.columns:
            df_model[f"{col}_log"] = np.log1p(df_model[col])

    if "label" in df_model.columns and df_model["label"].dtype == "object":
        df_model["label"] = df_model["label"].map(LABEL_MAPPING)

    if "age" in df_model.columns:
        df_model["age_group"] = pd.cut(
            df_model["age"],
            bins=[-1, 29, 59, 90],
            labels=["0-29", "29-59", "60-90"],
        )

    if "marital stat" in df_model.columns:
        df_model["marital_group"] = df_model["marital stat"].apply(
            lambda x: "Divorced_or_Married"
            if x in ["Divorced", "Married-civilian spouse present"]
            else "Other"
        )

    if "education" in df_model.columns:
        df_model["education_ordinal"] = df_model["education"].map(EDUCATION_ORDER)

    return df_model


def prepare_model_dataframe(
    data_path: str | Path = "census-bureau.data",
    columns_path: str | Path = "census-bureau.columns",
) -> pd.DataFrame:
    """Load, clean, and feature-engineer the full dataframe used by both scripts."""
    raw = load_raw_data(data_path=data_path, columns_path=columns_path)
    cleaned = clean_missing_values(raw)
    return add_feature_engineering(cleaned)


def build_classification_xy(df_model: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Create the encoded classification feature matrix and target vector."""
    exclude_cols = [
        "label",
        "weight",
        "education",
        "marital stat",
        "age_group",
    ]

    categorical_cols = [
        col
        for col in df_model.select_dtypes(include=["object", "category"]).columns
        if col not in exclude_cols
    ]

    # Treat this numeric-coded variable as categorical, matching the notebook logic.
    if "own business or self employed" in df_model.columns:
        categorical_cols.append("own business or self employed")

    df_encoded = pd.get_dummies(
        df_model,
        columns=list(dict.fromkeys(categorical_cols)),
        drop_first=False,
        dtype=int,
    )

    cols_to_drop = [
        "weight",
        "year",
        "education",
        "marital stat",
        "age_group",
        "detailed industry recode",
        "detailed occupation recode",
        "wage per hour",
        "capital gains",
        "capital losses",
        "dividends from stocks",
    ]
    df_train = df_encoded.drop(columns=[c for c in cols_to_drop if c in df_encoded.columns])

    X = df_train.drop(columns=["label"])
    y = df_train["label"].astype(int)
    return X, y


def ensure_output_dir(path: str | Path) -> Path:
    """Create an output directory if it does not already exist."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
