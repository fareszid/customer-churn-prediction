"""Feature engineering and preprocessing.

Everything is part of the sklearn pipeline, so the API takes raw records and applies exactly
the same steps as training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

RAW_NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

ENGINEERED_FEATURES = [
    "num_services",
    "avg_monthly_spend",
    "charge_increase",
    "is_new_customer",
    "auto_payment",
]

NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
INPUT_FEATURES = RAW_NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to raw customer records."""
    df = df.copy()
    services = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)
    has_internet = (df["InternetService"] != "No").astype(int)
    df["num_services"] = services + has_internet

    # current bill vs. average past bill: a recent price increase often comes before churn
    months_billed = df["tenure"].clip(lower=1)
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / months_billed, df["MonthlyCharges"]
    )
    df["charge_increase"] = df["MonthlyCharges"] - df["avg_monthly_spend"]
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    df["auto_payment"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    return df


def build_preprocessor() -> Pipeline:
    """Feature engineering, then scaling and one-hot encoding."""
    encode = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("engineer", FunctionTransformer(add_features, validate=False)),
            ("encode", encode),
        ]
    )


def original_feature_name(encoded_name: str) -> str:
    """Map an encoded column like `Contract_Two year` back to `Contract`."""
    for column in CATEGORICAL_FEATURES:
        if encoded_name.startswith(f"{column}_"):
            return column
    return encoded_name
