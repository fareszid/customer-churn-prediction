"""Fixtures: a synthetic dataset with the real schema and a small model trained on it.

No network access needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.train import train

INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def make_raw_dataset(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    tenure = rng.integers(0, 73, n)
    monthly = np.round(rng.uniform(18, 118, n), 2)
    df = pd.DataFrame(
        {
            "customerID": [f"C{i:05d}" for i in range(n)],
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": np.where(
                phone == "No", "No phone service", rng.choice(["Yes", "No"], n)
            ),
            "InternetService": internet,
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": monthly,
            # like the raw file: TotalCharges is text and blank when tenure is 0
            "TotalCharges": [
                " " if t == 0 else f"{t * m:.2f}" for t, m in zip(tenure, monthly, strict=True)
            ],
        }
    )
    for column in INTERNET_ADDONS:
        df[column] = np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n))
    logit = (
        -1.0
        + 1.6 * (contract == "Month-to-month")
        - 0.04 * tenure
        + 0.8 * (internet == "Fiber optic")
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    df["Churn"] = np.where(churn, "Yes", "No")
    return df


@pytest.fixture(scope="session")
def raw_dataset() -> pd.DataFrame:
    return make_raw_dataset()


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory, raw_dataset):
    root = tmp_path_factory.mktemp("project")
    data_path = root / "data.csv"
    raw_dataset.to_csv(data_path, index=False)
    model_path = root / "models" / "model.joblib"
    report = train(
        n_trials=2,
        data_path=data_path,
        model_path=model_path,
        reports_dir=root / "reports",
        figures_dir=root / "reports" / "figures",
    )
    return {"root": root, "model_path": model_path, "report": report}


@pytest.fixture
def customer_record(raw_dataset) -> dict:
    record = raw_dataset.drop(columns=["customerID", "Churn"]).iloc[1].to_dict()
    record["TotalCharges"] = float(record["TotalCharges"] or 0)
    return {k: v.item() if hasattr(v, "item") else v for k, v in record.items()}
