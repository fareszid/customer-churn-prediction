"""Load the trained model and score customers."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import joblib
import pandas as pd

from churn.config import MODEL_PATH
from churn.explain import ChurnExplainer
from churn.features import INPUT_FEATURES


def risk_level(probability: float, threshold: float) -> str:
    if probability >= max(0.5, threshold):
        return "high"
    if probability >= threshold:
        return "medium"
    return "low"


class ChurnModel:
    def __init__(self, bundle: dict):
        self.pipeline = bundle["pipeline"]
        self.threshold = float(bundle["threshold"])
        self.bundle = bundle

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> ChurnModel:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Run `make train` (python -m churn.train) first."
            )
        # joblib uses pickle, so only load files written by our own training script
        return cls(joblib.load(path))

    @cached_property
    def explainer(self) -> ChurnExplainer:
        return ChurnExplainer(self.pipeline, self.bundle["background"])

    def predict(self, records: list[dict], n_factors: int = 5) -> list[dict]:
        X = pd.DataFrame.from_records(records)[INPUT_FEATURES]
        probabilities = self.pipeline.predict_proba(X)[:, 1]
        factors = self.explainer.top_factors(X, k=n_factors)
        return [
            {
                "churn_probability": round(float(p), 4),
                "risk_level": risk_level(p, self.threshold),
                "contact_customer": bool(p >= self.threshold),
                "threshold": self.threshold,
                "top_factors": f,
            }
            for p, f in zip(probabilities, factors, strict=True)
        ]
