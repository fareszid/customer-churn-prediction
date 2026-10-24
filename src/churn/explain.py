"""SHAP explanations, grouped back to the original customer fields."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.features import add_features, original_feature_name


class ChurnExplainer:
    """SHAP explainer for a fitted preprocess -> model pipeline.

    Values are in log-odds and summed over one-hot columns, so `Contract_Month-to-month` and
    `Contract_Two year` both count towards `Contract`.
    """

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame, max_background: int = 100):
        self.preprocess = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        self.encoded_names = list(self.preprocess.named_steps["encode"].get_feature_names_out())
        sample = background.sample(min(max_background, len(background)), random_state=0)
        encoded_background = self.preprocess.transform(sample)
        if isinstance(self.model, LogisticRegression):
            self._explainer = shap.LinearExplainer(self.model, encoded_background)
        else:
            self._explainer = shap.TreeExplainer(self.model, encoded_background)

    def encoded_shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        values = self._explainer.shap_values(self.preprocess.transform(X))
        values = np.asarray(values)
        if values.ndim == 3:  # some tree explainers return one array per class
            values = values[..., 1]
        return pd.DataFrame(values, columns=self.encoded_names, index=X.index)

    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values per original feature, one row per customer."""
        encoded = self.encoded_shap_values(X)
        groups = [original_feature_name(name) for name in encoded.columns]
        return encoded.T.groupby(groups, sort=False).sum().T

    def top_factors(self, X: pd.DataFrame, k: int = 5) -> list[list[dict]]:
        """Top k drivers of each prediction (positive values push towards churn)."""
        contributions = self.shap_values(X)
        values = add_features(X)  # so we can show the values of engineered features too
        factors = []
        for idx, row in contributions.iterrows():
            strongest = row.reindex(row.abs().sort_values(ascending=False).index)[:k]
            factors.append(
                [
                    {
                        "feature": feature,
                        "value": _to_python(values.loc[idx, feature]),
                        "impact": round(float(impact), 4),
                    }
                    for feature, impact in strongest.items()
                ]
            )
        return factors


def _to_python(value):
    return value.item() if hasattr(value, "item") else value
