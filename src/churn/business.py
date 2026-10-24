"""Retention campaign economics: profit, best threshold and expected-value targeting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from churn.config import MONTHS_RETAINED, OFFER_COST, SAVE_RATE


@dataclass(frozen=True)
class CampaignEconomics:
    offer_cost: float = OFFER_COST
    save_rate: float = SAVE_RATE
    months_retained: int = MONTHS_RETAINED

    def value_if_saved(self, monthly_charges: np.ndarray) -> np.ndarray:
        return np.asarray(monthly_charges, dtype=float) * self.months_retained


DEFAULT_ECONOMICS = CampaignEconomics()


def campaign_profit(
    y_true: np.ndarray,
    targeted: np.ndarray,
    monthly_charges: np.ndarray,
    economics: CampaignEconomics = DEFAULT_ECONOMICS,
) -> float:
    """Profit of contacting the `targeted` customers.

    Each contact costs `offer_cost`. A contacted customer who would have churned stays with
    probability `save_rate` and pays for `months_retained` more months.
    """
    y_true = np.asarray(y_true, dtype=bool)
    targeted = np.asarray(targeted, dtype=bool)
    saved_value = economics.save_rate * economics.value_if_saved(monthly_charges)
    revenue = saved_value[targeted & y_true].sum()
    cost = economics.offer_cost * targeted.sum()
    return float(revenue - cost)


def profit_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    monthly_charges: np.ndarray,
    thresholds: np.ndarray | None = None,
    economics: CampaignEconomics = DEFAULT_ECONOMICS,
) -> tuple[np.ndarray, np.ndarray]:
    """Campaign profit at each probability threshold."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    profits = np.array(
        [campaign_profit(y_true, proba >= t, monthly_charges, economics) for t in thresholds]
    )
    return thresholds, profits


def best_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    monthly_charges: np.ndarray,
    economics: CampaignEconomics = DEFAULT_ECONOMICS,
) -> float:
    thresholds, profits = profit_curve(y_true, proba, monthly_charges, economics=economics)
    return float(thresholds[int(np.argmax(profits))])


def expected_value_targets(
    proba: np.ndarray,
    monthly_charges: np.ndarray,
    economics: CampaignEconomics = DEFAULT_ECONOMICS,
) -> np.ndarray:
    """Contact a customer when the expected gain of the offer is higher than its cost.

    Unlike a single threshold, high-paying customers get contacted at a lower churn probability.
    """
    expected_gain = proba * economics.save_rate * economics.value_if_saved(monthly_charges)
    return expected_gain > economics.offer_cost
