import numpy as np
import pytest

from churn.business import (
    CampaignEconomics,
    best_threshold,
    campaign_profit,
    expected_value_targets,
)

ECON = CampaignEconomics(offer_cost=50, save_rate=0.5, months_retained=10)


def test_campaign_profit_by_hand():
    y = np.array([1, 0, 1, 0])
    targeted = np.array([True, True, False, False])
    monthly = np.array([100.0, 100.0, 100.0, 100.0])
    # one churner reached: 0.5 * 100 * 10 = 500 revenue, two contacts cost 100
    assert campaign_profit(y, targeted, monthly, ECON) == pytest.approx(400)


def test_no_campaign_has_zero_profit():
    y = np.array([1, 0, 1])
    assert campaign_profit(y, np.zeros(3, bool), np.ones(3) * 70, ECON) == 0


def test_best_threshold_separates_perfect_scores():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = best_threshold(y, proba, np.full(4, 100.0), ECON)
    assert 0.2 < threshold <= 0.8


def test_expected_value_targets_favour_valuable_customers():
    proba = np.array([0.2, 0.2])
    monthly = np.array([20.0, 120.0])
    # EV = 0.2 * 0.5 * 10 * monthly -> 20 (below cost) and 120 (above cost)
    assert expected_value_targets(proba, monthly, ECON).tolist() == [False, True]
