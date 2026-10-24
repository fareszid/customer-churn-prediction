import json

import pytest
from fastapi.testclient import TestClient

import churn.api
from churn.predict import ChurnModel, risk_level


def test_training_writes_artifacts(trained_artifacts):
    root = trained_artifacts["root"]
    assert trained_artifacts["model_path"].exists()
    report = json.loads((root / "reports" / "metrics.json").read_text())
    assert report["selected_model"] in {"logistic_regression", "random_forest", "xgboost"}
    assert 0 < report["decision_threshold"] < 1
    assert len(list((root / "reports" / "figures").glob("*.png"))) == 6


def test_model_learns_signal(trained_artifacts):
    # the synthetic data has a strong signal, every model should beat random
    for metrics in trained_artifacts["report"]["test_metrics"].values():
        assert metrics["roc_auc"] > 0.65


def test_predictions_are_explained(trained_artifacts, customer_record):
    model = ChurnModel.load(trained_artifacts["model_path"])
    [prediction] = model.predict([customer_record], n_factors=3)
    assert 0 <= prediction["churn_probability"] <= 1
    assert len(prediction["top_factors"]) == 3
    impacts = [abs(f["impact"]) for f in prediction["top_factors"]]
    assert impacts == sorted(impacts, reverse=True)


@pytest.mark.parametrize(
    ("probability", "expected"), [(0.05, "low"), (0.3, "medium"), (0.7, "high")]
)
def test_risk_level(probability, expected):
    assert risk_level(probability, threshold=0.2) == expected


@pytest.fixture
def client(trained_artifacts, monkeypatch):
    monkeypatch.setattr(churn.api, "MODEL_PATH", trained_artifacts["model_path"])
    with TestClient(churn.api.app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_endpoint(client, customer_record):
    response = client.post("/predict", json=customer_record)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "churn_probability",
        "risk_level",
        "contact_customer",
        "threshold",
        "top_factors",
    }


def test_batch_endpoint(client, customer_record):
    response = client.post("/predict/batch", json={"customers": [customer_record] * 3})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3


def test_invalid_input_is_rejected(client, customer_record):
    bad = dict(customer_record, Contract="Weekly")
    assert client.post("/predict", json=bad).status_code == 422


def test_model_info(client):
    info = client.get("/model").json()
    assert info["model_name"] in {"logistic_regression", "random_forest", "xgboost"}
