"""FastAPI app. Run `uvicorn churn.api:app` and open http://localhost:8000/docs."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from churn.config import MODEL_PATH
from churn.predict import ChurnModel
from churn.schemas import BatchRequest, BatchResponse, Customer, ModelInfo, Prediction

state: dict[str, ChurnModel] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    model = ChurnModel.load(MODEL_PATH)
    _ = model.explainer  # warm up the SHAP explainer so the first request isn't slow
    state["model"] = model
    yield
    state.clear()


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Scores churn risk, explains each prediction with SHAP and recommends whether "
    "the customer should receive a retention offer.",
    lifespan=lifespan,
)


def get_model() -> ChurnModel:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state["model"]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model", response_model=ModelInfo)
def model_info() -> ModelInfo:
    bundle = get_model().bundle
    return ModelInfo(
        model_name=bundle["model_name"],
        trained_at=bundle["trained_at"],
        threshold=bundle["threshold"],
        test_metrics={k: round(float(v), 4) for k, v in bundle["metrics"].items()},
        versions=bundle["versions"],
    )


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    return Prediction(**get_model().predict([customer.model_dump()])[0])


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    records = [customer.model_dump() for customer in request.customers]
    return BatchResponse(predictions=[Prediction(**p) for p in get_model().predict(records)])
