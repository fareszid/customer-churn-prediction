"""Request and response models for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
InternetDependent = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """One customer, with the same field names as the dataset."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 3,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 99.65,
                "TotalCharges": 298.95,
            }
        }
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetDependent
    OnlineBackup: InternetDependent
    DeviceProtection: InternetDependent
    TechSupport: InternetDependent
    StreamingTV: InternetDependent
    StreamingMovies: InternetDependent
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class Factor(BaseModel):
    feature: str
    value: str | float | int | None
    impact: float = Field(description="SHAP contribution in log-odds; positive = more churn risk")


class Prediction(BaseModel):
    churn_probability: float
    risk_level: Literal["low", "medium", "high"]
    contact_customer: bool = Field(
        description="True when the probability is above the profit-optimal threshold"
    )
    threshold: float
    top_factors: list[Factor]


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    model_name: str
    trained_at: str
    threshold: float
    test_metrics: dict[str, float]
    versions: dict[str, str]
