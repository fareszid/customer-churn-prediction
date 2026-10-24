import numpy as np
import pandas as pd

from churn.data import clean
from churn.features import NUMERIC_FEATURES, add_features, build_preprocessor


def test_clean_fixes_blank_total_charges_and_encodes_target(raw_dataset):
    df = clean(raw_dataset)
    assert df["TotalCharges"].dtype.kind == "f"
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()
    assert set(df["Churn"].unique()) <= {0, 1}
    assert df.index.name == "customerID"


def test_add_features_values():
    customer = pd.DataFrame(
        [
            {
                "tenure": 10,
                "MonthlyCharges": 80.0,
                "TotalCharges": 600.0,
                "PaymentMethod": "Credit card (automatic)",
                "InternetService": "Fiber optic",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
            }
        ]
    )
    row = add_features(customer).iloc[0]
    assert row["num_services"] == 4  # phone + internet + security + tech support
    assert row["avg_monthly_spend"] == 60.0
    assert row["charge_increase"] == 20.0
    assert row["is_new_customer"] == 0
    assert row["auto_payment"] == 1


def test_new_customer_has_no_spurious_charge_increase(raw_dataset):
    df = add_features(clean(raw_dataset))
    new = df[df["tenure"] == 0]
    assert len(new) > 0
    assert np.allclose(new["charge_increase"], 0)


def test_preprocessor_output_is_finite(raw_dataset):
    df = clean(raw_dataset)
    encoded = build_preprocessor().fit_transform(df)
    assert np.isfinite(encoded).all()
    assert encoded.shape[1] > len(NUMERIC_FEATURES)
