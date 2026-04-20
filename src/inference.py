from pathlib import Path
import joblib
import pandas as pd

ARTIFACTS_DIR = Path("artifacts")


def load_artifacts():
    return {
        "lr_model": joblib.load(ARTIFACTS_DIR / "logistic_regression.pkl"),
        "rf_model": joblib.load(ARTIFACTS_DIR / "random_forest.pkl"),
        "scaler": joblib.load(ARTIFACTS_DIR / "scaler.pkl"),
        "feature_columns": joblib.load(ARTIFACTS_DIR / "feature_columns.pkl"),
        "metadata": joblib.load(ARTIFACTS_DIR / "model_metadata.pkl"),
    }


def prepare_input(payload, feature_columns):
    row = {col: 0 for col in feature_columns}
    for k, v in payload.items():
        if k in row:
            row[k] = v
    return pd.DataFrame([row])


def predict(payload, model_name="random_forest"):
    artifacts = load_artifacts()
    X = prepare_input(payload, artifacts["feature_columns"])

    if model_name == "logistic_regression":
        X_scaled = artifacts["scaler"].transform(X)
        model = artifacts["lr_model"]
        pred = int(model.predict(X_scaled)[0])
        prob = float(model.predict_proba(X_scaled)[0][1])
    else:
        model = artifacts["rf_model"]
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])

    return {
        "prediction": pred,
        "probability": prob
    }
