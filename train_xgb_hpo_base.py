from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from clearml import Task

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier


# ============================================================
# ClearML Base Task
# ============================================================

task = Task.init(
    project_name="AI-Based Product Recommendation System",
    task_name="Base XGBoost Training Task for HPO",
    tags=["HPO", "XGBoost", "Base Task", "Sprint 3"]
)

logger = task.get_logger()


# ============================================================
# Hyperparameters controlled by ClearML HPO
# ============================================================

params = {
    "max_depth": 6,
    "learning_rate": 0.10,
    "n_estimators": 100,
    "subsample": 0.80,
    "colsample_bytree": 0.80
}

params = task.connect(params)

print("Current hyperparameters:")
print(params)


# ============================================================
# Demo dataset for HPO
# Replace this with your real processed recommendation dataset later
# ============================================================

X, y = make_classification(
    n_samples=20000,
    n_features=25,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.67, 0.33],
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# Train XGBoost
# ============================================================

model = XGBClassifier(
    max_depth=int(params["max_depth"]),
    learning_rate=float(params["learning_rate"]),
    n_estimators=int(params["n_estimators"]),
    subsample=float(params["subsample"]),
    colsample_bytree=float(params["colsample_bytree"]),
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)


# ============================================================
# Evaluation
# ============================================================

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print("Validation Results")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)


# ============================================================
# Send metrics to ClearML
# ============================================================

logger.report_scalar(
    title="Validation",
    series="Accuracy",
    value=float(accuracy),
    iteration=0
)

logger.report_scalar(
    title="Validation",
    series="Precision",
    value=float(precision),
    iteration=0
)

logger.report_scalar(
    title="Validation",
    series="Recall",
    value=float(recall),
    iteration=0
)

logger.report_scalar(
    title="Validation",
    series="F1-Score",
    value=float(f1),
    iteration=0
)


# ============================================================
# Save artifacts
# ============================================================

Path("results").mkdir(exist_ok=True)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Value": [accuracy, precision, recall, f1]
})

metrics_df.to_csv("results/xgboost_hpo_metrics.csv", index=False)

joblib.dump(model, "results/xgboost_hpo_model.pkl")

task.upload_artifact(
    name="xgboost_hpo_metrics",
    artifact_object="results/xgboost_hpo_metrics.csv"
)

task.upload_artifact(
    name="xgboost_hpo_model",
    artifact_object="results/xgboost_hpo_model.pkl"
)

print("Base XGBoost HPO task completed.")
print("TASK_ID:", task.id)
print(task.get_output_log_web_page())

task.close()