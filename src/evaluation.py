import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

def evaluate_models(models, X_test, X_test_scaled, y_test, results_dir):
    metrics = {}
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            X_input = X_test_scaled
        else:
            X_input = X_test

        y_pred = model.predict(X_input)
        y_prob = model.predict_proba(X_input)[:, 1]

        predictions[name] = y_pred
        probabilities[name] = y_prob

        metrics[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "AUROC": roc_auc_score(y_test, y_prob),
        }

    metrics_df = pd.DataFrame(metrics).T.round(4)
    metrics_df.to_csv(results_dir / "metrics_comparison.csv")

    print("\n=== MODEL PERFORMANCE COMPARISON ===")
    print(metrics_df)

    with open(results_dir / "classification_reports.txt", "w") as f:
        for name in models:
            f.write(f"\n=== {name} ===\n")
            f.write(classification_report(y_test, predictions[name], zero_division=0))

    return metrics_df, predictions, probabilities
