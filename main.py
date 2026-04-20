from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_data
from src.feature_engineering import build_features
from src.models import train_models
from src.evaluation import evaluate_models
from src.utils import save_plots


def main():
    results_dir = Path("results")
    artifacts_dir = Path("artifacts")

    results_dir.mkdir(exist_ok=True)
    (results_dir / "plots").mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)

    print("Loading data...")
    clients, events, product_props = load_data()

    print("Building features...")
    df = build_features(clients, events, product_props, sample_size=8000)

    X = df.drop(["client_id", "target"], axis=1).fillna(0)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training models...")
    models = train_models(X_train, y_train, X_train_scaled)

    print("Evaluating models...")
    metrics_df, predictions, probabilities = evaluate_models(
        models, X_test, X_test_scaled, y_test, results_dir
    )

    print("Saving plots...")
    save_plots(predictions, probabilities, y_test, results_dir)

    print("Saving artifacts...")
    joblib.dump(models["Logistic Regression"], artifacts_dir / "logistic_regression.pkl")
    joblib.dump(models["Random Forest"], artifacts_dir / "random_forest.pkl")
    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    joblib.dump(list(X.columns), artifacts_dir / "feature_columns.pkl")
    joblib.dump(
        {
            "target_definition": "1 if total_buys <= 1 else 0",
            "feature_count": len(X.columns),
            "models": list(models.keys()),
        },
        artifacts_dir / "model_metadata.pkl",
    )

    print("Done.")


if __name__ == "__main__":
    main()
