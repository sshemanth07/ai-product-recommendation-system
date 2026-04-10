from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_data
from src.feature_engineering import build_features
from src.models import train_models
from src.evaluation import evaluate_models
from src.utils import save_plots

def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    (results_dir / "plots").mkdir(exist_ok=True)

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

    print("\n🎉 Evaluation completed successfully!")
    print("Check the 'results/' folder.")

if __name__ == "__main__":
    main()
