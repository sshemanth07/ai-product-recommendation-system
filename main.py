from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import subprocess
import sys
import time
import threading
import webbrowser
import os

from src.data_loader import load_data
from src.feature_engineering import build_features
from src.recommender import create_training_pairs, build_product_features, merge_features
from src.models import train_models
from src.evaluation import evaluate_models
from src.utils import save_plots

def run_training():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    (results_dir / "plots").mkdir(exist_ok=True)

    print("Loading data...")
    clients, events, product_props = load_data()

    print("Building user features...")
    user_features = build_features(clients, events, product_props, sample_size=150000)
    
    user_features.to_csv(results_dir / "user_features.csv", index=False)
    
    sample_clients = user_features["client_id"].tolist()

    print("Building product features...")
    product_features = build_product_features(product_props)

    print("Creating client-product training pairs...")
    training_pairs, popular_skus = create_training_pairs(events, product_props, sample_clients)

    print("Merging features...")
    df = merge_features(training_pairs, user_features, product_features)

    X = df.drop(["client_id", "sku", "label"], axis=1)
    y = df["label"]
    X = X.select_dtypes(include=[np.number])

    print(f"Dataset: {X.shape[0]} pairs, {X.shape[1]} features")
    print(f"Label distribution: {y.value_counts().to_dict()}")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("Training models...")
    models = train_models(X_train, y_train, X_train_scaled, X_val, y_val, X_val_scaled)

    print("Evaluating models...")
    metrics_df, predictions, probabilities = evaluate_models(
        models, X_test, X_test_scaled, y_test, results_dir
    )

    print("Saving plots...")
    save_plots(predictions, probabilities, y_test, results_dir)

    print("\n Training completed successfully!")
    print(f"Best model: {metrics_df['F1-Score'].idxmax()}")

def start_api():
    os.chdir("api")
    subprocess.run([sys.executable, "main.py"])
    os.chdir("..")

def start_dashboard():
    time.sleep(3)
    webbrowser.open("http://localhost:8501")
    dashboard_path = Path(__file__).parent / "dashboard.py"
    subprocess.run(["streamlit", "run", str(dashboard_path)])

def main():
    print("=" * 60)
    print("Product Recommendation System")
    print("=" * 60)
    
    print("\n[1/3] Training models and generating results...")
    run_training()
    
    print("\n[2/3] Starting API server on port 8000...")
    api_thread = threading.Thread(target=start_api)
    api_thread.start()
    
    time.sleep(2)
    
    print("\n[3/3] Starting Dashboard...")
    dashboard_thread = threading.Thread(target=start_dashboard)
    dashboard_thread.start()
    
    print("\n" + "=" * 60)
    print(" System Ready!")
    print("   Dashboard: http://localhost:8501")
    print("   API: http://localhost:8000")
    print("   API Docs: http://localhost:8000/docs")
    print("=" * 60)
    print("\nPress Ctrl+C to stop all services\n")
    
    try:
        api_thread.join()
        dashboard_thread.join()
    except KeyboardInterrupt:
        print("\nShutting down services...")
        os._exit(0)

if __name__ == "__main__":
    main()