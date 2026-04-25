# test.py - Quick model validation
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from src.data_loader import load_data
from src.feature_engineering import build_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

print("=" * 60)
print("MODEL VALIDATION TEST")
print("=" * 60)

# Load data
print("\n[1/5] Loading data...")
clients, events, product_props = load_data()

# Build features
print("[2/5] Building features...")
df = build_features(clients, events, product_props, sample_size=1000)

# Prepare data
X = df.drop(["client_id", "target"], axis=1).fillna(0)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
print("[3/5] Training models...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Predict
print("[4/5] Making predictions...")
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
rf_prob = rf.predict_proba(X_test)[:, 1]
lr_pred = lr.predict(X_test_scaled)
rf_pred = rf.predict(X_test)

# Results
print("\n[5/5] RESULTS")
print("=" * 60)

print("\nLOGISTIC REGRESSION:")
print(f"  AUROC:     {roc_auc_score(y_test, lr_prob):.4f}")
print(f"  Accuracy:  {accuracy_score(y_test, lr_pred):.4f}")
print(f"  F1-Score:  {f1_score(y_test, lr_pred):.4f}")

print("\nRANDOM FOREST:")
print(f"  AUROC:     {roc_auc_score(y_test, rf_prob):.4f}")
print(f"  Accuracy:  {accuracy_score(y_test, rf_pred):.4f}")
print(f"  F1-Score:  {f1_score(y_test, rf_pred):.4f}")

# Class balance check
print("\n" + "=" * 60)
print("DATA CHECK")
print("=" * 60)
print(f"Test set size: {len(y_test)} samples")
print(f"Class 0 (engaged buyers): {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
print(f"Class 1 (low purchase):   {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")

# Interpretation
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

if roc_auc_score(y_test, rf_prob) == 1.0:
    print("\n AUROC = 1.0 (Perfect classification)")
    
    if (y_test == 1).sum() < 50 or (y_test == 0).sum() < 50:
        print("   → Small sample size in one class")
        print("   → AUROC may be artificially high")
    else:
        print("   → Large test set with balanced classes")
        print("   → Perfect classification is genuine!")
        print("   → Your features separate classes perfectly")
        
    print("\n   Suggestion: Try different random_state to verify")
    
elif roc_auc_score(y_test, rf_prob) > 0.95:
    print(f"\n✓ Excellent AUROC: {roc_auc_score(y_test, rf_prob):.4f}")
    print("  Model performs very well on test data")
    
elif roc_auc_score(y_test, rf_prob) > 0.85:
    print(f"\n✓ Good AUROC: {roc_auc_score(y_test, rf_prob):.4f}")
    print("  Model is reliable for recommendations")
    
else:
    print(f"\n AUROC: {roc_auc_score(y_test, rf_prob):.4f}")
    print("  Model may need improvement")

# Test with different random state
print("\n" + "=" * 60)
print("QUICK CROSS-VALIDATION (3 different splits)")
print("=" * 60)

auroc_scores = []
for seed in [42, 123, 456]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    auroc_scores.append(roc_auc_score(y_test, rf_prob))

print(f"AUROC scores: {[f'{s:.4f}' for s in auroc_scores]}")
print(f"Mean AUROC: {np.mean(auroc_scores):.4f}")
print(f"Std Dev: {np.std(auroc_scores):.4f}")

if np.std(auroc_scores) < 0.01:
    print("✓ Stable performance across different splits")
else:
    print("⚠️ Performance varies with data split")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)