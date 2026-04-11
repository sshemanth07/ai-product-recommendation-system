from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, y_train, X_train_scaled):
    models = {}

    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train_scaled, y_train)
    models["Logistic Regression"] = lr

    rf = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    return models
