from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class ANNRecommender(nn.Module):
    def __init__(self, input_dim):
        super(ANNRecommender, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def train_ann_model(X_train, y_train, X_val, y_val, input_dim, epochs=50, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training ANN on {device}")
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values if hasattr(y_val, 'values') else y_val, dtype=torch.float32).view(-1, 1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = ANNRecommender(input_dim).to(device)
    
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    pos_weight = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    return model, device

class ANNWrapper:
    def __init__(self, model, device, scaler=None):
        self.model = model
        self.device = device
        self.scaler = scaler
    
    def predict(self, X):
        if self.scaler:
            X = self.scaler.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits)
        return (probs.cpu().numpy() > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        if self.scaler:
            X = self.scaler.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits)
        probs = probs.cpu().numpy().flatten()
        return np.column_stack((1 - probs, probs))

def train_models(X_train, y_train, X_train_scaled, X_val, y_val, X_val_scaled):
    models = {}
    
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, C=0.5, class_weight='balanced', n_jobs=-1)
    lr.fit(X_train_scaled, y_train)
    models["Logistic Regression"] = lr
    
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10,
                                random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf
    
    print("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        eval_metric='logloss', use_label_encoder=False, n_jobs=-1)
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb
    
    print("Training ANN...")
    ann_model, device = train_ann_model(X_train_scaled, y_train, X_val_scaled, y_val, X_train_scaled.shape[1])
    ann_wrapper = ANNWrapper(ann_model, device)
    models["Neural Network"] = ann_wrapper
    
    return models