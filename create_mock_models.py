import os
import joblib
import json
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

import os
os.makedirs("models/technical_model", exist_ok=True)
os.makedirs("models/meta_model", exist_ok=True)
os.makedirs("models/rl", exist_ok=True)

# 1. Technical Model
tech_model = DummyClassifier(strategy="constant", constant=2) # 2 corresponds to BUY
X_tech_dummy = np.zeros((3, 7)) # 7 features
y_dummy = np.array([0, 1, 2])
tech_model.fit(X_tech_dummy, y_dummy)
joblib.dump(tech_model, "models/technical_model/technical_model_best.pkl")

# 2. Scaler
scaler = StandardScaler()
scaler.mean_ = np.zeros(7)
scaler.scale_ = np.ones(7)
scaler.var_ = np.ones(7)
scaler.n_features_in_ = 7
joblib.dump(scaler, "models/technical_model/scaler_best.pkl")

# 2b. Feature List
tech_features = ["Close", "Volume", "RSI_14", "MACD", "MACD_Signal", "BB_High", "BB_Low"]
with open("models/technical_model/feature_list.json", "w") as f:
    json.dump(tech_features, f)

# 3. Meta Model
meta_model = DummyClassifier(strategy="constant", constant=2) # 2 corresponds to BUY
X_meta_dummy = np.zeros((3, 8)) # 8 features
meta_model.fit(X_meta_dummy, y_dummy)
joblib.dump(meta_model, "models/meta_model/xgb_meta_model.pkl")

print("Mock models created with correct shapes.")
