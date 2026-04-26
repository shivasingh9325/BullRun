import joblib
import xgboost as xgb
import os

model_path = r"backend/models/meta_model/xgb_meta_model.pkl"

try:
    model = joblib.load(model_path)
    print(f"MODEL_TYPE: {type(model)}")
    
    # Try Booster names
    if hasattr(model, 'get_booster'):
        booster = model.get_booster()
        f_names = booster.feature_names
        print(f"FEATURE_NAMES: {f_names}")
    
    # Try Scikit-learn names
    if hasattr(model, 'feature_names_in_'):
        print(f"FEATURE_NAMES_IN: {model.feature_names_in_}")

except Exception as e:
    print(f"ERROR: {e}")
