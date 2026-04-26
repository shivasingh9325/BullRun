import joblib
m = joblib.load('backend/models/meta_model/xgb_meta_model.pkl')
if hasattr(m, 'feature_names_in_'):
    print(f"FEATURE_NAMES_IN: {list(m.feature_names_in_)}")
else:
    print("FEATURE_NAMES_IN: None")
