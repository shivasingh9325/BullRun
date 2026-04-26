import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
import joblib

# Ensure backend and its subdirs are in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from bullrun.core.fetcher import DataFetcher
from bullrun.core.technical import calculate_technical_features
from bullrun.models.technical_model import TechnicalPredictor
from bullrun.models.meta import MetaModel

def diagnose():
    print("\n" + "="*50)
    print("BULLRUN PIPELINE DIAGNOSTIC: FINDING THE FAULT")
    print("="*50)

    # 1. Config & Path Check
    config_path = os.path.join(project_root, "configs", "prod_params.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    models_dir = os.path.join(project_root, "models")
    tech_model_dir = os.path.join(models_dir, "technical_model")
    meta_model_path = os.path.join(models_dir, "meta_model", "xgb_meta_model.pkl")

    # 2. Inspect Meta Model Internals (XGBoost Feature Audit)
    print("\n--- STAGE 1: METAMODEL FEATURE AUDIT ---")
    try:
        model = joblib.load(meta_model_path)
        print(f"MetaModel loaded: {type(model)}")
        
        # Check XGBoost/Sklearn feature attributes
        if hasattr(model, 'feature_names_in_'):
            print(f"Feature Names in (Training): {model.feature_names_in_}")
        elif hasattr(model, 'get_booster'):
            print(f"Feature Names (Booster): {model.get_booster().feature_names}")
        else:
            print("Could not retrieve internal feature names. Using meta.py configuration.")
    except Exception as e:
        print(f"MetaModel Inspection Failed: {e}")

    # 3. Fetch Real Data for Trace
    print("\n--- STAGE 2: DATA FETCHING & FEATURE ENGINEERING ---")
    fetcher = DataFetcher(config_path=config_path)
    raw_df = fetcher.fetch_all(["RELIANCE.NS"], period="60d")
    print(f"Raw Data: {raw_df.shape}")
    
    feat_df = calculate_technical_features(raw_df)
    
    # 4. Technical Model Trace
    print("\n--- STAGE 3: TECHNICAL MODEL TRACE ---")
    tech_pred = TechnicalPredictor(tech_model_dir)
    tech_results = tech_pred.predict(feat_df)
    latest_tech = tech_results.tail(1)
    
    probs = [latest_tech['Tech_Prob_SELL'].item(), 
             latest_tech['Tech_Prob_HOLD'].item(), 
             latest_tech['Tech_Prob_BUY'].item()]
    print(f"Technical Probs (SELL/HOLD/BUY): {['%.4f' % x for x in probs]}")

    # 5. Meta Model Trace
    print("\n--- STAGE 4: METAMODEL TRACE ---")
    # Add sentiment baseline (mimic pipeline.py)
    tech_results['Sentiment_Score'] = 0.5
    tech_results['News_Volume'] = 10
    
    meta_model = MetaModel(meta_model_path)
    meta_results = meta_model.predict(tech_results)
    
    latest_meta_buy = meta_results['Meta_Prob_BUY'].values[-1]
    latest_meta_sell = meta_results['Meta_Prob_SELL'].values[-1]
    print(f"Meta Probability (BUY): {latest_meta_buy:.4f}")
    print(f"Meta Probability (SELL): {latest_meta_sell:.4f}")
    
    # 6. Varied Input Sanity Test
    print("\n--- STAGE 5: VARIED INPUT SANITY TEST ---")
    mock_row = latest_tech.copy()
    mock_row['Tech_Prob_BUY'] = 0.95
    mock_row['Tech_Prob_SELL'] = 0.01
    mock_row['Tech_Prob_HOLD'] = 0.04
    mock_row['Sentiment_Score'] = 0.90
    mock_row['Volatility_14d'] = 1.0 # High vol
    
    sanity_out = meta_model.predict(mock_row)
    sanity_buy = sanity_out['Meta_Prob_BUY'].values[0]
    print(f"Forced Bullish Input Result (BUY Prob): {sanity_buy:.4f}")

if __name__ == "__main__":
    diagnose()
