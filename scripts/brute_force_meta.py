import os
import sys
import yaml
import numpy as np
import pandas as pd
import joblib
from itertools import permutations

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path: sys.path.append(project_root)

from bullrun.core.fetcher import DataFetcher
from bullrun.core.technical import calculate_technical_features
from bullrun.models.technical_model import TechnicalPredictor
from bullrun.models.meta import MetaModel

def brute_force():
    meta_model_path = os.path.join(project_root, "models", "meta_model", "xgb_meta_model.pkl")
    model = joblib.load(meta_model_path)
    
    # Fetch real data for RELIANCE.NS
    fetcher = DataFetcher(config_path=os.path.join(project_root, "configs", "prod_params.yaml"))
    raw_df = fetcher.fetch_all(["RELIANCE.NS"], period="60d")
    feat_df = calculate_technical_features(raw_df)
    
    tech_pred = TechnicalPredictor(os.path.join(project_root, "models", "technical_model"))
    tech_results = tech_pred.predict(feat_df)
    latest = tech_results.tail(1).copy()
    
    # Probabilities from Technical Model
    t_sell = latest['Tech_Prob_SELL'].item()
    t_hold = latest['Tech_Prob_HOLD'].item()
    t_buy = latest['Tech_Prob_BUY'].item()
    
    # Fixed features (from meta.py distribution)
    sent = 0.5
    vol = latest['Volatility_14d'].item()
    ret = latest['Pred_Return_5d'].item()
    dd = latest['Max_Drawdown_30d'].item()
    news = 10
    
    tech_perms = list(permutations(['SELL', 'HOLD', 'BUY']))
    
    print(f"\n{'Permutation (Prob0, Prob1, Prob2)':<40} | {'Meta Prob BUY':<15} | {'Meta Prob SELL':<15}")
    print("-" * 80)
    
    results = []
    for perm in tech_perms:
        # map SELL/HOLD/BUY to the technical outputs
        mapping = {'SELL': t_sell, 'HOLD': t_hold, 'BUY': t_buy}
        p0, p1, p2 = mapping[perm[0]], mapping[perm[1]], mapping[perm[2]]
        
        # Construct input vector (8 features)
        X = np.array([[p0, p1, p2, sent, news, ret, vol, dd]])
        m_probs = model.predict_proba(X)[0]
        
        m_buy = m_probs[2]
        m_sell = m_probs[0]
        
        results.append({
            'perm': perm,
            'buy': m_buy,
            'sell': m_sell,
            'variance': np.var(m_probs)
        })
        print(f"{str(perm):<40} | {m_buy:.4f}          | {m_sell:.4f}")

    # Best permutation is likely the one with the highest variance (least neutral)
    best = max(results, key=lambda x: x['variance'])
    print(f"\n[BEST] MOST LIKELY ORDER: {best['perm']} (Variance: {best['variance']:.4f})")

if __name__ == "__main__":
    brute_force()
