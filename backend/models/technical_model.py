import os
import joblib
import json
import pandas as pd
import numpy as np
from backend.models.base import BasePredictor
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class TechnicalPredictor(BasePredictor):
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        # Primary path for model
        model_path = os.path.join(model_dir, "technical_model_best.pkl")
        super().__init__(model_path)
        
        # Load scaler and features metadata
        self.scaler_path = os.path.join(model_dir, "scaler_best.pkl")
        self.feature_list_path = os.path.join(model_dir, "feature_list.json")
        self.scaler = None
        self.features = []

    def load_model(self):
        super().load_model()
        try:
            self.scaler = joblib.load(self.scaler_path)
            with open(self.feature_list_path, 'r') as f:
                self.features = json.load(f)
            logger.info(f"Loaded scaler and {len(self.features)} features for TechnicalPredictor.")
        except Exception as e:
            logger.error(f"Failed to load technical metadata: {e}")
            raise

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.scaler is None:
            self.load_model()
            
        # Ensure only required features are passed
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            logger.warning(f"TechnicalPredictor missing features: {missing}. Filling with 0.")
            for f in missing: df[f] = 0.0
            
        X = df[self.features].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0) # Safety check for NaNs/Infs
        X_scaled = self.scaler.transform(X)
        
        # Predict Probabilities
        probs = self.model.predict_proba(X_scaled)
        
        df['Tech_Prob_SELL'] = probs[:, 0]
        df['Tech_Prob_HOLD'] = probs[:, 1]
        df['Tech_Prob_BUY'] = probs[:, 2]
        df['Tech_Pred'] = np.argmax(probs, axis=1)
        
        return df
