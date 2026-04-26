"""
MetaModel: High-precision XGBoost gate that validates technical signals
against volatility, sentiment, and drawdown history.

Features (strict positional order — must match training):
    0: Tech_Prob_SELL
    1: Tech_Prob_BUY
    2: Tech_Prob_HOLD
    3: Sentiment_Score
    4: News_Volume
    5: Pred_Return_5d
    6: Volatility_14d
    7: Max_Drawdown_30d

Output classes (must match training label encoding):
    0: SELL | 1: HOLD | 2: BUY
"""

import pandas as pd
import numpy as np
from backend.models.base import BasePredictor
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class MetaModel(BasePredictor):
    def __init__(self, model_path: str, confidence_threshold: float = 0.55):
        super().__init__(model_path)
        # CRITICAL: Feature order must match training data (positional)
        # Brute-force audit confirms order: SELL, BUY, HOLD
        self.features = [
            'Tech_Prob_SELL', 'Tech_Prob_BUY', 'Tech_Prob_HOLD',
            'Sentiment_Score', 'News_Volume',
            'Pred_Return_5d',
            'Volatility_14d', 'Max_Drawdown_30d'
        ]
        # Configurable threshold — overridden from prod_params.yaml in pipeline
        self.confidence_threshold = confidence_threshold

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            self.load_model()
            if hasattr(self.model, 'feature_names_in_'):
                logger.info(f"META-VALIDATION: Internal feature names: {list(self.model.feature_names_in_)}")
            else:
                logger.warning("META-VALIDATION: No internal feature names found. Using strict positional alignment.")

        # Ensure all features exist
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required features for MetaModel: {missing}")

        X = df[self.features].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # XGBoost is scale-invariant; prediction without scaling.
        meta_probs = self.model.predict_proba(X)

        df['Meta_Prob_SELL'] = meta_probs[:, 0]
        df['Meta_Prob_HOLD'] = meta_probs[:, 1]
        df['Meta_Prob_BUY'] = meta_probs[:, 2]
        df['Meta_Pred'] = np.argmax(meta_probs, axis=1)

        return df

    def evaluate(self, df: pd.DataFrame, y_true: pd.Series) -> dict:
        """
        Evaluate the meta model on labeled data.
        Returns accuracy, precision, recall, and confusion matrix.

        Args:
            df: Feature DataFrame (same schema as predict).
            y_true: Ground-truth labels as integers (0=SELL, 1=HOLD, 2=BUY).

        Returns:
            dict with 'accuracy', 'precision', 'recall', 'confusion_matrix'.
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

        if self.model is None:
            self.load_model()

        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features for evaluation: {missing}")

        X = df[self.features].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
        y_pred = self.model.predict(X)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        rec = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        cm = confusion_matrix(y_true, y_pred).tolist()

        metrics = {
            "accuracy": round(acc, 4),
            "precision_weighted": round(prec, 4),
            "recall_weighted": round(rec, 4),
            "confusion_matrix": cm,
            "labels": ["SELL", "HOLD", "BUY"],
            "n_samples": len(y_true)
        }

        logger.info(f"META-EVAL: Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f}")
        return metrics
