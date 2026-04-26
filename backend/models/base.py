import joblib
import pandas as pd
from abc import ABC, abstractmethod
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class BasePredictor(ABC):
    """
    Abstract base class for all inference models in the BullRun pipeline.
    Enforces standardized load and predict structure.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Successfully loaded model from {self.model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}. Predictor will fail.")
            raise
            
    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a dataframe containing required features and appends prediction columns.
        Must return the modified DataFrame.
        """
        pass
