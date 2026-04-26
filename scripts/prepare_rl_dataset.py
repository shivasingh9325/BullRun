import pandas as pd
import os
import sys

# Ensure src is in python path
sys.path.append('e:/Bull_Run/src')

from bullrun.models.technical_model import TechnicalPredictor
from bullrun.models.sentiment_model import SentimentPredictor
from bullrun.models.meta import MetaModel
from bullrun.utils.logger import get_logger

logger = get_logger(__name__)

input_path = 'e:/Bull_Run/data/processed/nifty50_features.csv'
output_path = 'e:/Bull_Run/data/processed/nifty50_rl_train_set.csv'

def main():
    if not os.path.exists(input_path):
        logger.error(f"Input features not found at {input_path}")
        return
        
    logger.info("Loading technical features...")
    df = pd.read_csv(input_path)
    
    # Initialize Predictors
    # Note: Using absolute paths for reliability
    tech_model = TechnicalPredictor("e:/Bull_Run/models/technical_model")
    sent_model = SentimentPredictor()
    meta_model = MetaModel(os.path.join("e:/Bull_Run/models", "meta_model", "xgb_meta_model.pkl"))
    
    logger.info("Running technical inference...")
    df = tech_model.predict(df)
    
    logger.info("Running sentiment inference (stubs)...")
    df = sent_model.predict(df)
    
    logger.info("Running Meta Model fusion...")
    df = meta_model.predict(df)
    
    logger.info(f"Final training set columns: {df.columns.tolist()}")
    df.to_csv(output_path, index=False)
    logger.info(f"SUCCESS: Prepared RL training set at {output_path}")

if __name__ == "__main__":
    main()
