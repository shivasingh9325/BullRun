import pandas as pd
import os
import sys

# Ensure src is in python path
sys.path.append('e:/Bull_Run/src')

from bullrun.rl.agent import PortfolioManagerAgent
from bullrun.utils.logger import get_logger

logger = get_logger(__name__)

processed_path = 'e:/Bull_Run/data/processed/nifty50_rl_train_set.csv'

def main():
    if not os.path.exists(processed_path):
        logger.error(f"Features not found at {processed_path}. Run feature generation first.")
        return
        
    logger.info("Loading NIFTY 50 features...")
    df = pd.read_csv(processed_path)
    
    # Split into map for Multi-Stock Callback
    stock_data_map = {}
    for ticker, group in df.groupby('Stock'):
        stock_data_map[ticker] = group.copy()
        
    logger.info(f"Initialized training map for {len(stock_data_map)} stocks.")
    
    # Initialize Agent (use the first stock data just for env init)
    first_ticker = list(stock_data_map.keys())[0]
    agent = PortfolioManagerAgent(stock_data_map[first_ticker])
    
    # Start Multi-Stock Training
    # 150k steps is recommended for alpha discovery in a multi-asset universe
    agent.train(stock_data_map, total_timesteps=150000)
    
    # Save the generalized model
    model_dir = "e:/Bull_Run/models/rl"
    os.makedirs(model_dir, exist_ok=True)
    agent.save(os.path.join(model_dir, "ppo_multi_asset_manager"))
    
    logger.info("SUCCESS: Generalized Multi-Asset model saved.")

if __name__ == "__main__":
    main()
