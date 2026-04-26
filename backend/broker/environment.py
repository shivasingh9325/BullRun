import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yaml
import os
import json

class BullRunEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface for Stock Trading.
    The agent acts as a Portfolio Manager: It is fed the Meta Model probability
    and chooses the absolute Position Size (0.0 to 1.0) of capital to allocate.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data: pd.DataFrame, config_path: str = 'configs/prod_params.yaml', initial_balance=100000.0):
        super(BullRunEnv, self).__init__()
        import json
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        
        # Robust config resolution
        resolved_config = config_path
        if not os.path.exists(resolved_config):
             base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
             alt_path = os.path.join(base_dir, "configs", "prod_params.yaml")
             if os.path.exists(alt_path):
                 resolved_config = alt_path
        
        # Resolve nifty50.json
        nifty_path = 'configs/nifty50.json'
        if not os.path.exists(nifty_path):
             base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
             alt_nifty = os.path.join(base_dir, "configs", "nifty50.json")
             if os.path.exists(alt_nifty):
                 nifty_path = alt_nifty

        with open(nifty_path, 'r') as f:
            self.tickers = json.load(f)
        self.ticker_map = {t: i for i, t in enumerate(self.tickers)}
        self.num_stocks = len(self.tickers)
        
        # Load RL configuration
        with open(resolved_config, 'r') as f:
            cfg = yaml.safe_load(f)
        rl_cfg = cfg.get('rl', {})
        self.transaction_cost = rl_cfg.get('transaction_cost', 0.0015)
        self.drawdown_penalty = rl_cfg.get('drawdown_penalty', 0.15)
        self.overtrade_penalty = rl_cfg.get('overtrade_penalty', 0.0001)
        self.hold_penalty = rl_cfg.get('hold_penalty', 0.00005)
        self.profit_incentive = rl_cfg.get('profit_incentive', 1.5)
        self.profit_threshold = rl_cfg.get('profit_threshold', 0.001)
        self.trade_close_bonus = rl_cfg.get('trade_close_bonus', 0.01)
        
        # Advanced constraints
        self.min_holding_period = rl_cfg.get('min_holding_period', 3)
        self.early_exit_penalty = rl_cfg.get('early_exit_penalty', 0.0025)
        self.trade_window = rl_cfg.get('trade_window', 10)
        self.max_trades_window = rl_cfg.get('max_trades_window', 2)
        
        exposure_cfg = rl_cfg.get('exposure_limits', {})
        self.max_per_stock = exposure_cfg.get('max_per_stock', 0.10)
        self.max_total_exposure = exposure_cfg.get('max_total_exposure', 0.90)
        
        # Tracking for trade-level bonuses and constraints
        self.avg_buy_price = 0.0
        self.last_buy_step = -999
        from collections import deque
        self.trade_history = deque(maxlen=self.trade_window)
        
        # State: [Balance, Holdings, Unrealized_PnL, Meta_Prob_BUY, Volatility_14d, Pred_Return_5d] + One-Hot Stock ID
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6 + self.num_stocks,), dtype=np.float32
        )
        
        # Action: Continuous value [0.0, 1.0] representing percentage of total capital to deploy
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.prev_action = 0 # Track previous action for overtrading penalty
        self.prev_net_worth = self.initial_balance

    def _next_observation(self):
        current_price = self.data.loc[self.current_step, 'Close']
        current_exposure = (self.holdings * current_price) / self.net_worth if self.net_worth > 0 else 0
        
        # One-Hot Encoding for Stock Identity
        symbol = self.data.loc[self.current_step, 'Stock']
        one_hot = np.zeros(self.num_stocks, dtype=np.float32)
        if symbol in self.ticker_map:
            one_hot[self.ticker_map[symbol]] = 1.0
            
        core_obs = np.array([
            self.balance / self.initial_balance,
            current_exposure, # Normalized exposure [0, 1]
            (self.net_worth - self.initial_balance) / self.initial_balance,
            self.data.loc[self.current_step, 'Meta_Prob_BUY'],
            self.data.loc[self.current_step, 'Volatility_14d'] / 100.0,
            self.data.loc[self.current_step, 'Pred_Return_5d']
        ], dtype=np.float32)
        
        obs = np.concatenate([core_obs, one_hot])
        
        # Final safety check for NaNs/Infs
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def step(self, action):
        self.current_step += 1
        
        if self.current_step >= len(self.data) - 1:
            # Revert to last valid step for extracting final observation safely
            self.current_step = len(self.data) - 1
            return self._next_observation(), 0.0, True, False, {"net_worth": self.net_worth, "action": "HOLD", "quantity": 0.0, "price": 0.0, "step_reward": 0.0}

        current_price = self.data.loc[self.current_step, 'Close']
        target_allocation_pct = action[0]
        
        # Enforce Realism: Cap allocation by max_per_stock limit
        target_allocation_pct = min(target_allocation_pct, self.max_per_stock)
        
        # Calculate ideal capital to be invested
        target_fiat_exposure = self.net_worth * target_allocation_pct
        current_fiat_exposure = self.holdings * current_price
        
        # Difference dictates buy or sell
        trade_fiat = target_fiat_exposure - current_fiat_exposure
        
        step_reward = 0.0
        
        action_taken = "HOLD"
        quantity = 0.0
        
        # Slippage / Friction
        cost = abs(trade_fiat) * self.transaction_cost
        
        if trade_fiat > 0.001: # Buy threshold to ignore fp noise
            if self.balance >= (trade_fiat + cost):
                shares_bought = trade_fiat / current_price
                
                # Update Avg Cost
                total_cost = (self.holdings * self.avg_buy_price) + trade_fiat
                self.holdings += shares_bought
                self.avg_buy_price = total_cost / self.holdings if self.holdings > 0 else 0
                self.last_buy_step = self.current_step
                self.trade_history.append(1) # Record trade
                
                self.balance -= (trade_fiat + cost)
                action_taken = "BUY"
                quantity = shares_bought
            else:
                trade_fiat = 0 # Invalid
        elif trade_fiat < -0.001: # Sell
            shares_sold = abs(trade_fiat) / current_price
            if shares_sold <= self.holdings:
                # Early Exit Penalty
                if (self.current_step - self.last_buy_step) < self.min_holding_period:
                    step_reward -= self.early_exit_penalty
                
                # Check for profitable trade close bonus
                if current_price > self.avg_buy_price:
                    step_reward += self.trade_close_bonus
                
                self.balance += (abs(trade_fiat) - cost)
                self.holdings -= shares_sold
                self.trade_history.append(1) # Record trade
                action_taken = "SELL"
                quantity = shares_sold
                if self.holdings == 0:
                    self.avg_buy_price = 0.0
                    self.last_buy_step = -999
            else:
                trade_fiat = 0 # Invalid
                
        # Update Net Worth
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + (self.holdings * current_price)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # PnL Step Reward
        if prev_net_worth > 0:
            profit_reward = (self.net_worth - prev_net_worth) / prev_net_worth
        else:
            profit_reward = 0.0
        
        # Proportional Drawdown Penalty: penalty = drawdown * drawdown_penalty
        drawdown = (self.net_worth - self.max_net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0.0
        drawdown_pen = abs(drawdown) * self.drawdown_penalty
        
        step_reward += profit_reward - drawdown_pen
        
        # Profit Incentive for outsized gains (avoid noise threshold)
        if profit_reward > self.profit_threshold:
            step_reward += profit_reward * self.profit_incentive
        
        # HOLD Penalty to discourage inactivity bias
        if action_taken == "HOLD":
            step_reward -= self.hold_penalty
        
        # Stability check for reward
        if np.isnan(step_reward) or np.isinf(step_reward):
            step_reward = -1.0
        
        # Scaled Trade Frequency Penalty
        recent_trades = sum(self.trade_history)
        if recent_trades > self.max_trades_window:
            # Scale penalty by excess trades
            freq_pen = self.overtrade_penalty * (recent_trades - self.max_trades_window)
            step_reward -= freq_pen
        
        self.prev_action = target_allocation_pct
        
        # Clip reward to prevent gradient explosion (standard RL practice)
        step_reward = np.clip(step_reward, -10, 10)
        
        obs = self._next_observation()
        done = bool(self.net_worth <= self.initial_balance * 0.10) # Blowup condition
        
        # Logging hook
        from backend.utils.logger import get_logger
        logger = get_logger(__name__)
        import json
        log_entry = {
            "step": self.current_step,
            "action": action_taken,
            "quantity": float(quantity),
            "price": float(current_price),
            "prev_net_worth": float(prev_net_worth),
            "net_worth": float(self.net_worth),
            "reward": float(step_reward)
        }
        # Only log trades or every 100 steps to keep the file clean
        if action_taken != "HOLD" or self.current_step % 100 == 0:
            logger.info(json.dumps(log_entry))
            
        info = {
            "net_worth": self.net_worth,
            "prev_net_worth": prev_net_worth,
            "action": action_taken,
            "quantity": quantity,
            "price": current_price,
            "step_reward": step_reward
        }
        
        return obs, step_reward, done, False, info
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.avg_buy_price = 0.0
        self.last_buy_step = -999
        self.trade_history.clear()
        return self._next_observation(), {}

    def set_data(self, new_data: pd.DataFrame):
        """
        Updates the internal dataset. Used for multi-stock training.
        """
        self.data = new_data.reset_index(drop=True)
