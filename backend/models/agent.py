"""
PortfolioManagerAgent — RL PPO Agent using Stable-Baselines3.

STATE SPACE (6 + N_stocks one-hot):
    0: fiat_balance / initial_balance        — Capital utilization [0, 1]
    1: current_exposure                      — % of net worth in current stock [0, 1]
    2: (net_worth - initial) / initial       — Relative P&L [-inf, inf]
    3: Meta_Prob_BUY                         — Meta model's BUY confidence [0, 1]
    4: Volatility_14d / 100                  — Annualized volatility scaled [0, ~5]
    5: Pred_Return_5d                        — Expected 5d forward return [-inf, inf]
    6..N: one-hot stock identity             — Which ticker we are acting on

ACTION SPACE:
    Continuous [0.0, 1.0] — fraction of total fiat to allocate to the stock.
    0.0 = sell all (exit) | 1.0 = deploy all (full concentration)

REWARD FUNCTION:
    step_reward = profit_reward - drawdown_penalty
    + profit_incentive if profit_reward > threshold
    - hold_penalty if no action taken
    - overtrading_penalty if too many trades in window
    + trade_close_bonus if selling at profit
    - early_exit_penalty if holding period < min_holding_period

FALLBACK:
    If RL model file is missing, predict_action falls back to rule-based allocation:
    - If Meta_Prob_BUY >= 0.7: allocate 20%
    - If Meta_Prob_BUY >= 0.55: allocate 10%
    - Else: allocate 0% (HOLD/SKIP)
"""

import os
import yaml
import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from backend.broker.environment import BullRunEnv
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioManagerAgent:
    def __init__(self, data_df, config_path: str = 'configs/prod_params.yaml'):
        self.data_df = data_df
        self.config_path = config_path
        self._rule_based_mode = False

        # Wrap environment
        self.env = DummyVecEnv([lambda: BullRunEnv(self.data_df)])
        self.model = None

        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

    def train(self, stock_data_map: dict, total_timesteps=None, use_extended=False):
        """
        Trains the agent across multiple stocks using uniform random sampling per episode.
        """
        from stable_baselines3.common.callbacks import BaseCallback

        rl_cfg = self.config.get('rl', {})
        if total_timesteps is None:
            total_timesteps = rl_cfg.get('extended_training_steps' if use_extended else 'training_steps', 10000)

        lr = rl_cfg.get('learning_rate', 0.0001)
        ent_coef = rl_cfg.get('ent_coef', 0.01)

        class MultiStockCallback(BaseCallback):
            def __init__(self, env, data_map, verbose=0):
                super().__init__(verbose)
                self.env = env
                self.data_map = data_map
                self.tickers = list(data_map.keys())

            def _on_rollout_start(self) -> None:
                ticker = random.choice(self.tickers)
                new_data = self.data_map[ticker]
                self.env.env_method('set_data', new_data)

            def _on_step(self) -> bool:
                return True

        logger.info(f"Starting Multi-Stock PPO Training for {total_timesteps} steps...")
        initial_ticker = random.choice(list(stock_data_map.keys()))
        self.env = DummyVecEnv([lambda: BullRunEnv(stock_data_map[initial_ticker])])

        self.model = PPO("MlpPolicy", self.env, verbose=0, learning_rate=lr, ent_coef=ent_coef)
        self.model.learn(total_timesteps=total_timesteps, callback=MultiStockCallback(self.env, stock_data_map))
        logger.info("Multi-Stock Training complete.")

    def _get_default_model_path(self):
        models_dir = self.config.get('paths', {}).get('models', 'models')
        return os.path.join(models_dir, "rl", "ppo_portfolio_manager")

    def save(self, path=None):
        save_path = path or self._get_default_model_path()
        if self.model:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save(save_path)
            logger.info(f"RL PPO Model saved to {save_path}.zip")

    def load(self, path=None):
        load_path = path or self._get_default_model_path()
        if not os.path.exists(load_path + ".zip"):
            logger.warning(f"RL-AGENT: Model not found at {load_path}.zip. Switching to rule-based allocation.")
            self._rule_based_mode = True
            return
        self.model = PPO.load(load_path)
        self._rule_based_mode = False
        logger.info(f"RL PPO Model loaded from {load_path}.zip")

    def predict_action_rule_based(self, obs: np.ndarray) -> np.ndarray:
        """
        Fallback allocation logic when RL model is not available.
        Uses Meta_Prob_BUY (index 3 in the core observation) to decide size.

        Returns:
            np.array([allocation_fraction]) — same interface as RL model.
        """
        meta_prob_buy = float(obs[3]) if len(obs) > 3 else 0.0

        if meta_prob_buy >= 0.70:
            allocation = 0.20  # High confidence: up to 20%
        elif meta_prob_buy >= 0.60:
            allocation = 0.15  # Medium confidence
        elif meta_prob_buy >= 0.55:
            allocation = 0.10  # Low confidence threshold
        else:
            allocation = 0.0   # Below threshold: skip

        logger.debug(f"RULE-BASED: Meta={meta_prob_buy:.3f} => Alloc={allocation:.2f}")
        return np.array([allocation], dtype=np.float32)

    def predict_action(self, obs):
        if self._rule_based_mode or self.model is None:
            return self.predict_action_rule_based(obs)
        action, _states = self.model.predict(obs, deterministic=True)
        return action
