'The code of WF-MTD adapted for Multi-UAV Security'
'Doi: 10.1109/TDSC.2022.3232537'

import numpy as np
from agent_base import BaseAgent
from collections import deque

class WrightFisherAgent(BaseAgent):
    def __init__(self, obs_dim, config, action_list=None, population_size=500, mutation_rate=0.4, alpha=0.3, prob_smooth=0.8, window_size=100):
        self.obs_dim = obs_dim
        self.act_dim = 4 if action_list is None else len(action_list)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.alpha = alpha  # Smoothing coefficient for moving average
        self.prob_smooth = prob_smooth  # Probability smoothing coefficient
        self.actions = action_list or [
            [None, None, None],      # 0: No action
            ['hop', None, None],     # 1: Frequency hopping
            [None, 'route', None],   # 2: Routing
            [None, None, 'switch']   # 3: Switch leader
        ]
        self.strategy_probs = np.ones(self.act_dim) / self.act_dim
        self.payoff_history = np.zeros(self.act_dim)
        self.last_action = None
        self.last_reward = None
        # Each action has its own sliding window for rewards
        self.reward_windows = [deque(maxlen=window_size) for _ in range(self.act_dim)]

    def select_action(self, obs):
        action_idx = np.random.choice(self.act_dim, p=self.strategy_probs)
        self.last_action = action_idx
        return action_idx, self.strategy_probs[action_idx], None

    def decode_action(self, action_idx):
        return self.actions[action_idx]

    def store(self, obs, action, reward, next_obs, done):
        self.last_action = action
        self.last_reward = reward
        self.reward_windows[action].append(reward)

    def update(self):
        # Use the mean of the sliding window as the payoff for each action
        for i in range(self.act_dim):
            if len(self.reward_windows[i]) > 0:
                self.payoff_history[i] = np.mean(self.reward_windows[i])
            else:
                self.payoff_history[i] = 0

        # Multi-elite retention: boost the probability of the top-k actions
        k = 2  # Number of elite actions
        elite_boost = 0.2  # Boost amount for each elite action
        elite_indices = np.argsort(self.payoff_history)[-k:]
        for idx in elite_indices:
            self.strategy_probs[idx] += elite_boost
        self.strategy_probs /= self.strategy_probs.sum()

        # Evolution step (shift + scale + softmax)
        fitness = self.payoff_history.copy()
        min_fit = fitness.min()
        fitness = (fitness - min_fit + 1.0) * 5
        fitness = np.exp(fitness)
        fitness = fitness / fitness.sum()

        new_counts = np.random.multinomial(self.population_size, fitness)
        new_probs = new_counts / self.population_size

        # Probability smoothing + mutation
        new_probs = (1 - self.mutation_rate) * new_probs + self.mutation_rate / self.act_dim
        self.strategy_probs = (
            self.prob_smooth * self.strategy_probs + (1 - self.prob_smooth) * (new_probs / new_probs.sum())
        )

        # Minimum probability threshold
        min_prob = 0.05
        self.strategy_probs = np.clip(self.strategy_probs, min_prob, 1.0)
        self.strategy_probs /= self.strategy_probs.sum()

    def reset(self):
        self.strategy_probs = np.ones(self.act_dim) / self.act_dim
        self.payoff_history = np.zeros(self.act_dim)
        self.last_action = None
        self.last_reward = None