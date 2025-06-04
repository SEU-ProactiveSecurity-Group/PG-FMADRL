'The code of DESOLATER adapted for Multi-UAV Security'
'Doi: 10.1109/ACCESS.2021.3076599'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agent_base import BaseAgent
from collections import deque

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
    def forward(self, x):
        return self.net(x)  # Output logits

class PGBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    def push(self, *transition):
        self.buffer.append(tuple(transition))
    def sample(self):
        batch = list(self.buffer)
        self.buffer.clear()
        return map(np.array, zip(*batch))
    def __len__(self):
        return len(self.buffer)

class PolicyGradientAgent(BaseAgent):
    def __init__(self, obs_dim, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.obs_dim = obs_dim
        # Action definitions: [hop, route, switch]
        self.actions = [
            [None, None, None],      # 0: No action
            ['hop', None, None],     # 1: Frequency hopping
            [None, 'route', None],   # 2: Routing
            [None, None, 'switch']   # 3: Switch leader
        ]
        self.act_dim = len(self.actions)
        self.device = device
        self.gamma = config.GAMMA
        self.lr = 5e-4
        self.policy = PolicyNet(obs_dim, self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.buffer = PGBuffer(getattr(config, "MEMORY_SIZE", 10000))

    def select_action(self, obs, noise=True):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        logits = self.policy(obs)
        probs = torch.softmax(logits, dim=1).cpu().detach().numpy()[0]
        probs = np.clip(probs, 1e-8, 1)  # Prevent zero or NaN
        probs = probs / probs.sum()      # Normalize
        action_idx = np.random.choice(self.act_dim, p=probs)
        return action_idx, np.log(probs[action_idx]), None

    def decode_action(self, action_idx):
        # Convert action index to action meaning
        return self.actions[action_idx]

    def store(self, obs, action, reward, next_obs, done, logprob=None):
        # logprob is optional for compatibility with PPO, etc.
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self):
        if len(self.buffer) == 0:
            return
        obs, action, reward, next_obs, done = self.buffer.sample()
        obs = torch.FloatTensor(obs).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = np.array(reward)
        # Compute discounted returns
        returns = []
        G = 0
        for r, d in zip(reversed(reward), reversed(done)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        # Normalize returns (only if more than 1 sample and std > 0)
        if len(returns) > 1 and returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        logits = self.policy(obs)
        log_probs = torch.log_softmax(logits, dim=1)
        chosen_log_probs = log_probs[range(len(action)), action]
        loss = -(chosen_log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()