import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from agent_base import BaseAgent

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

class FedMultiPGAgent(BaseAgent):
    def __init__(self, obs_dim, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.obs_dim = obs_dim
        self.device = device
        self.gamma = config.GAMMA
        self.lr = config.LR
        self.batch_size = config.BATCH_SIZE
        # Action definitions: [hop, route, switch]
        self.actions = [
            [None, None, None],      # 0: No action
            ['hop', None, None],     # 1: Frequency hopping
            [None, 'route', None],   # 2: Routing
            [None, None, 'switch']   # 3: Switch leader
        ]
        self.act_dim = len(self.actions)
        self.policy = PolicyNet(obs_dim, self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        self.reward_window = deque(maxlen=20)
        self.last_reward = 0.0

    def select_action(self, obs, noise=True):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        logits = self.policy(obs)
        probs = torch.softmax(logits, dim=1).cpu().detach().numpy()[0]
        probs = np.clip(probs, 1e-8, 1)
        probs = probs / probs.sum()
        action_idx = np.random.choice(self.act_dim, p=probs)
        return action_idx, np.log(probs[action_idx]), None

    def decode_action(self, action_idx):
        # Convert action index to action meaning
        if 0 <= action_idx < len(self.actions):
            return self.actions[action_idx]
        else:
            return [None, None, None]

    def store(self, obs, action, reward, next_obs, done, logprob=None):
        # Store transition in memory
        self.memory.append((obs, action, reward, next_obs, done))
        alpha = 0.1
        if isinstance(self.last_reward, np.ndarray):
            self.last_reward = (1 - alpha) * self.last_reward + alpha * np.array(reward)
        else:
            self.last_reward = (1 - alpha) * self.last_reward + alpha * reward
        self.reward_window.append(np.mean(reward))

    @property
    def avg_recent_reward(self):
        # Return average reward of recent episodes
        if len(self.reward_window) == 0:
            return 0.0
        return np.mean(self.reward_window)

    def update(self):
        # Update policy using all transitions in memory
        if len(self.memory) == 0:
            return
        batch = list(self.memory)
        self.memory.clear()
        obs, action, reward, next_obs, done = zip(*batch)
        obs = torch.FloatTensor(np.array(obs)).to(self.device)
        action = torch.LongTensor(np.array(action)).to(self.device)
        reward = np.array(reward)
        done = np.array(done)
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
            # returns = returns - returns.mean()

        logits = self.policy(obs)
        log_probs = torch.log_softmax(logits, dim=1)
        chosen_log_probs = log_probs[range(len(action)), action]
        # loss = -(chosen_log_probs * returns).mean()
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        loss = -(chosen_log_probs * returns).mean() - 0.05 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()