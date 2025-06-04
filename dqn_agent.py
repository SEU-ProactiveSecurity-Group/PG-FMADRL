'The code of RE-MTD adapted for Multi-UAV Security'
'Doi: 10.1109/TNSM.2024.3413685'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from agent_base import BaseAgent

class QNetwork(nn.Module):
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
        return self.net(x)

class DQNAgent(BaseAgent):
    def __init__(self, obs_dim, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.obs_dim = obs_dim
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.001
        self.gamma = 0.99
        self.lr = 5e-4
        self.batch_size = 128
        self.device = device

        # Action definitions: [hop, route, switch]
        self.actions = [
            [None, None, None],      # 0: No action
            ['hop', None, None],     # 1: Frequency hopping
            [None, 'route', None],   # 2: Routing
            [None, None, 'switch']   # 3: Switch leader
        ]
        self.act_dim = len(self.actions)

        self.policy_net = QNetwork(obs_dim, self.act_dim).to(device)
        self.target_net = QNetwork(obs_dim, self.act_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = deque(maxlen=20000)
        self.update_count = 0
        self.target_update_freq = 200

    def select_action(self, obs):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(obs_tensor)
        action_idx = q_values.argmax().item()
        return action_idx

    def decode_action(self, action_idx):
        # Convert action index to action meaning
        if 0 <= action_idx < len(self.actions):
            return self.actions[action_idx]
        else:
            return [None, None, None]

    def store(self, obs, action, reward, next_obs, done):
        # Store transition in replay buffer
        self.memory.append((obs, action, reward, next_obs, done))

    def update(self):
        # Only update if enough samples in memory
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        obs = np.array(obs)
        next_obs = np.array(next_obs)
        reward = np.array(reward)
        done = np.array(done)
        action = np.array(action)  # Here action is an integer index

        assert (action >= 0).all() and (action < self.act_dim).all(), f"Action index out of range! max={action.max()}, act_dim={self.act_dim}"

        obs = torch.FloatTensor(obs).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # Compute Q(s,a)
        q_values = self.policy_net(obs).gather(1, action)
        with torch.no_grad():
            # Compute target Q value
            next_q = self.target_net(next_obs).max(1, keepdim=True)[0]
            target = reward + self.gamma * next_q * (1 - done)
        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        # Update target network
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        # Automatically decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)