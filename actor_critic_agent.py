'The code of ID-HAM adapted for Multi-UAV Security'
'Doi: 10.1109/TIFS.2023.3314219'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agent_base import BaseAgent

class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(64, act_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.net(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

class ActorCriticAgent(BaseAgent):
    def __init__(self, obs_dim, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.obs_dim = obs_dim
        self.act_dim = 4
        self.device = device
        self.gamma = config.GAMMA
        self.lr = 1e-3
        self.buffer = []

        # Action definitions: [hop, route, switch]
        self.actions = [
            [None, None, None],      # 0: No action
            ['hop', None, None],     # 1: Frequency hopping
            [None, 'route', None],   # 2: Routing
            [None, None, 'switch']   # 3: Switch leader
        ]

        self.net = ActorCriticNet(obs_dim, self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def select_action(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        logits, value = self.net(obs_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def decode_action(self, action_idx):
        if 0 <= action_idx < len(self.actions):
            return self.actions[action_idx]
        else:
            return [None, None, None]

    def store(self, obs, action, logprob, reward, next_obs, done, value):
        # Store one step transition, no history buffer
        self.buffer.append((obs, action, logprob, reward, next_obs, done, value))

    def update(self):
        if len(self.buffer) == 0:
            return
        # Unpack all transitions
        obs, action, logprob, reward, next_obs, done, value = zip(*self.buffer)
        self.buffer.clear()  # Clear buffer for next episode

        obs_tensor = torch.FloatTensor(np.array(obs)).to(self.device)          
        next_obs_tensor = torch.FloatTensor(np.array(next_obs)).to(self.device)
        action_tensor = torch.LongTensor(action).to(self.device)
        value_tensor = torch.FloatTensor(value).to(self.device)
        reward_tensor = torch.FloatTensor(reward).to(self.device)
        done_tensor = torch.FloatTensor(done).to(self.device)

        # Compute target and advantage
        with torch.no_grad():
            _, next_value = self.net(next_obs_tensor[-1].unsqueeze(0))
            next_value = next_value.item()
        targets = []
        for r, d in zip(reversed(reward), reversed(done)):
            next_value = r + self.gamma * next_value * (1 - d)
            targets.insert(0, next_value)
        target_tensor = torch.FloatTensor(targets).to(self.device)
        advantage = target_tensor - value_tensor

        logits, value_pred = self.net(obs_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_logprob = dist.log_prob(action_tensor)

        actor_loss = -(new_logprob * advantage.detach()).mean()
        critic_loss = nn.MSELoss()(value_pred.view(-1), target_tensor)
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()