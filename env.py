import gymnasium as gym
import numpy as np
from communication import CommunicationSystem
from formation import FormationSystem
from defense import DefenseSystem
from attack import AttackSystem
import random

class MultiUAVEnv(gym.Env):
    def __init__(self, config):
        self.num_uavs = config.NUM_UAVS
        self.config = config
        self.comm = CommunicationSystem(self, config)
        self.formation = FormationSystem(self, config)
        self.defense = DefenseSystem(self, config)
        self.attack = AttackSystem(self, config)
        self.gcs_position = np.array(self.formation.center) 
        self.gcs_range = config.GCS_RANGE
        self.current_step = 0
        self.history = []
        self.dt = 1.0
        self.last_attack_step_agent = [None] * self.num_uavs
        self.recovery_steps_history = [[] for _ in range(self.num_uavs)]
        self.recovery_steps_agent = np.zeros(self.num_uavs, dtype=int)

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
        return seed

    def reset(self):
        self.comm.reset()
        self.formation.reset()
        self.attack.reset()
        self.defense.reset()
        self.current_step = 0
        self.history.clear()
        self.last_command_received = np.zeros(self.num_uavs, dtype=bool)
        self.consecutive_attack_steps = 0
        self.last_attack_step_agent = [None] * self.num_uavs
        self.recovery_steps_history = [[] for _ in range(self.num_uavs)]
        self.recovery_steps_agent = np.zeros(self.num_uavs, dtype=int)
        return self._get_obs()

    def step(self, actions):
        # Apply defense actions
        self.defense.apply(actions)
        # Update communication system
        self.comm.update()
        # Update formation system
        self.formation.step()
        # Apply attack actions
        self.attack.step(self.current_step)

        obs = self._get_obs()
        reward_n, info = self._get_reward()

        # Count consecutive attack steps
        attack_penalty = info.get('attack_penalty', 0)
        if isinstance(attack_penalty, (list, np.ndarray)):
            if np.any(np.array(attack_penalty) < 0):
                self.consecutive_attack_steps += 1
            else:
                self.consecutive_attack_steps = 0
        else:
            if attack_penalty < 0:
                self.consecutive_attack_steps += 1
            else:
                self.consecutive_attack_steps = 0

        # Check termination conditions
        done = self._check_done()

        self.current_step += 1
        self.history.append(info)
        return obs, reward_n, done, info

    def _get_obs(self):
        # Get observation for each UAV (communication + formation state)
        obs = []
        for i in range(self.num_uavs):
            obs.append(
                self.comm.get_state(i) +
                self.formation.get_state(i)
            )
        return np.array(obs)

    def _get_reward(self):
        # Calculate reward for each UAV and info dict
        comm = self.comm.get_network_connectivity_per_agent()  # shape: [num_uavs]
        formation = self.formation.get_formation_score_per_agent()  # shape: [num_uavs]
        cost = self.defense.get_cost_per_agent() 
        attack_penalty = self.attack.get_penalty_per_agent()
        speed_penalty = [
            max(0, self.formation.speeds[i] - self.formation.patrol_speed) / 
            (self.formation.max_speed - self.formation.patrol_speed + 1e-8)
            for i in range(self.num_uavs)
        ]
        info = {
            'comm': comm,
            'formation': formation,
            'speed_penalty': speed_penalty,
            'cost': cost,
            'attack_penalty': attack_penalty
        }

        reward_n = []
        for i in range(self.num_uavs):
            reward_n.append(
                self.config.COMM_WEIGHT * comm[i] +
                self.config.FORMATION_WEIGHT * formation[i] +
                self.config.SPEED_PENALTY_WEIGHT * speed_penalty[i] +
                self.config.COST_WEIGHT * cost[i] +
                self.config.ATTACK_PENALTY_WEIGHT * attack_penalty[i]
            )
        return np.array(reward_n), info
    

    def _check_done(self):
        # Terminate if max steps reached
        if self.current_step > self.config.MAX_STEPS:
            return True
        # Terminate if deviation is too large
        if self.current_step > 25 and self.formation.get_deviation() > 0.8:
            return True
        # Terminate if consecutive attack penalty steps exceed threshold
        if self.current_step > 25 and getattr(self, "consecutive_attack_steps", 0) >= self.config.MAX_CONSECUTIVE_ATTACK_STEPS:
            return True
        return False