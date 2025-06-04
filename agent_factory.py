import torch
from dqn_agent import DQNAgent
from actor_critic_agent import ActorCriticAgent
from wright_fisher_agent import WrightFisherAgent
from pg_agent import PolicyGradientAgent
from fed_multi_pg_agent import FedMultiPGAgent

def create_agent(alg, obs_dim, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
    if alg == 'dqn':
        return DQNAgent(obs_dim, config, device)
    elif alg == 'actor_critic':
        return ActorCriticAgent(obs_dim, config, device)
    elif alg == 'wright_fisher':
        return WrightFisherAgent(obs_dim, config)
    elif alg == 'pg':  # 新增
        return PolicyGradientAgent(obs_dim, config, device)
    elif alg == 'fed_multi_pg':
        return FedMultiPGAgent(obs_dim, config, device)
    else:
        raise ValueError(f"Unknown algorithm: {alg}")