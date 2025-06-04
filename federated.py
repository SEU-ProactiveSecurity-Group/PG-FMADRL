import numpy as np
from fed_multi_pg_agent import FedMultiPGAgent
from config import Config

def federated_average(agents):
    # Only aggregate FedMultiPGAgent
    shared_agents = [agent for agent in agents if isinstance(agent, FedMultiPGAgent)]
    if not shared_agents:
        return
    
    state_dicts = [agent.policy.state_dict() for agent in shared_agents]

    # Weighted average by reward
    rewards = np.array([getattr(agent, "avg_recent_reward", 1.0) for agent in shared_agents])
    rewards = np.clip(rewards, 1e-6, None)
    weights = rewards / rewards.sum()
    avg_state_dict = {}
    for key in state_dicts[0]:
        avg_state_dict[key] = sum(w * sd[key] for w, sd in zip(weights, state_dicts))

    # Only aggregate shared layers ('shared'), do not aggregate output layers ('output')
    for key in state_dicts[0]:
        if key.startswith('shared'):
            avg_state_dict[key] = sum([sd[key] for sd in state_dicts]) / len(state_dicts)
    for agent in shared_agents:
        agent_state = agent.policy.state_dict()
        # Update shared layer parameters
        for key in avg_state_dict:
            agent_state[key] = avg_state_dict[key]
        agent.policy.load_state_dict(agent_state)
        # Output layer parameters remain local
        local_finetune_steps = Config.LOCAL_FINETUNE_STEPS
        for _ in range(local_finetune_steps):
            agent.update()