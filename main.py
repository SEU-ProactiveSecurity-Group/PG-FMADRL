import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
from env import MultiUAVEnv
from config import Config
from federated import federated_average
import random
import numpy as np
import torch
from agent_factory import create_agent
import copy


def evaluate_on_val_env(agents, val_env, num_episodes=3):
    agents_eval = [copy.deepcopy(agent) for agent in agents]
    for agent in agents_eval:
        if hasattr(agent, "policy"):
            agent.policy.eval()
        if hasattr(agent, "net"):
            agent.net.eval()
    total_reward = 0
    for _ in range(num_episodes):
        obs_n = val_env.reset()
        done = False
        ep_reward = 0
        while not done:
            actions = [agent.decode_action(agent.select_action(obs_n[i])[0]) for i, agent in enumerate(agents_eval)]
            next_obs_n, reward_n, done, info = val_env.step(actions)
            ep_reward += np.mean(reward_n)
            obs_n = next_obs_n
        total_reward += ep_reward
    return total_reward / num_episodes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_uavs', type=int, default=Config.NUM_UAVS, help='Number of UAVs')
    parser.add_argument('--attacker_type', type=str, default=Config.ATTACKER_TYPE, help='Attacker type')
    parser.add_argument('--attack_type', type=str, default=Config.ATTACK_TYPE, help='Attack type')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--episode', type=int, default=Config.EPISODE, help='Number of episodes')
    parser.add_argument('--algorithm', type=str, default=Config.ALGORITHM, help='Algorithm')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set config parameters from args
    Config.NUM_UAVS = args.num_uavs
    Config.ATTACKER_TYPE = args.attacker_type
    Config.ATTACK_TYPE = args.attack_type
    Config.EPISODE = args.episode
    Config.ALGORITHM = args.algorithm

    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    env = MultiUAVEnv(Config())
    env.seed(seed)
    num_agents = Config.NUM_UAVS
    episode = Config.EPISODE

    epsilon_decay = Config.EPSILON_DECAY
    min_epsilon = Config.MIN_EPSILON

    obs_dim = len(env._get_obs()[0])

    alg = Config.ALGORITHM
    agents = [create_agent(alg, obs_dim, Config()) for _ in range(num_agents)]

    FED_ROUND = Config.FED_ROUND
    episode_rewards = []
    episode_costs = []
    reward_parts = {
        'comm': [],
        'formation': [],
        'speed_penalty': [],
        'cost': [],
        'attack_penalty': [],
        'robustness_reward': []
    }

    best_avg_reward = -float('inf')
    val_env = MultiUAVEnv(Config())
    val_env.seed(seed + 100)
    best_val_reward = -float('inf')

    for episode in range(episode):
        obs_n = env.reset()
        done = False
        ep_reward = 0
        ep_cost = np.zeros(num_agents)
        ep_parts = {k: 0 for k in reward_parts}
        step_count = 0
        while not done:
            actions_raw = [agent.select_action(obs_n[i]) for i, agent in enumerate(agents)]
            if isinstance(actions_raw[0], tuple):
                actions = [a[0] for a in actions_raw]
                logprobs = [a[1] for a in actions_raw]
                values = [a[2] for a in actions_raw]
            else:
                actions = actions_raw
                logprobs = [0 for _ in actions_raw]
                values = [0 for _ in actions_raw]

            decoded_actions = [agent.decode_action(a) for agent, a in zip(agents, actions)]
            next_obs_n, reward_n, done, info = env.step(decoded_actions)
            for i, agent in enumerate(agents):
                agent_name = agent.__class__.__name__.lower()
                if hasattr(agent, "store") and ("ppo" in agent_name or "actorcritic" in agent_name or 'dqnppo' in agent_name):
                    agent.store(obs_n[i], actions[i], logprobs[i], reward_n[i], next_obs_n[i], done, values[i])
                elif hasattr(agent, "store") and ("pg" in agent_name or "policygradient" in agent_name):
                    agent.store(obs_n[i], actions[i], reward_n[i], next_obs_n[i], done)
                elif hasattr(agent, "store") and ("dqn" in agent_name):
                    agent.store(obs_n[i], actions[i], reward_n[i], next_obs_n[i], done)
                else:
                    agent.store(obs_n[i], actions[i], reward_n[i], next_obs_n[i], done)
                agent.update()
            obs_n = next_obs_n
            ep_reward += reward_n.mean()
            costs = info.get('cost', np.zeros(num_agents))
            ep_cost += costs
            for k in ep_parts:
                v = info.get(k, 0)
                if isinstance(v, list) or isinstance(v, np.ndarray):
                    ep_parts[k] += np.mean(v)
                else:
                    ep_parts[k] += v
            step_count += 1

        episode_rewards.append(ep_reward)
        episode_costs.append(ep_cost)
        for k in reward_parts:
            reward_parts[k].append(ep_parts[k] / step_count if step_count > 0 else 0)
        
        # Print every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_cost = np.mean(episode_costs[-10:])
            print(f"Episode {episode+1}, Avg Reward (last 10): {avg_reward:.3f}, Avg Defense Cost: {avg_cost:.3f}")

            # Evaluate on validation environment
            val_reward = evaluate_on_val_env(agents, val_env, num_episodes=5)
            print(f"Episode {episode+1}, Validation Reward: {val_reward:.3f}")

            # Save best model
            if val_reward > best_val_reward and avg_reward > best_avg_reward:
                best_val_reward = val_reward
                best_avg_reward = avg_reward
                for i, agent in enumerate(agents):
                    agent_name = agent.__class__.__name__.lower()
                    model_path = os.path.join(MODEL_DIR, f"{agent_name}_seed{seed}_{Config.ATTACK_TYPE}_attacker_{Config.ATTACKER_TYPE}_uav{num_agents}_[{i}]_best.pth")
                    if "fed" in agent_name and "pg" in agent_name:
                        torch.save(agent.policy.state_dict(), model_path)
                    elif "dqn" in agent_name:
                        torch.save(agent.policy_net.state_dict(), model_path)
                    elif "actorcritic" in agent_name:
                        torch.save(agent.net.state_dict(), model_path)
                    elif "policygradient" in agent_name or "pg" in agent_name:
                        torch.save(agent.policy.state_dict(), model_path)
                    elif "wrightfisher" in agent_name:
                        np.savez(model_path.replace('.pth', '.npz'),
                                strategy_probs=agent.strategy_probs,
                                payoff_history=agent.payoff_history)
                # Add other agent types as needed
        # Federated averaging every FED_ROUND episodes
        if (episode + 1) % FED_ROUND == 0:
            federated_average(agents)
