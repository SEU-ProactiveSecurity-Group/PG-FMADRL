import numpy as np

class DefenseSystem:
    def __init__(self, env, config):
        self.env = env
        self.num_uavs = config.NUM_UAVS
        self.num_nodes = self.num_uavs + 1  # +1 for ground station
        # [freq_hop, route, leader] cooldowns
        self.cooldowns = np.zeros(3)
        # Flags and timers for actions in progress
        self.freq_hop_in_progress = False
        self.freq_hop_timer = 0
        self.route_change_in_progress = False
        self.route_change_timer = 0
        self.leader_change_in_progress = False
        self.leader_change_timer = 0
        self.pending_freq = None
        self.pending_leader = None
        self.pending_route = None  # (target, relay)
        self.cost = 0.0
        self.last_cost_agents = np.zeros(self.num_uavs)  # Record last step cost for each UAV

    def reset(self):
        self.cooldowns = np.zeros(3)
        self.freq_hop_in_progress = False
        self.route_change_in_progress = False
        self.leader_change_in_progress = False
        self.freq_hop_timer = 0
        self.route_change_timer = 0
        self.leader_change_timer = 0
        self.pending_freq = None
        self.pending_leader = None
        self.pending_route = None
        self.cost = 0.0
        self.last_cost_agents = np.zeros(self.num_uavs)

    def apply(self, actions):
        # actions: num_agents x 3
        self.cooldowns = np.maximum(0, self.cooldowns - 1)
        leader = self.env.formation.leader
        current_freq = self.env.comm.frequencies[leader]
        action_executed = False
        self.last_cost_agents = np.zeros(self.num_uavs)  # Reset every step

        for act in actions:
            if action_executed:
                break
            # Frequency hopping (only execute if freq is not current)
            if act[0] == 'hop' and not self.freq_hop_in_progress and self.cooldowns[0] <= 0:
                available_freqs = [f for f in range(self.env.comm.num_channels) if f != current_freq]
                if available_freqs:
                    chosen_freq = np.random.choice(available_freqs)
                    self.freq_hop_in_progress = True
                    self.freq_hop_timer = 1
                    self.pending_freq = chosen_freq
                    self.cost += self.num_uavs  # Total cost equals number of UAVs
                    self.last_cost_agents += np.ones(self.num_uavs) * 1.0  
                    action_executed = True
                    break
            # Route change (target cannot be leader, relay cannot be leader or target)
            if act[1] == 'route' and not self.route_change_in_progress and self.cooldowns[1] <= 0:
                num_uavs = self.num_uavs
                leader = self.env.formation.leader
                positions = self.env.formation.get_positions()
                # Only try to set relay for attacked links
                for (i, j) in self.env.attack.link_attack_timer:
                    if self.env.attack.is_link_attack_effective(i, j):
                        # Only consider leader-to-member attacked links
                        if i == leader and j != leader and j != self.num_nodes - 1:
                            target = j
                        elif j == leader and i != leader and i != self.num_nodes - 1:
                            target = i
                        else:
                            continue
                        # If leader and target can communicate directly, skip (no relay needed)
                        if self.env.comm._can_communicate(leader, target, positions):
                            continue
                        # Find a node that can be used as relay
                        relay = None
                        for candidate in range(num_uavs):
                            if candidate != leader and candidate != target:
                                if self.env.comm._can_communicate(leader, candidate, positions) and self.env.comm._can_communicate(candidate, target, positions):
                                    relay = candidate
                                    break
                        if relay is not None:
                            self.pending_route = (target, relay)
                            self.route_change_in_progress = True
                            self.route_change_timer = 1
                            self.cost += 1
                            self.last_cost_agents[target] += 0.5
                            self.last_cost_agents[relay] += 0.5
                            action_executed = True
                            break
            # Leader switch (new_leader cannot be current leader)
            if act[2] == 'switch' and not self.leader_change_in_progress and self.cooldowns[2] <= 0:
                attacked_nodes = set()
                for (i, j) in self.env.attack.link_attack_timer:
                    if self.env.attack.is_link_attack_effective(i, j):
                        attacked_nodes.add(i)
                        attacked_nodes.add(j)
                # Candidates cannot be current leader or endpoints of attacked links
                candidates = [i for i in range(self.num_uavs) if i != leader and i not in attacked_nodes]
                if candidates:
                    new_leader = np.random.choice(candidates)
                    self.leader_change_in_progress = True
                    self.leader_change_timer = 1
                    self.pending_leader = new_leader
                    self.cost += 2
                    self.last_cost_agents[leader] += 1.0
                    self.last_cost_agents[new_leader] += 1.0
                    action_executed = True
                    break
        # Timers for actions in progress
        if self.freq_hop_in_progress:
            self.freq_hop_timer -= 1
            if self.freq_hop_timer <= 0:
                for i in range(self.num_nodes):
                    self.env.comm.frequencies[i] = self.pending_freq
                self.freq_hop_in_progress = False
                self.cooldowns[0] = 5
        if self.route_change_in_progress:
            self.route_change_timer -= 1
            if self.route_change_timer <= 0:
                target, relay = self.pending_route
                self.env.comm.set_route(target, relay)
                self.route_change_in_progress = False
                self.cooldowns[1] = 3
        if self.leader_change_in_progress:
            self.leader_change_timer -= 1
            if self.leader_change_timer <= 0:
                self.env.formation.leader = self.pending_leader
                self.leader_change_in_progress = False
                self.cooldowns[2] = 5
                self.env.comm.update()

    def get_cost(self):
        c = self.cost
        self.cost = 0.0
        return c

    def get_cost_per_agent(self):
        c = self.last_cost_agents.copy()
        self.last_cost_agents = np.zeros(self.num_uavs)
        return c
    
    def get_state(self):
        # Return normalized cooldowns for each action
        return list(self.cooldowns / np.array([2, 2, 5]))