import numpy as np
import networkx as nx

class AttackSystem:
    def __init__(self, env, config):
        self.env = env
        self.num_nodes = config.NUM_UAVS + 1
        self.ground = self.num_nodes - 1
        self.attack_type = config.ATTACK_TYPE  # 'link' or 'node'
        self.attacker_type = config.ATTACKER_TYPE  # 'random' or 'greedy' or 'fixed'
        self.attack_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=bool)
        self.node_attack = np.zeros(self.num_nodes, dtype=bool)
        self.link_attack_timer = {}
        self.link_attack_freq = {}
        self.link_cooldown_timer = {}
        self.node_attack_timer = {}
        self.node_attack_freq = {}
        self.random_link_freqs = {}
        self.random_node_freqs = {}
        self.node_cooldown_timer = {}
        self.ATTACK_DURATION = 15
        self.ATTACK_COOLDOWN = 5
        self.random_link_targets = []
        self.random_node_targets = []
        self.fixed_link_target = None
        self.fixed_link_freq = None
        self.fixed_node_target = None
        self.fixed_node_freq = None

    def reset(self):
        self.attack_matrix[:] = False
        self.node_attack[:] = False
        self.link_attack_timer.clear()
        self.link_attack_freq.clear()
        self.link_cooldown_timer.clear()
        self.node_attack_timer.clear()
        self.node_attack_freq.clear()
        self.node_cooldown_timer.clear()
        self.random_link_freqs = {}
        self.random_node_freqs = {}
        self.random_link_targets = []
        self.random_node_targets = []
        self.fixed_link_target = None
        self.fixed_link_freq = None
        self.fixed_node_target = None
        self.fixed_node_freq = None

        # Fixed attacker: only select one target
        if self.attacker_type == 'fixed':
            if self.attack_type == 'link':
                new_targets = self._select_link_target_random()
                if new_targets:
                    key = (min(new_targets[0][0], new_targets[0][1]), max(new_targets[0][0], new_targets[0][1]))
                    freq_i = self.env.comm.frequencies[new_targets[0][0]]
                    freq_j = self.env.comm.frequencies[new_targets[0][1]]
                    if freq_i == freq_j:
                        self.fixed_link_target = key
                        self.fixed_link_freq = freq_i
                        self.link_attack_timer[key] = self.ATTACK_DURATION
                        self.link_attack_freq[key] = freq_i
            if self.attack_type == 'node':
                new_targets = self._select_node_target_random()
                if new_targets:
                    self.fixed_node_target = new_targets[0]
                    self.fixed_node_freq = self.env.comm.frequencies[self.fixed_node_target]
                    self.node_attack_timer[self.fixed_node_target] = self.ATTACK_DURATION
                    self.node_attack_freq[self.fixed_node_target] = self.fixed_node_freq
        
        # Random attacker: only select one target
        if self.attacker_type == 'random':
            if self.attack_type == 'link':
                busy_links = set()
                new_targets = self._select_link_target_random(exclude=busy_links)
                for new_target in new_targets:
                    key = (min(new_target[0], new_target[1]), max(new_target[0], new_target[1]))
                    freq_i = self.env.comm.frequencies[new_target[0]]
                    freq_j = self.env.comm.frequencies[new_target[1]]
                    if freq_i == freq_j:
                        self.link_attack_timer[key] = self.ATTACK_DURATION
                        self.link_attack_freq[key] = freq_i
            if self.attack_type == 'node':
                busy_nodes = set()
                new_targets = self._select_node_target_random(exclude=busy_nodes)
                for new_target in new_targets:
                    self.node_attack_timer[new_target] = self.ATTACK_DURATION
                    self.node_attack_freq[new_target] = self.env.comm.frequencies[new_target]

    def step(self, step_count):
        self.attack_matrix[:] = False
        self.node_attack[:] = False

        if self.attacker_type == 'fixed':
            if self.attack_type == 'link' and self.fixed_link_target is not None:
                key = self.fixed_link_target
                # Decrease timer
                if key in self.link_attack_timer and self.link_attack_timer[key] > 0:
                    self.link_attack_timer[key] -= 1
                    if self.link_attack_timer[key] <= 0:
                        self.link_attack_timer[key] = 0
                        self.link_cooldown_timer[key] = self.ATTACK_COOLDOWN
                # After cooldown, resume attack automatically
                if key in self.link_cooldown_timer:
                    self.link_cooldown_timer[key] -= 1
                    if self.link_cooldown_timer[key] <= 0:
                        del self.link_cooldown_timer[key]
                        self.link_attack_timer[key] = self.ATTACK_DURATION
                        self.link_attack_freq[key] = self.fixed_link_freq

            if self.attack_type == 'node' and self.fixed_node_target is not None:
                k = self.fixed_node_target
                if k in self.node_attack_timer and self.node_attack_timer[k] > 0:
                    self.node_attack_timer[k] -= 1
                    if self.node_attack_timer[k] <= 0:
                        self.node_attack_timer[k] = 0
                        self.node_cooldown_timer[k] = self.ATTACK_COOLDOWN
                if k in self.node_cooldown_timer:
                    self.node_cooldown_timer[k] -= 1
                    if self.node_cooldown_timer[k] <= 0:
                        del self.node_cooldown_timer[k]
                        self.node_attack_timer[k] = self.ATTACK_DURATION
                        self.node_attack_freq[k] = self.fixed_node_freq
        
        if self.attacker_type == 'random':
            if self.attack_type == 'link':
                # Decrease timer and handle cooldown
                expired_links = [k for k in self.link_attack_timer if self.link_attack_timer[k] > 0]
                for k in expired_links:
                    self.link_attack_timer[k] -= 1
                    if self.link_attack_timer[k] <= 0:
                        self.link_attack_timer[k] = 0
                        self.link_cooldown_timer[k] = self.ATTACK_COOLDOWN
                expired_cooldown = [k for k in self.link_cooldown_timer if self.link_cooldown_timer[k] > 0]
                for k in expired_cooldown:
                    self.link_cooldown_timer[k] -= 1
                    if self.link_cooldown_timer[k] <= 0:
                        del self.link_cooldown_timer[k]
                        # After cooldown, randomly select a new target
                        busy_links = set(self.link_attack_timer.keys()) | set(self.link_cooldown_timer.keys())
                        new_targets = self._select_link_target_random(exclude=busy_links)
                        for new_target in new_targets:
                            key = (min(new_target[0], new_target[1]), max(new_target[0], new_target[1]))
                            self.link_attack_timer[key] = self.ATTACK_DURATION
                            # Assign frequency only if both ends have the same frequency, otherwise skip
                            freq_i = self.env.comm.frequencies[new_target[0]]
                            freq_j = self.env.comm.frequencies[new_target[1]]
                            if freq_i == freq_j:
                                self.link_attack_freq[key] = freq_i
                            else:
                                self.link_attack_freq[key] = freq_i  # or freq_j, adjust as needed

            if self.attack_type == 'node':
                # Decrease timer and handle cooldown
                expired_nodes = [k for k in self.node_attack_timer if self.node_attack_timer[k] > 0]
                for k in expired_nodes:
                    self.node_attack_timer[k] -= 1
                    if self.node_attack_timer[k] <= 0:
                        self.node_attack_timer[k] = 0
                        self.node_cooldown_timer[k] = self.ATTACK_COOLDOWN
                expired_cooldown = [k for k in self.node_cooldown_timer if self.node_cooldown_timer[k] > 0]
                for k in expired_cooldown:
                    self.node_cooldown_timer[k] -= 1
                    if self.node_cooldown_timer[k] <= 0:
                        del self.node_cooldown_timer[k]
                        # After cooldown, randomly select a new target
                        busy_nodes = set(self.node_attack_timer.keys()) | set(self.node_cooldown_timer.keys())
                        new_targets = self._select_node_target_random(exclude=busy_nodes)
                        for new_target in new_targets:
                            self.node_attack_timer[new_target] = self.ATTACK_DURATION
                            self.node_attack_freq[new_target] = self.env.comm.frequencies[new_target]

        elif self.attacker_type == 'greedy':
            ground = self.ground

            if self.attack_type == 'node':
                # 1. Decrease attack timer, collect targets just entering cooldown
                to_cooldown = []
                for k in list(self.node_attack_timer):
                    self.node_attack_timer[k] -= 1
                    if self.node_attack_timer[k] <= 0:
                        self.node_attack_timer[k] = 0
                        self.node_cooldown_timer[k] = self.ATTACK_COOLDOWN
                        to_cooldown.append(k)
                # Remove targets that have entered cooldown
                for k in to_cooldown:
                    del self.node_attack_timer[k]
                    if k in self.node_attack_freq:
                        del self.node_attack_freq[k]

                # 2. Decrease cooldown timer, collect targets just finished cooldown
                to_reattack = []
                for k in list(self.node_cooldown_timer):
                    self.node_cooldown_timer[k] -= 1
                    if self.node_cooldown_timer[k] <= 0:
                        to_reattack.append(k)
                # Remove targets that have finished cooldown
                for k in to_reattack:
                    del self.node_cooldown_timer[k]

                # 3. Only attack the leader node
                leader = self.env.formation.get_leader()
                busy_nodes = set(self.node_attack_timer.keys()) | set(self.node_cooldown_timer.keys())
                if leader != ground and leader not in busy_nodes:
                    self.node_attack_timer[leader] = self.ATTACK_DURATION
                    self.node_attack_freq[leader] = self.env.comm.frequencies[leader]

                # 4. Update node_attack
                for target in self.node_attack_timer:
                    self.node_attack[target] = True

            elif self.attack_type == 'link':
                # 1. Decrease attack timer, collect links just entering cooldown
                to_cooldown = []
                for k in list(self.link_attack_timer):
                    self.link_attack_timer[k] -= 1
                    if self.link_attack_timer[k] <= 0:
                        self.link_attack_timer[k] = 0
                        self.link_cooldown_timer[k] = self.ATTACK_COOLDOWN
                        to_cooldown.append(k)
                # Remove links that have entered cooldown
                for k in to_cooldown:
                    del self.link_attack_timer[k]
                    if k in self.link_attack_freq:
                        del self.link_attack_freq[k]

                # 2. Decrease cooldown timer, collect links just finished cooldown
                to_reattack = []
                for k in list(self.link_cooldown_timer):
                    self.link_cooldown_timer[k] -= 1
                    if self.link_cooldown_timer[k] <= 0:
                        to_reattack.append(k)
                # Remove links that have finished cooldown
                for k in to_reattack:
                    del self.link_cooldown_timer[k]

                # 3. Only attack the ground-leader link
                leader = self.env.formation.get_leader()
                link = (min(ground, leader), max(ground, leader))
                busy_links = set(self.link_attack_timer.keys()) | set(self.link_cooldown_timer.keys())
                if link not in busy_links:
                    self.link_attack_timer[link] = self.ATTACK_DURATION
                    self.link_attack_freq[link] = self.env.comm.frequencies[link[0]]

                for link in self.link_attack_timer:
                    self.attack_matrix[link] = True

        else:
            return

    def _select_link_target_random(self, exclude=None):
        # Only select one target
        exclude = exclude or []
        leader = self.env.formation.get_leader()
        ground = self.ground
        candidates = []
        if (ground, leader) not in exclude:
            candidates.append((ground, leader))
        for i in range(self.num_nodes):
            if i != leader and i != ground and (leader, i) not in exclude:
                candidates.append((leader, i))
        if not candidates:
            return []
        idx = np.random.choice(len(candidates), 1, replace=False)
        return [candidates[idx[0]]]

    def _select_node_target_random(self, exclude=None):
        # Only select one target
        exclude = exclude or []
        candidates = [i for i in range(self.num_nodes) if i != self.ground and i not in exclude]
        if not candidates:
            return []
        idx = np.random.choice(len(candidates), 1, replace=False)
        return [candidates[idx[0]]]

    def is_link_attacked(self, i, j):
        key = (min(i, j), max(i, j))
        attack_freq = self.link_attack_freq.get(key, None)
        freq_i = self.env.comm.frequencies[i]
        freq_j = self.env.comm.frequencies[j]
        return key in self.link_attack_timer and self.link_attack_timer[key] > 0 and freq_i == attack_freq and freq_j == attack_freq

    def is_link_attack_effective(self, i, j):
        key = (min(i, j), max(i, j))
        if not self.is_link_attacked(i, j):
            return False
        graph = self.env.comm.graph
        if not graph.has_edge(*key):
            return False
        # Use a temporary copy to avoid polluting the original graph
        temp_graph = graph.copy()
        temp_graph.remove_edge(*key)
        has_path = nx.has_path(temp_graph, *key)
        return not has_path

    def is_node_attack_effective(self, i):
        attack_freq = self.node_attack_freq.get(i, None)
        freq = self.env.comm.frequencies[i]
        return i in self.node_attack_timer and self.node_attack_timer[i] > 0 and freq == attack_freq

    def get_penalty(self):
        effective_links = sum(
            self.is_link_attack_effective(i, j)
            for (i, j) in self.link_attack_timer
        )
        effective_nodes = sum(
            self.is_node_attack_effective(i)
            for i in self.node_attack_timer
        )
        return -effective_links * 0.5 - effective_nodes * 1.0

    def get_penalty_per_agent(self):
        """
        Return the attack penalty (negative value) for each UAV:
        - If the leader node is attacked, all UAVs are penalized by 1
        - If the ground-leader link is attacked, all UAVs are penalized by 0.5
        - Otherwise: node -1, both ends of the link -0.5
        """
        penalties = np.zeros(self.num_nodes - 1)  # Exclude the ground station
        leader = self.env.formation.get_leader()
        ground = self.ground

        # 1. First check if the leader node is attacked
        for i in self.node_attack_timer:
            if self.is_node_attack_effective(i):
                if i == leader and i != ground:
                    penalties[:] -= 1.0
                    # print("[Penalty] Leader node attacked, all penalized by 1")
                    return penalties

        # 2. Check if the ground-leader link is attacked
        for (i, j) in self.link_attack_timer:
            if self.is_link_attack_effective(i, j):
                if (i == ground and j == leader) or (j == ground and i == leader):
                    penalties[:] -= 0.5
                    # print("[Penalty] Ground-leader link attacked, all penalized by 0.5")
                    return penalties

        # 3. Otherwise, distribute penalty normally
        for i in self.node_attack_timer:
            if self.is_node_attack_effective(i):
                if i < self.num_nodes - 1:
                    penalties[i] -= 1.0

        for (i, j) in self.link_attack_timer:
            if self.is_link_attack_effective(i, j):
                if i < self.num_nodes - 1:
                    penalties[i] -= 0.5
                if j < self.num_nodes - 1:
                    penalties[j] -= 0.5
        return penalties