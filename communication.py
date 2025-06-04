import numpy as np
import networkx as nx

class CommunicationSystem:
    def __init__(self, env, config):
        self.env = env
        self.num_uavs = config.NUM_UAVS
        self.comm_range = config.COMM_RANGE
        self.num_channels = config.NUM_CHANNELS
        self.num_nodes = self.num_uavs + 1  # +1 for ground station
        self.frequencies = np.zeros(self.num_nodes, dtype=int)
        self.connectivity_matrix = np.zeros((self.num_nodes, self.num_nodes))
        self.graph = nx.Graph()
        # 确保所有节点都在图中
        for i in range(self.num_nodes):
            self.graph.add_node(i)
        self.gcs_connected = False
        self.command_received = np.ones(self.num_nodes, dtype=bool)
        self.command_recovered = np.zeros(self.num_nodes, dtype=bool)
        self.route_table = {}  # target: relay

    def reset(self):
        freq = np.random.randint(0, self.num_channels)
        self.frequencies[:] = freq
        self.command_received[:] = True
        self.command_recovered[:] = False
        self.graph.clear()
        for i in range(self.num_nodes):
            self.graph.add_node(i)
        self.update()

    def update(self):
        prev_command_received = self.command_received.copy()
        self._update_graph()
        self._update_gcs_connection()
        self._update_command_propagation()
        # 只在刚恢复时为True
        self.command_recovered[:] = np.logical_and(~prev_command_received, self.command_received)

    def _update_graph(self):
        self.graph.clear()
        for i in range(self.num_nodes):
            self.graph.add_node(i)
        positions = self.env.formation.get_positions()
        leader = self.env.formation.get_leader()
        ground = self.num_nodes - 1  # 假设地面站编号为最后一个

        # 清空连接矩阵
        self.connectivity_matrix[:] = 0

        # 地面站与leader（如果满足条件）连边
        if self._can_communicate(ground, leader, positions):
            self.graph.add_edge(ground, leader)
            self.connectivity_matrix[ground, leader] = 1
            self.connectivity_matrix[leader, ground] = 1

        # leader与每个成员（如果满足条件）连边，成员之间不连边
        for i in range(self.num_nodes):
            if i != leader and i != ground:
                if self._can_communicate(leader, i, positions):
                    self.graph.add_edge(leader, i)
                    self.connectivity_matrix[leader, i] = 1
                    self.connectivity_matrix[i, leader] = 1

        # === 新增：根据 route_table 添加 relay-成员边 ===
        for target, relay in self.route_table.items():
            if relay is not None and relay != target:
                if self._can_communicate(relay, target, positions):
                    self.graph.add_edge(relay, target)
                    self.connectivity_matrix[relay, target] = 1
                    self.connectivity_matrix[target, relay] = 1

    def _can_communicate(self, i, j, positions):
        if self.env.attack.is_link_attack_effective(i, j):
            return False
        if self.env.attack.is_node_attack_effective(i) or self.env.attack.is_node_attack_effective(j):
            return False
        if self.frequencies[i] != self.frequencies[j]:
            return False
        dist = np.linalg.norm(positions[i] - positions[j])
        return dist <= self.comm_range

    def _update_gcs_connection(self):
        leader = self.env.formation.get_leader()
        leader_pos = self.env.formation.get_position(leader)
        ground = self.num_nodes - 1
        # leader被节点攻击或链路被攻击都无法接收GCS命令
        self.gcs_connected = (
            np.linalg.norm(leader_pos - self.env.gcs_position) <= self.env.gcs_range
            and not self.env.attack.is_node_attack_effective(leader)
            and not self.env.attack.is_link_attack_effective(ground, leader)
        )

    def _update_command_propagation(self):
        leader = self.env.formation.get_leader()
        ground = self.num_nodes - 1
        self.command_received[:] = False
        if self.gcs_connected:
            self.command_received[leader] = True

        # 遍历所有成员
        for i in range(self.num_nodes):
            if i != leader and i != ground:
                relay = self.route_table.get(i, None)
                if relay is None:
                    # 直连
                    if self.graph.has_edge(leader, i):
                        self.command_received[i] = self.command_received[leader]
                else:
                    # 通过relay
                    if self.graph.has_edge(leader, relay) and self.graph.has_edge(relay, i):
                        if self.command_received[leader]:
                            self.command_received[relay] = True
                        if self.command_received[relay]:
                            self.command_received[i] = True

    def set_route(self, target, relay):
        """
        设置leader到target的通信路由。
        relay为None表示leader与target直连，否则通过relay中继。
        """
        if relay is None or relay == target:
            self.route_table[target] = None
        else:
            self.route_table[target] = relay
    
    def get_network_connectivity(self):
        """
        Calculate the network connectivity score, combining GCS-cluster and leader-member, normalized to [0,1].
        """
        # 1. GCS connectivity
        gcs = 1.0 if self.gcs_connected else 0.0

        # 2. Leader-member connectivity (whether each member actually receives the command)
        leader = self.env.formation.get_leader()
        reachable = []
        for i in range(self.num_nodes):
            if i == leader:
                continue
            # Use whether the command is actually received
            reachable.append(1.0 if self.command_received[i] else 0.0)
        leader_member = np.mean(reachable) if reachable else 0.0

        # Weighted sum of the two parts
        return 0.5 * gcs + 0.5 * leader_member
    
    def get_network_connectivity_per_agent(self):
        """
        Return the communication connectivity score for each UAV, in [0,1].
        - For the leader: whether it is connected to the GCS
        - For members: whether the command is received
        """
        leader = self.env.formation.get_leader()
        comm_scores = np.zeros(self.num_uavs)
        for i in range(self.num_uavs):
            if i == leader:
                # Leader's connectivity is determined by GCS connection
                comm_scores[i] = 1.0 if self.gcs_connected else 0.0
            else:
                # Member's connectivity is determined by whether the command is received
                comm_scores[i] = 1.0 if self.command_received[i] else 0.0
        return comm_scores

    def get_state(self, uav_id):
        """
        Return the communication-related state for a single UAV (whether the command is received, GCS connection, normalized current frequency)
        """
        leader = self.env.formation.get_leader()
        is_leader = 1.0 if uav_id == leader else 0.0
        # Whether the command is received
        received = 1.0 if self.command_received[uav_id] else 0.0
        gcs_conn = 1.0 if (uav_id == leader and self.gcs_connected) else 0.0
        freq = self.frequencies[uav_id] / max(1, self.num_channels - 1)
        # Optional: whether it is under attack
        is_attacked = 1.0 if self.env.attack.is_node_attack_effective(uav_id) else 0.0
        return [is_leader, received, gcs_conn, freq, is_attacked]