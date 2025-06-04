import numpy as np

class FormationSystem:
    def __init__(self, env, config):
        self.env = env
        self.num_uavs = config.NUM_UAVS
        self.num_nodes = self.num_uavs + 1  # Including ground station
        self.center = np.array(config.PATROL_CENTER)
        self.radius = config.PATROL_RADIUS
        self.height = config.PATROL_HEIGHT
        self.patrol_speed = config.PATROL_SPEED
        self.max_speed = config.MAX_SPEED
        self.angles = np.linspace(0, 2*np.pi, self.num_uavs, endpoint=False)
        self.ideal_angles = np.linspace(0, 2*np.pi, self.num_uavs, endpoint=False)
        self.positions = np.zeros((self.num_nodes, 3))
        self.positions[-1] = np.array([self.center[0], self.center[1], self.center[2]])  # Ground station at the center
        self.headings = np.zeros(self.num_nodes)
        self.speeds = np.ones(self.num_nodes) * self.patrol_speed
        # Randomly select leader (not ground station)
        self.leader = np.random.randint(0, self.num_uavs)
        self.recovering = np.zeros(self.num_uavs, dtype=bool)  # Record recovery status

    def reset(self):
        self.angles = np.linspace(0, 2*np.pi, self.num_uavs, endpoint=False)
        self.ideal_angles = np.copy(self.angles)
        self.positions = np.zeros((self.num_nodes, 3))
        for i in range(self.num_uavs):
            self.positions[i] = [
                self.center[0] + self.radius * np.cos(self.angles[i]),
                self.center[1] + self.radius * np.sin(self.angles[i]),
                self.height
            ]
        self.positions[-1] = np.array([self.center[0], self.center[1], self.center[2]])
        self.headings = np.zeros(self.num_nodes)
        self.headings[:self.num_uavs] = self.angles + np.pi/2
        self.speeds = np.ones(self.num_nodes) * self.patrol_speed
        self.speeds[-1] = 0
        # Randomly select leader (not ground station) at reset
        self.leader = np.random.randint(0, self.num_uavs)
        self.recovering = np.zeros(self.num_uavs, dtype=bool)

    def _update_positions(self):
        comm = self.env.comm
        for i in range(self.num_uavs):  # Only update UAVs
            if comm.command_received[i] or comm.command_recovered[i]:
                x = self.center[0] + self.radius * np.cos(self.angles[i])
                y = self.center[1] + self.radius * np.sin(self.angles[i])
                self.positions[i] = [x, y, self.height]
                self.headings[i] = self.angles[i] + np.pi/2

    def step(self):
        comm = self.env.comm
        self.ideal_angles += self.patrol_speed / self.radius * self.env.dt
        self.ideal_angles = self.ideal_angles % (2 * np.pi)
        for i in range(self.num_uavs):  # Only process UAVs
            # If just recovered communication, mark as recovering
            if comm.command_recovered[i]:
                self.recovering[i] = True

            # As long as command is received or recovering, chase the current ideal point
            if comm.command_received[i] or self.recovering[i]:
                target = np.array([
                    self.center[0] + self.radius * np.cos(self.ideal_angles[i]),
                    self.center[1] + self.radius * np.sin(self.ideal_angles[i]),
                    self.height
                ])
                direction = target - self.positions[i]
                dist = np.linalg.norm(direction)
                if dist > 2 * self.max_speed * self.env.dt:
                    move = direction / dist * min(self.max_speed * self.env.dt, dist)
                    self.positions[i] += move
                    self.headings[i] = np.arctan2(direction[1], direction[0])
                    self.speeds[i] = self.max_speed
                    self.recovering[i] = True  # Stay in recovering status until caught up
                else:
                    self.positions[i] = target
                    self.speeds[i] = self.patrol_speed
                    self.headings[i] = self.ideal_angles[i] + np.pi/2
                    self.recovering[i] = False
                # Update angle
                self.angles[i] += self.patrol_speed / self.radius * self.env.dt
                self.angles[i] = self.angles[i] % (2 * np.pi)
            else:
                # If disconnected, cruise along current heading
                dx = self.patrol_speed * self.env.dt * np.cos(self.headings[i])
                dy = self.patrol_speed * self.env.dt * np.sin(self.headings[i])
                self.positions[i][0] += dx
                self.positions[i][1] += dy
                self.speeds[i] = self.patrol_speed
        self._update_positions()

    def get_leader(self):
        return self.leader

    def get_position(self, idx):
        return self.positions[idx]

    def get_positions(self):
        return self.positions

    def get_state(self, uav_id):
        # Distance to ideal target point (normalized)
        target = np.array([
            self.center[0] + self.radius * np.cos(self.ideal_angles[uav_id]),
            self.center[1] + self.radius * np.sin(self.ideal_angles[uav_id]),
            self.height
        ])
        dist_to_target = np.linalg.norm(self.positions[uav_id] - target) / self.radius

        # Current heading (normalized to [0,1])
        heading = self.headings[uav_id] / (2 * np.pi)

        # Speed (normalized to [0,1])
        speed = self.speeds[uav_id] / self.max_speed

        # Return more info if needed:
        return [dist_to_target, heading, speed]
        # Or just return dist_to_target, heading, speed
    
    def get_formation_score(self):
        """
        Return the formation score, measuring whether all UAVs are evenly distributed on the circle.
        Score is normalized to [0,1], 1 means ideal uniform distribution.
        """
        # Ideal angle intervals
        ideal_angles = np.linspace(0, 2 * np.pi, self.num_uavs, endpoint=False)
        # Calculate the minimum difference between each UAV's current angle and ideal angles
        angle_diffs = []
        for i in range(self.num_uavs):
            diffs = np.abs((self.angles[i] - ideal_angles + np.pi) % (2 * np.pi) - np.pi)
            angle_diffs.append(np.min(diffs))
        # Normalize: maximum possible difference is pi
        mean_diff = np.mean(angle_diffs)
        score = 1.0 - mean_diff / np.pi
        return np.clip(score, 0.0, 1.0)
    
    def get_formation_score_per_agent(self):
        """
        Return the formation score for each UAV, in [0,1], the closer to the target position, the higher the score.
        """
        scores = np.zeros(self.num_uavs)
        for i in range(self.num_uavs):
            pos = self.positions[i]  # Current UAV position
            # Calculate the target formation position (ideal point on the circle)
            target = np.array([
                self.center[0] + self.radius * np.cos(self.ideal_angles[i]),
                self.center[1] + self.radius * np.sin(self.ideal_angles[i]),
                self.height
            ])
            dist = np.linalg.norm(np.array(pos) - target)
            # Normalized score (distance 0 gets 1, max distance gets 0)
            max_dist = self.radius  # Maximum deviation is set as the radius
            score = 1.0 - min(dist / max_dist, 1.0)
            scores[i] = score
        return scores

    def get_deviation(self):
        """
        Return the average distance between all UAVs' current positions and their expected points on the circle, normalized to [0,1].
        Also prints the deviation (normalized) for each UAV.
        """
        deviations = []
        for i in range(self.num_uavs):
            expected_pos = np.array([
                self.center[0] + self.radius * np.cos(self.ideal_angles[i]),
                self.center[1] + self.radius * np.sin(self.ideal_angles[i]),
                self.height
            ])
            dist = np.linalg.norm(self.positions[i] - expected_pos)
            deviations.append(dist)
        mean_dev = np.mean(deviations) / self.radius
        return np.clip(mean_dev, 0.0, 1.0)