class BaseAgent:
    def select_action(self, obs):
        raise NotImplementedError
    def decode_action(self, action_idx):
        raise NotImplementedError
    def store(self, obs, action, reward, next_obs, done):
        raise NotImplementedError
    def update(self):
        raise NotImplementedError