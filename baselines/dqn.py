from common import TrainableAlgorithm
from stable_baselines3.dqn import DQN as StableDQN

class DQN(TrainableAlgorithm, StableDQN):
  """A Trainable extension to DQN"""
  def __init__(self, **kwargs): super(DQN, self).__init__(**kwargs); self.n_steps, self.prev_steps = None, 0

  def train(self, **kwargs) -> None:
    self.n_steps = (self.num_timesteps - self.prev_steps) / self.env.num_envs
    self.logger.record("rewards/environment", self.replay_buffer.rewards.copy()) 
    super(DQN, self).train(**kwargs); self.prev_steps = self.num_timesteps
