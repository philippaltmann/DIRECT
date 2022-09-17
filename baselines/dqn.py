from util import TrainableAlgorithm
from stable_baselines3.dqn import DQN as StableDQN

class DQN(TrainableAlgorithm, StableDQN):
  """A Trainable extension to DQN"""
  def __init__(self, **kwargs): super(DQN, self).__init__(**kwargs); self.n_steps = None

  def train(self, **kwargs) -> None:
    if not self.n_steps: self.n_steps = self.num_timesteps
    self.logger.record("rewards/environment", self.replay_buffer.rewards.copy()) 
    super(DQN, self).train(**kwargs)
