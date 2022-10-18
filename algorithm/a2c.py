from common import TrainableAlgorithm
from stable_baselines3.a2c import A2C as StableA2C

class A2C(TrainableAlgorithm, StableA2C):
  """A Trainable extension to A2C"""
  def train(self) -> None:
    self.logger.record("rewards/environment", self.rollout_buffer.rewards.copy()) 
    super(A2C, self).train()
